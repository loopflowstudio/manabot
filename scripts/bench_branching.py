#!/usr/bin/env python3
"""Run and verify the search-branching contract-v1 benchmark.

Invoke only through uv, for example:

    uv run scripts/bench_branching.py run
    uv run scripts/bench_branching.py verify
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/benchmarks/search-branching-contract-v1.md"
FULL_CLONE_DRIVER = "full_clone/current_game_v1"
CLONE_PLUS_UNDO_DRIVER = "compact_clone_undo/current_game_v1"

# Each candidate lands its own artifact pair. Both pairs are excluded from
# source_sha256 so that generating one candidate's evidence does not invalidate
# the other's source digest -- the two must be comparable on one tree.
ARTIFACTS: dict[str, tuple[Path, Path]] = {
    FULL_CLONE_DRIVER: (
        ROOT / "experiments/data/w2-182-search-branching-v1.json",
        ROOT / "experiments/w2-182-search-branching-v1.md",
    ),
    CLONE_PLUS_UNDO_DRIVER: (
        ROOT / "experiments/data/w2-198-compact-clone-undo-v1.json",
        ROOT / "experiments/w2-198-compact-clone-undo-v1.md",
    ),
}

DEFAULT_RAW = ARTIFACTS[FULL_CLONE_DRIVER][0]
DEFAULT_REPORT = ARTIFACTS[FULL_CLONE_DRIVER][1]
BINARY = ROOT / "managym/target/release/branching_bench"
SCHEMA = "manabot.search-branching.result.v1"
WHOLE_CELLS = {
    "flat-single-64-v1",
    "flat-saturated-64-v1",
    "retained-single-8-v1",
    "retained-saturated-16-v1",
}
ALL_CELLS = {"step-v1", "clone-v1", *WHOLE_CELLS}
POLL_SECONDS = 0.005
READY_TIMEOUT_SECONDS = 120.0
WORKER_TIMEOUT_SECONDS = 300.0
TERMINATE_GRACE_SECONDS = 2.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


DRIVER_COUNTER_FIELDS = {
    "allocation_count",
    "allocation_bytes",
    "journal_bytes",
    "cow_bytes",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hash(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    return hashlib.sha256(canonical_json(unhashed)).hexdigest()


def deterministic_repeat_checksum(
    cell_id: str, root_seed: int, workers: list[dict[str, Any]]
) -> str:
    """Hash only logical workload identity and ordered native checksums."""
    logical = {
        "cell_id": cell_id,
        "root_seed": root_seed,
        "workers": [
            {
                "worker_index": worker["seed_path"]["worker_index"],
                "worker_seed": worker["seed_path"]["worker_seed"],
                "result_checksum": worker["metrics"]["result_checksum"],
            }
            for worker in sorted(
                workers, key=lambda value: value["seed_path"]["worker_index"]
            )
        ],
    }
    return hashlib.sha256(canonical_json(logical)).hexdigest()


def expected_worker_simulations(dimensions: dict[str, Any], action_count: int) -> int:
    """Contract-v1 W * A * L * R count for one worker and root seed."""
    return (
        dimensions["actors_per_worker"]
        * action_count
        * dimensions["worlds"]
        * dimensions["rollouts_per_world"]
    )


def verify_worker_contract_shape(
    cell: dict[str, Any], group: dict[str, Any], driver: str
) -> None:
    """Reject results that improved a driver by changing its logical work."""
    dimensions = cell["dimensions"]
    workers = group["workers"]
    expected_indices = set(range(dimensions["workers"]))
    actual_indices = {worker["seed_path"]["worker_index"] for worker in workers}
    if actual_indices != expected_indices:
        raise RuntimeError(f"worker topology mismatch for {cell['id']}")

    for worker in workers:
        if worker["driver"] != driver:
            raise RuntimeError(f"driver mismatch for {cell['id']}")
        if worker["workload_id"] != cell["id"]:
            raise RuntimeError(f"workload identity mismatch for {cell['id']}")
        if worker["shape"] != dimensions["shape"]:
            raise RuntimeError(f"workload shape mismatch for {cell['id']}")
        if worker["root_seed"] != group["root_seed"]:
            raise RuntimeError(f"root seed mismatch for {cell['id']}")
        if worker["fixture"]["id"] != dimensions["fixture"]:
            raise RuntimeError(f"fixture mismatch for {cell['id']}")

        metrics = worker["metrics"]
        missing_counters = DRIVER_COUNTER_FIELDS - metrics.keys()
        if missing_counters:
            raise RuntimeError(
                f"missing driver counters for {cell['id']}: {sorted(missing_counters)}"
            )
        if (
            all(metrics[field] is None for field in DRIVER_COUNTER_FIELDS)
            and not metrics["unsupported_counters_reason"]
        ):
            raise RuntimeError(f"unsupported counters need a reason for {cell['id']}")

        if cell["id"] in WHOLE_CELLS:
            expected = expected_worker_simulations(
                dimensions, worker["fixture"]["action_count"]
            )
            if metrics["simulations"] != expected:
                raise RuntimeError(
                    f"W*A*L*R simulation mismatch for {cell['id']}: "
                    f"{metrics['simulations']} != {expected}"
                )
        elif dimensions["shape"] == "step":
            if metrics["transitions"] != dimensions["measured_count"]:
                raise RuntimeError(f"step count mismatch for {cell['id']}")
        elif dimensions["shape"] == "clone_latency":
            if len(metrics["clone_latency_ns"]) != dimensions["measured_count"]:
                raise RuntimeError(f"clone sample count mismatch for {cell['id']}")


def percentile(values: list[float | int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def command_output(command: list[str], *, required: bool = True) -> str | None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if required:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return None


def source_sha256() -> str:
    excluded_parts = {
        ".git",
        ".lf",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "htmlcov",
        "target",
        "node_modules",
        "scratch",
    }
    excluded_files = {
        path.resolve() for pair in ARTIFACTS.values() for path in pair
    }
    digest = hashlib.sha256()
    files: list[Path] = []
    for directory, names, filenames in os.walk(ROOT):
        names[:] = sorted(name for name in names if name not in excluded_parts)
        base = Path(directory)
        files.extend(
            base / filename
            for filename in sorted(filenames)
            if filename != ".DS_Store"
            and Path(filename).suffix not in {".dylib", ".pyc", ".so"}
        )
    for path in sorted(files):
        if path.resolve() not in excluded_files:
            relative = path.relative_to(ROOT).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def build_binary() -> dict[str, Any]:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            "managym/Cargo.toml",
            "--bin",
            "branching_bench",
        ],
        cwd=ROOT,
        check=True,
    )
    return {
        "profile": "release",
        "allocator": "system",
        "threads_per_worker": 1,
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "uv": command_output(["uv", "--version"]),
        "python": command_output(["uv", "run", "python", "--version"]),
        "binary": str(BINARY.relative_to(ROOT)),
    }


def native_json(arguments: list[str]) -> dict[str, Any]:
    output = command_output([str(BINARY), *arguments])
    assert output is not None
    return json.loads(output)


def hardware_metadata(oversubscribed: bool) -> dict[str, Any]:
    cpu_model = command_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], required=False
    )
    if not cpu_model:
        cpu_model = platform.processor() or None
    power = command_output(["pmset", "-g", "batt"], required=False)
    thermal = command_output(["pmset", "-g", "therm"], required=False)
    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_memory_bytes": psutil.virtual_memory().total,
        "power_mode": power,
        "power_mode_unavailable_reason": None if power else "pmset unavailable",
        "thermal_state": thermal,
        "thermal_state_unavailable_reason": None if thermal else "pmset unavailable",
        "oversubscribed": oversubscribed,
        "rss_method": "sum(psutil.Process(worker_pid).memory_info().rss)",
        "rss_poll_interval_seconds": POLL_SECONDS,
        "rss_shared_page_note": "Summed process RSS may double-count shared pages; 5 ms sampling may miss shorter spikes.",
    }


def smoke_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = dict(spec)
    if value["shape"] == "step":
        value.update(warmup_count=20, measured_count=100)
    elif value["shape"] == "clone_latency":
        value.update(warmup_count=10, measured_count=100)
    elif value["shape"] == "sequential":
        value.update(workers=min(value["workers"], 2), worlds=1, rollouts_per_world=1)
    else:
        value.update(actors_per_worker=min(value["actors_per_worker"], 2), worlds=1)
    return value


def read_worker_result(process: subprocess.Popen[str]) -> tuple[str, str]:
    assert process.stdout is not None
    assert process.stderr is not None
    result_line = process.stdout.readline()
    remainder = process.stdout.read()
    error = process.stderr.read()
    return result_line + remainder, error


def read_ready_record(
    process: subprocess.Popen[str], timeout_seconds: float
) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout_seconds):
            raise RuntimeError(
                f"worker {process.pid} missed the ready barrier after "
                f"{timeout_seconds:.3f}s"
            )
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise RuntimeError(
            f"worker {process.pid} exited before emitting a ready record "
            f"(returncode={process.poll()})"
        )
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"worker {process.pid} emitted malformed ready JSON: {line!r}"
        ) from error
    if not isinstance(record, dict):
        raise RuntimeError(f"worker {process.pid} emitted a non-object ready record")
    return record


def terminate_process_group(
    process_group_id: int | None, processes: list[subprocess.Popen[str]]
) -> None:
    alive = [process for process in processes if process.poll() is None]
    if process_group_id is not None and alive:
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.01)
    for process in processes:
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def release_barrier(processes: list[subprocess.Popen[str]]) -> None:
    try:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"worker {process.pid} exited before barrier release "
                    f"(returncode={process.returncode})"
                )
            assert process.stdin is not None
            process.stdin.write("x")
            process.stdin.flush()
            process.stdin.close()
            process.stdin = None
    except (BrokenPipeError, OSError, ValueError) as error:
        raise RuntimeError("worker start barrier failed") from error


def sample_process_rss(
    processes: list[subprocess.Popen[str]],
    *,
    sequence: int,
    phase: str,
    offset_seconds: float,
) -> dict[str, Any]:
    worker_samples: list[dict[str, Any]] = []
    for process in processes:
        returncode = process.poll()
        rss_bytes = 0
        state = "exited"
        if returncode is None:
            state = "running"
            try:
                rss_bytes = psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                try:
                    returncode = process.wait(timeout=0.01)
                except subprocess.TimeoutExpired as timeout_error:
                    raise RuntimeError(
                        f"RSS sampler lost live worker {process.pid}"
                    ) from timeout_error
                state = "exited"
            except psutil.AccessDenied as error:
                raise RuntimeError(
                    f"RSS sampler cannot inspect live worker {process.pid}"
                ) from error
            except psutil.Error as error:
                raise RuntimeError(
                    f"RSS sampler failed for live worker {process.pid}: {error}"
                ) from error
        worker_samples.append(
            {
                "pid": process.pid,
                "state": state,
                "returncode": returncode,
                "rss_bytes": rss_bytes,
            }
        )
    return {
        "sequence": sequence,
        "captured_at": utc_now(),
        "offset_seconds": offset_seconds,
        "phase": phase,
        "rss_bytes": sum(sample["rss_bytes"] for sample in worker_samples),
        "workers": worker_samples,
    }


def run_group(
    spec: dict[str, Any], root_seed: int, max_steps: int, driver: str
) -> dict[str, Any]:
    repeat_started_at = utc_now()
    processes: list[subprocess.Popen[str]] = []
    process_group_id: int | None = None
    commands: list[list[str]] = []
    executor: ThreadPoolExecutor | None = None
    try:
        for worker in range(spec["workers"]):
            request = {
                "schema_version": 1,
                "driver": driver,
                "workload": spec,
                "root_seed": root_seed,
                "worker": worker,
                "max_steps": max_steps,
            }
            command = [
                str(BINARY),
                "--request-json",
                json.dumps(request, separators=(",", ":")),
            ]
            commands.append(command)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                process_group=0 if process_group_id is None else process_group_id,
            )
            processes.append(process)
            actual_process_group = os.getpgid(process.pid)
            if process_group_id is None:
                process_group_id = actual_process_group
                if process_group_id != process.pid:
                    raise RuntimeError(
                        f"worker {process.pid} did not lead its fresh process group"
                    )
            elif actual_process_group != process_group_id:
                raise RuntimeError(
                    f"worker {process.pid} joined process group {actual_process_group}, "
                    f"expected {process_group_id}"
                )

        assert process_group_id is not None
        if process_group_id == os.getpgrp():
            raise RuntimeError(
                "worker process group is not isolated from the orchestrator"
            )

        ready: list[dict[str, Any]] = []
        ready_deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        for process, command in zip(processes, commands, strict=True):
            remaining = ready_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("worker ready barrier timed out")
            record = read_ready_record(process, remaining)
            if record.get("type") != "ready" or record.get("pid") != process.pid:
                raise RuntimeError(f"malformed ready record: {record}")
            if record.get("argv") != command:
                raise RuntimeError(
                    f"worker {process.pid} argv receipt does not match invoked argv"
                )
            if os.getpgid(process.pid) != process_group_id:
                raise RuntimeError(
                    f"worker {process.pid} left process group {process_group_id}"
                )
            record["process_group_id"] = process_group_id
            ready.append(record)

        baseline_sample = sample_process_rss(
            processes,
            sequence=0,
            phase="baseline",
            offset_seconds=0.0,
        )
        if any(sample["state"] != "running" for sample in baseline_sample["workers"]):
            raise RuntimeError("worker exited before the RSS baseline was complete")
        rss_baseline = baseline_sample["rss_bytes"]
        parent_rss = psutil.Process().memory_info().rss
        rss_samples = [baseline_sample]

        executor = ThreadPoolExecutor(max_workers=len(processes))
        readers = [
            executor.submit(read_worker_result, process) for process in processes
        ]
        barrier_started = time.perf_counter()
        release_barrier(processes)
        barrier_released_at = utc_now()
        completion_deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
        next_sample_at = time.perf_counter()
        sequence = 1
        while True:
            now = time.perf_counter()
            complete = all(process.poll() is not None for process in processes)
            rss_samples.append(
                sample_process_rss(
                    processes,
                    sequence=sequence,
                    phase="complete" if complete else "running",
                    offset_seconds=now - barrier_started,
                )
            )
            sequence += 1
            if complete:
                break
            if time.monotonic() >= completion_deadline:
                raise RuntimeError(
                    f"worker group {process_group_id} timed out after "
                    f"{WORKER_TIMEOUT_SECONDS:.1f}s"
                )
            next_sample_at += POLL_SECONDS
            delay = next_sample_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        barrier_wall = time.perf_counter() - barrier_started
        outputs = [reader.result(timeout=TERMINATE_GRACE_SECONDS) for reader in readers]

        workers: list[dict[str, Any]] = []
        for process, (stdout, stderr) in zip(processes, outputs, strict=True):
            if process.returncode != 0:
                raise RuntimeError(
                    f"worker {process.pid} failed ({process.returncode}): {stderr}"
                )
            lines = [line for line in stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise RuntimeError(
                    f"worker {process.pid} returned {len(lines)} result lines"
                )
            try:
                result = json.loads(lines[0])
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"worker {process.pid} returned malformed result JSON"
                ) from error
            if not isinstance(result, dict):
                raise RuntimeError(f"worker {process.pid} returned a non-object result")
            workers.append(result)

        workers.sort(key=lambda value: value["seed_path"]["worker_index"])
        rss_peak = max(sample["rss_bytes"] for sample in rss_samples)
        sample_offsets = [sample["offset_seconds"] for sample in rss_samples]
        sample_gaps = [
            later - earlier
            for earlier, later in zip(sample_offsets, sample_offsets[1:])
        ]
        completed_at = utc_now()
        return {
            "status": "complete",
            "started_at": repeat_started_at,
            "completed_at": completed_at,
            "root_seed": root_seed,
            "process_group_id": process_group_id,
            "ready": ready,
            "commands": commands,
            "workers": workers,
            "barrier_released_at": barrier_released_at,
            "barrier_wall_seconds": barrier_wall,
            "sampler": {
                "status": "complete",
                "interval_seconds": POLL_SECONDS,
                "started_at": barrier_released_at,
                "completed_at": completed_at,
                "sample_count": len(rss_samples),
                "max_observed_gap_seconds": max(sample_gaps, default=0.0),
                "series_start": "pre-barrier-baseline",
                "series_end": "all-workers-exited",
            },
            "rss_baseline_bytes": rss_baseline,
            "rss_peak_bytes": rss_peak,
            "rss_peak_delta_bytes": max(0, rss_peak - rss_baseline),
            "rss_samples": rss_samples,
            "parent_rss_bytes": parent_rss,
            "result_checksum": deterministic_repeat_checksum(
                spec["id"], root_seed, workers
            ),
        }
    except Exception:
        terminate_process_group(process_group_id, processes)
        raise
    finally:
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
                process.stdin = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def summarize_cell(cell: dict[str, Any]) -> dict[str, Any]:
    repeats = cell["repeats"]
    metrics = [worker["metrics"] for repeat in repeats for worker in repeat["workers"]]
    simulations = sum(metric["simulations"] for metric in metrics)
    transitions = sum(metric["transitions"] for metric in metrics)
    barrier_seconds = sum(repeat["barrier_wall_seconds"] for repeat in repeats)
    root_latencies = [
        max(worker["metrics"]["elapsed_seconds"] for worker in repeat["workers"])
        for repeat in repeats
    ]
    caps = sum(metric["cap_hits"] for metric in metrics)

    def counter_total(name: str) -> int | None:
        """Sum a driver counter, or None if the driver does not support it.

        A counter the driver cannot measure stays null with a reason rather
        than being reported as a zero it did not observe.
        """
        values = [metric[name] for metric in metrics]
        if any(value is None for value in values):
            return None
        return sum(values)

    return {
        "simulations": simulations,
        "transitions": transitions,
        "barrier_wall_seconds": barrier_seconds,
        "simulations_per_second": simulations / barrier_seconds
        if barrier_seconds
        else 0.0,
        "transitions_per_second": transitions / barrier_seconds
        if barrier_seconds
        else 0.0,
        "root_latency_seconds_p50": percentile(root_latencies, 0.50),
        "root_latency_seconds_p95": percentile(root_latencies, 0.95),
        "root_latency_seconds_p99": percentile(root_latencies, 0.99),
        "rss_peak_bytes": max(repeat["rss_peak_bytes"] for repeat in repeats),
        "rss_peak_delta_bytes": max(
            repeat["rss_peak_delta_bytes"] for repeat in repeats
        ),
        "max_live_states": max(metric["max_live_states"] for metric in metrics),
        "eager_forks": sum(metric["eager_forks"] for metric in metrics),
        "checkpoint_copies": sum(metric["checkpoint_copies"] for metric in metrics),
        "fork_seconds": sum(metric["fork_seconds"] for metric in metrics),
        "mark_seconds": sum(metric["mark_seconds"] for metric in metrics),
        "rollback_seconds": sum(metric["rollback_seconds"] for metric in metrics),
        "journal_peak_bytes": max(
            (metric["journal_bytes"] for metric in metrics), default=None
        )
        if all(metric["journal_bytes"] is not None for metric in metrics)
        else None,
        "journal_peak_entries": max(
            (metric["journal_peak_entries"] for metric in metrics), default=None
        )
        if all(metric["journal_peak_entries"] is not None for metric in metrics)
        else None,
        "journal_marks": counter_total("journal_marks"),
        "journal_commits": counter_total("journal_commits"),
        "journal_rollbacks": counter_total("journal_rollbacks"),
        "allocation_count": counter_total("allocation_count"),
        "allocation_bytes": counter_total("allocation_bytes"),
        "cow_bytes": counter_total("cow_bytes"),
        "unsupported_counters_reason": metrics[0]["unsupported_counters_reason"],
        "cap_rate": caps / simulations if simulations else 0.0,
        "repeat_checksums": [repeat["result_checksum"] for repeat in repeats],
    }


def summarize_diagnostic(cell: dict[str, Any]) -> dict[str, Any]:
    metrics = cell["repeats"][0]["workers"][0]["metrics"]
    latencies = (
        metrics["step_latency_ns"]
        if cell["id"] == "step-v1"
        else metrics["clone_latency_ns"]
    )
    count = len(latencies)
    elapsed = metrics["elapsed_seconds"]
    return {
        "samples": count,
        "samples_per_second": count / elapsed if elapsed else 0.0,
        "latency_ns_p50": percentile(latencies, 0.50),
        "latency_ns_p95": percentile(latencies, 0.95),
        "latency_ns_p99": percentile(latencies, 0.99),
        "resets": metrics["resets"],
        "reset_seconds": metrics["reset_seconds"],
        "allocation_count": metrics["allocation_count"],
        "allocation_bytes": metrics["allocation_bytes"],
        "result_checksum": metrics["result_checksum"],
    }


def run_equivalence(manifest: dict[str, Any], driver: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for fixture in manifest["fixtures"]:
        for seed in manifest["equivalence_seeds"]:
            arguments = [
                "--equivalence",
                fixture,
                "--seed",
                str(seed),
                "--max-steps",
                str(manifest["max_steps"]),
                "--driver",
                driver,
            ]
            first = native_json(arguments)
            second = native_json(arguments)
            first["repeat_match"] = (
                first["replay_checksum"] == second["replay_checksum"]
                and first["replay_final_hash"] == second["replay_final_hash"]
            )
            checks.append(first)
    return {
        "seeds": manifest["equivalence_seeds"],
        "checks": checks,
        "passed": all(check["passed"] and check["repeat_match"] for check in checks),
    }


def run_benchmark(
    profile: str,
    oversubscribed: bool,
    *,
    driver: str,
    started_at: str,
    invoked_argv: list[str],
) -> dict[str, Any]:
    initial_source_sha256 = source_sha256()
    build = build_binary()
    manifest = native_json(["--manifest", "--driver", driver])
    if manifest["driver"] != driver:
        raise RuntimeError("native manifest driver does not match the request")
    if manifest["contract_id"] != "manabot.search-branching.v1":
        raise RuntimeError("native manifest does not implement contract v1")
    physical_cores = psutil.cpu_count(logical=False)
    if profile == "full" and (physical_cores or 0) < 8 and not oversubscribed:
        raise RuntimeError(
            "canonical saturated cell requires >=8 physical cores; pass --oversubscribed for a labeled noncanonical run"
        )

    fixtures = [native_json(["--fixture", fixture]) for fixture in manifest["fixtures"]]
    equivalence = run_equivalence(manifest, driver)
    if not equivalence["passed"]:
        raise RuntimeError("deterministic equivalence gate failed")

    measured_seeds = (
        manifest["measured_seeds"]
        if profile == "full"
        else [manifest["measured_seeds"][0]]
    )
    cells: list[dict[str, Any]] = []
    for original_spec in manifest["workloads"]:
        spec = original_spec if profile == "full" else smoke_spec(original_spec)
        seeds = measured_seeds if spec["id"] in WHOLE_CELLS else [measured_seeds[0]]
        cell = {
            "id": spec["id"],
            "dimensions": spec,
            "warmup": {
                "count": spec["warmup_count"],
                "seed": manifest["warmup_seed"] if spec["id"] in WHOLE_CELLS else None,
                "included_in_measurement": False,
            },
            "repeats": [
                run_group(spec, seed, manifest["max_steps"], driver) for seed in seeds
            ],
        }
        if cell["id"] in WHOLE_CELLS:
            replay = run_group(spec, seeds[0], manifest["max_steps"], driver)
            replay["matches_first_repeat"] = (
                replay["result_checksum"] == cell["repeats"][0]["result_checksum"]
            )
            if not replay["matches_first_repeat"]:
                raise RuntimeError(
                    f"deterministic checksum replay failed for {cell['id']}"
                )
            cell["determinism_replay"] = replay
        cell["summary"] = (
            summarize_cell(cell)
            if cell["id"] in WHOLE_CELLS
            else summarize_diagnostic(cell)
        )
        cells.append(cell)

    task_status = command_output(
        ["lf", "task", "status", "W2-198", "--json"], required=False
    )
    completed_source_sha256 = source_sha256()
    if completed_source_sha256 != initial_source_sha256:
        raise RuntimeError("source tree changed during benchmark execution")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "contract": {
            "id": manifest["contract_id"],
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "run": {
            "started_at": started_at,
            "completed_at": None,
            "timezone": str(datetime.now().astimezone().tzinfo),
            "argv": invoked_argv,
            "cwd": str(ROOT),
            "pid": os.getpid(),
            "source_sha256": initial_source_sha256,
            "driver": manifest["driver"],
            "profile": profile,
            "canonical": profile == "full" and not oversubscribed,
            "status": "complete",
            "loopflow": {
                "task": "W2-198",
                "task_session": "ts_b39e9de2861c44bc8c4234a32c2f6bcf",
                "worktree": str(ROOT),
                "branch": "jack-heart/implement-compact-clone-plus-undo",
                "status_snapshot": json.loads(task_status) if task_status else None,
                "status_unavailable_reason": None
                if task_status
                else "local Loopflow registry unavailable during evidence run",
            },
        },
        "hardware": hardware_metadata(oversubscribed),
        "build": build,
        "manifest": manifest,
        "fixtures": fixtures,
        "equivalence": equivalence,
        "cells": cells,
    }
    payload["run"]["completed_at"] = utc_now()
    payload["artifact_sha256"] = artifact_hash(payload)
    return payload


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bytes_mib(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# Whole-rollout branching evidence: `{payload['run']['driver']}`",
        "",
        f"Contract: `{payload['contract']['id']}` (`{payload['contract']['sha256']}`)",
        f"Driver: `{payload['run']['driver']}`",
        f"Run: `{payload['run']['started_at']}`; canonical: `{str(payload['run']['canonical']).lower()}`",
        "Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.",
        "",
        "## Primary whole-rollout evidence",
        "",
        "| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        if cell["id"] not in WHOLE_CELLS:
            continue
        summary = cell["summary"]
        lines.append(
            f"| `{cell['id']}` | {summary['simulations_per_second']:.1f} | "
            f"{summary['transitions_per_second']:.1f} | "
            f"{summary['root_latency_seconds_p50']:.3f}s / {summary['root_latency_seconds_p95']:.3f}s / {summary['root_latency_seconds_p99']:.3f}s | "
            f"{bytes_mib(summary['rss_peak_bytes']):.1f} MiB | "
            f"{bytes_mib(summary['rss_peak_delta_bytes']):.1f} MiB | "
            f"{summary['max_live_states']} | {summary['cap_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            "Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.",
            "",
            "## Branch lifecycle and journal counters",
            "",
            "| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | journal entries | marks / commits / rollbacks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        if cell["id"] not in WHOLE_CELLS:
            continue
        summary = cell["summary"]
        journal_peak = (
            f"{bytes_mib(summary['journal_peak_bytes']):.2f} MiB"
            if summary["journal_peak_bytes"] is not None
            else "null"
        )
        entries = (
            str(summary["journal_peak_entries"])
            if summary["journal_peak_entries"] is not None
            else "null"
        )
        counts = " / ".join(
            str(summary[name]) if summary[name] is not None else "null"
            for name in ("journal_marks", "journal_commits", "journal_rollbacks")
        )
        lines.append(
            f"| `{cell['id']}` | {summary['eager_forks']} | "
            f"{summary['checkpoint_copies']} | {summary['fork_seconds']:.3f}s | "
            f"{summary['mark_seconds']:.3f}s | {summary['rollback_seconds']:.3f}s | "
            f"{journal_peak} | {entries} | {counts} |"
        )
    unsupported = {
        cell["summary"]["unsupported_counters_reason"]
        for cell in payload["cells"]
        if cell["id"] in WHOLE_CELLS
    }
    lines.extend(
        [
            "",
            "`null` counters are unsupported by this driver, not observed zeros: "
            + "; ".join(sorted(reason for reason in unsupported if reason))
            + ".",
            "",
            "## Step and clone diagnostics",
            "",
            "| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        if cell["id"] in WHOLE_CELLS:
            continue
        summary = cell["summary"]
        lines.append(
            f"| `{cell['id']}` | {summary['samples_per_second']:.1f} | "
            f"{summary['latency_ns_p50'] / 1000:.1f}µs / {summary['latency_ns_p95'] / 1000:.1f}µs / {summary['latency_ns_p99'] / 1000:.1f}µs | "
            f"{summary['resets']} | {summary['reset_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Reproduction and evidence",
            "",
            "```bash",
            f"uv run scripts/bench_branching.py run --driver {payload['run']['driver']}",
            f"uv run scripts/bench_branching.py verify --driver {payload['run']['driver']}",
            "```",
            "",
            f"Equivalence: `{str(payload['equivalence']['passed']).lower()}` across {len(payload['equivalence']['checks'])} fixture/seed checks, each replayed twice.",
            "Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.",
            f"Artifact SHA-256: `{payload['artifact_sha256']}`.",
            f"Source SHA-256: `{payload['run']['source_sha256']}`.",
            "",
            "`source_sha256` hashes the whole tree except generated evidence, so this receipt is bound to one exact source state: it must be generated at the final landing tree and landed before `origin/main` moves under it. Any later rebase or source edit, even one touching no benchmark file, invalidates it and requires re-running `run`; re-check `verify` immediately before submitting. Regenerating this receipt is also what repairs the W2-182 baseline, which had been left failing `verify` with a contract-digest mismatch after the witness refactor edited the contract and benchmark without regenerating the artifact.",
            "",
            "The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.",
            "",
            "This artifact measures one candidate. Comparing candidates means comparing two artifacts from the same host and source state, matched by their equal per-cell result checksums. No page-COW representation is implemented, measured, or selected here.",
            "",
        ]
    )
    return "\n".join(lines)


def require_keys(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be an object")
    missing = sorted(keys - value.keys())
    if missing:
        raise RuntimeError(f"missing required {context} fields: {', '.join(missing)}")
    return value


def validate_utc_timestamp(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing required {context}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"invalid {context}: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{context} must be an explicit UTC timestamp")


def validate_argv(value: Any, context: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(argument, str) or not argument for argument in value)
    ):
        raise RuntimeError(f"missing required exact {context}")
    return value


def validate_repeat_evidence(
    repeat: Any,
    *,
    dimensions: dict[str, Any],
    max_steps: int,
    context: str,
    driver: str,
) -> int:
    record = require_keys(
        repeat,
        {
            "status",
            "started_at",
            "completed_at",
            "root_seed",
            "process_group_id",
            "commands",
            "ready",
            "workers",
            "barrier_released_at",
            "barrier_wall_seconds",
            "sampler",
            "rss_baseline_bytes",
            "rss_peak_bytes",
            "rss_peak_delta_bytes",
            "rss_samples",
            "parent_rss_bytes",
            "result_checksum",
        },
        context,
    )
    if record["status"] != "complete":
        raise RuntimeError(f"{context} is not complete")
    validate_utc_timestamp(record["started_at"], f"{context}.started_at")
    validate_utc_timestamp(record["completed_at"], f"{context}.completed_at")
    validate_utc_timestamp(
        record["barrier_released_at"], f"{context}.barrier_released_at"
    )
    if not isinstance(record["root_seed"], int):
        raise RuntimeError(f"missing required {context}.root_seed")
    process_group_id = record["process_group_id"]
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        raise RuntimeError(f"missing required {context}.process_group_id")

    expected_workers = dimensions.get("workers")
    if not isinstance(expected_workers, int) or expected_workers <= 0:
        raise RuntimeError(f"invalid {context} worker dimensions")
    commands = record["commands"]
    ready = record["ready"]
    workers = record["workers"]
    if not isinstance(commands, list) or len(commands) != expected_workers:
        raise RuntimeError(f"missing required exact {context}.commands")
    if not isinstance(ready, list) or len(ready) != expected_workers:
        raise RuntimeError(f"missing required {context}.ready receipts")
    if not isinstance(workers, list) or len(workers) != expected_workers:
        raise RuntimeError(f"missing required {context}.workers")

    ready_pids: list[int] = []
    for worker_index, (command_value, ready_value) in enumerate(
        zip(commands, ready, strict=True)
    ):
        command = validate_argv(command_value, f"{context}.commands[{worker_index}]")
        receipt = require_keys(
            ready_value,
            {"type", "pid", "argv", "process_group_id"},
            f"{context}.ready[{worker_index}]",
        )
        if receipt["type"] != "ready" or not isinstance(receipt["pid"], int):
            raise RuntimeError(f"invalid {context} ready receipt")
        if receipt["process_group_id"] != process_group_id:
            raise RuntimeError(f"{context} ready receipt has the wrong process group")
        if receipt["argv"] != command:
            raise RuntimeError(f"{context} ready argv does not match invoked argv")
        if len(command) != 3 or command[1] != "--request-json":
            raise RuntimeError(f"invalid worker argv in {context}")
        try:
            request = json.loads(command[2])
        except json.JSONDecodeError as error:
            raise RuntimeError(f"malformed worker request argv in {context}") from error
        request = require_keys(
            request,
            {
                "schema_version",
                "driver",
                "workload",
                "root_seed",
                "worker",
                "max_steps",
            },
            f"{context}.worker_request[{worker_index}]",
        )
        if (
            request["schema_version"] != 1
            or request["driver"] != driver
            or request["workload"] != dimensions
            or request["root_seed"] != record["root_seed"]
            or request["worker"] != worker_index
            or request["max_steps"] != max_steps
        ):
            raise RuntimeError(f"worker argv dimensions mismatch in {context}")
        ready_pids.append(receipt["pid"])
    if len(set(ready_pids)) != expected_workers or ready_pids[0] != process_group_id:
        raise RuntimeError(f"{context} is not a fresh worker process group")

    samples = record["rss_samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise RuntimeError(f"missing required complete {context}.rss_samples")
    offsets: list[float] = []
    for sequence, sample_value in enumerate(samples):
        sample = require_keys(
            sample_value,
            {
                "sequence",
                "captured_at",
                "offset_seconds",
                "phase",
                "rss_bytes",
                "workers",
            },
            f"{context}.rss_samples[{sequence}]",
        )
        if sample["sequence"] != sequence:
            raise RuntimeError(f"non-contiguous RSS sequence in {context}")
        validate_utc_timestamp(
            sample["captured_at"], f"{context}.rss_samples[{sequence}].captured_at"
        )
        offset = sample["offset_seconds"]
        if (
            not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or offset < 0
        ):
            raise RuntimeError(f"invalid RSS sample offset in {context}")
        if offsets and offset < offsets[-1]:
            raise RuntimeError(f"non-monotonic RSS sample offsets in {context}")
        offsets.append(float(offset))
        worker_samples = sample["workers"]
        if (
            not isinstance(worker_samples, list)
            or len(worker_samples) != expected_workers
        ):
            raise RuntimeError(f"incomplete aggregate-worker RSS sample in {context}")
        sampled_pids: list[int] = []
        sampled_total = 0
        for worker_sample_value in worker_samples:
            worker_sample = require_keys(
                worker_sample_value,
                {"pid", "state", "returncode", "rss_bytes"},
                f"{context}.rss_samples[{sequence}].worker",
            )
            if worker_sample["state"] not in {"running", "exited"}:
                raise RuntimeError(f"invalid worker state in {context} RSS sample")
            rss_bytes = worker_sample["rss_bytes"]
            if not isinstance(rss_bytes, int) or rss_bytes < 0:
                raise RuntimeError(f"invalid worker RSS in {context}")
            sampled_pids.append(worker_sample["pid"])
            sampled_total += rss_bytes
        if sampled_pids != ready_pids or sample["rss_bytes"] != sampled_total:
            raise RuntimeError(f"aggregate RSS mismatch in {context}")
    if samples[0]["phase"] != "baseline" or samples[0]["offset_seconds"] != 0.0:
        raise RuntimeError(f"missing pre-barrier RSS baseline in {context}")
    if any(worker["state"] != "running" for worker in samples[0]["workers"]):
        raise RuntimeError(f"incomplete pre-barrier worker coverage in {context}")
    if samples[-1]["phase"] != "complete" or any(
        worker["state"] != "exited" or worker["returncode"] != 0
        for worker in samples[-1]["workers"]
    ):
        raise RuntimeError(f"worker failure or incomplete RSS series in {context}")

    sampler = require_keys(
        record["sampler"],
        {
            "status",
            "interval_seconds",
            "started_at",
            "completed_at",
            "sample_count",
            "max_observed_gap_seconds",
            "series_start",
            "series_end",
        },
        f"{context}.sampler",
    )
    validate_utc_timestamp(sampler["started_at"], f"{context}.sampler.started_at")
    validate_utc_timestamp(sampler["completed_at"], f"{context}.sampler.completed_at")
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    if (
        sampler["status"] != "complete"
        or sampler["interval_seconds"] != POLL_SECONDS
        or sampler["sample_count"] != len(samples)
        or sampler["max_observed_gap_seconds"] != max(gaps, default=0.0)
        or sampler["series_start"] != "pre-barrier-baseline"
        or sampler["series_end"] != "all-workers-exited"
    ):
        raise RuntimeError(f"invalid or incomplete RSS sampler receipt in {context}")

    baseline = samples[0]["rss_bytes"]
    peak = max(sample["rss_bytes"] for sample in samples)
    if (
        record["rss_baseline_bytes"] != baseline
        or record["rss_peak_bytes"] != peak
        or record["rss_peak_delta_bytes"] != max(0, peak - baseline)
    ):
        raise RuntimeError(f"RSS summary mismatch in {context}")
    for field in ("barrier_wall_seconds", "parent_rss_bytes"):
        if not isinstance(record[field], (int, float)) or record[field] < 0:
            raise RuntimeError(f"missing required {context}.{field}")
    if offsets[-1] > record["barrier_wall_seconds"]:
        raise RuntimeError(f"RSS series exceeds barrier wall time in {context}")
    return process_group_id


def validate_required_evidence(payload: Any) -> None:
    root = require_keys(
        payload,
        {
            "schema",
            "contract",
            "run",
            "hardware",
            "build",
            "manifest",
            "fixtures",
            "equivalence",
            "cells",
            "artifact_sha256",
        },
        "artifact",
    )
    run = require_keys(
        root["run"],
        {
            "started_at",
            "completed_at",
            "timezone",
            "argv",
            "cwd",
            "pid",
            "source_sha256",
            "driver",
            "profile",
            "canonical",
            "status",
            "loopflow",
        },
        "run",
    )
    validate_utc_timestamp(run["started_at"], "run.started_at")
    validate_utc_timestamp(run["completed_at"], "run.completed_at")
    validate_argv(run["argv"], "run.argv")
    if run["status"] != "complete":
        raise RuntimeError("run status is not complete")
    if not isinstance(run["cwd"], str) or not run["cwd"]:
        raise RuntimeError("missing required run.cwd")
    if not isinstance(run["pid"], int) or run["pid"] <= 0:
        raise RuntimeError("missing required run.pid")
    require_keys(
        run["loopflow"],
        {"task", "task_session", "worktree", "branch"},
        "run.loopflow",
    )
    hardware = require_keys(
        root["hardware"],
        {
            "os",
            "kernel",
            "architecture",
            "cpu_model",
            "physical_cores",
            "logical_cores",
            "total_memory_bytes",
            "power_mode",
            "power_mode_unavailable_reason",
            "thermal_state",
            "thermal_state_unavailable_reason",
            "oversubscribed",
            "rss_method",
            "rss_poll_interval_seconds",
            "rss_shared_page_note",
        },
        "hardware",
    )
    if hardware["rss_poll_interval_seconds"] != POLL_SECONDS:
        raise RuntimeError("RSS polling interval does not match the contract")
    require_keys(
        root["build"],
        {
            "profile",
            "allocator",
            "threads_per_worker",
            "rustc",
            "cargo",
            "uv",
            "python",
            "binary",
        },
        "build",
    )
    manifest = require_keys(root["manifest"], {"max_steps"}, "manifest")
    if not isinstance(root["cells"], list):
        raise RuntimeError("missing required cells")
    process_groups: set[int] = set()
    for cell_index, cell_value in enumerate(root["cells"]):
        cell = require_keys(
            cell_value,
            {"id", "dimensions", "warmup", "repeats", "summary"},
            f"cells[{cell_index}]",
        )
        dimensions = require_keys(
            cell["dimensions"],
            {
                "id",
                "fixture",
                "shape",
                "workers",
                "actors_per_worker",
                "worlds",
                "rollouts_per_world",
                "policy_plies",
                "warmup_count",
                "measured_count",
                "primary_evidence",
            },
            f"cells[{cell_index}].dimensions",
        )
        repeats = cell["repeats"]
        if not isinstance(repeats, list) or not repeats:
            raise RuntimeError(f"missing required repeats for {cell['id']}")
        repeat_records = [
            (f"cell {cell['id']} repeat {repeat_index}", repeat)
            for repeat_index, repeat in enumerate(repeats)
        ]
        if cell["id"] in WHOLE_CELLS:
            require_keys(cell, {"determinism_replay"}, f"cell {cell['id']}")
            repeat_records.append(
                (f"cell {cell['id']} determinism replay", cell["determinism_replay"])
            )
        for context, repeat in repeat_records:
            process_group_id = validate_repeat_evidence(
                repeat,
                dimensions=dimensions,
                max_steps=manifest["max_steps"],
                context=context,
                driver=run["driver"],
            )
            if process_group_id in process_groups:
                raise RuntimeError(
                    f"process group {process_group_id} was reused across cell/repeat records"
                )
            process_groups.add(process_group_id)


def verify(payload: dict[str, Any]) -> None:
    validate_required_evidence(payload)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected schema {payload.get('schema')}")
    if payload["contract"]["id"] != "manabot.search-branching.v1":
        raise RuntimeError("unexpected contract ID")
    if payload["contract"]["sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("contract digest mismatch")
    if payload["artifact_sha256"] != artifact_hash(payload):
        raise RuntimeError("artifact digest mismatch")
    if {cell["id"] for cell in payload["cells"]} != ALL_CELLS:
        raise RuntimeError("required cell set mismatch")
    if not payload["equivalence"]["passed"]:
        raise RuntimeError("equivalence evidence failed")
    if (
        payload["run"]["canonical"]
        and payload["run"]["source_sha256"] != source_sha256()
    ):
        raise RuntimeError("source digest mismatch")
    for cell in payload["cells"]:
        expected = (
            summarize_cell(cell)
            if cell["id"] in WHOLE_CELLS
            else summarize_diagnostic(cell)
        )
        if expected != cell["summary"]:
            raise RuntimeError(f"summary mismatch for {cell['id']}")
        for repeat in cell["repeats"]:
            verify_worker_contract_shape(cell, repeat, payload["run"]["driver"])
            expected_checksum = deterministic_repeat_checksum(
                cell["id"], repeat["root_seed"], repeat["workers"]
            )
            if repeat["result_checksum"] != expected_checksum:
                raise RuntimeError(f"repeat checksum mismatch for {cell['id']}")
        if cell["id"] in WHOLE_CELLS:
            replay = cell.get("determinism_replay")
            if not replay or not replay.get("matches_first_repeat"):
                raise RuntimeError(f"missing deterministic replay for {cell['id']}")
            verify_worker_contract_shape(cell, replay, payload["run"]["driver"])
            expected_checksum = deterministic_repeat_checksum(
                cell["id"], replay["root_seed"], replay["workers"]
            )
            if replay["result_checksum"] != expected_checksum:
                raise RuntimeError(f"replay checksum mismatch for {cell['id']}")


def main() -> int:
    pre_execution_started_at = utc_now()
    invoked_argv = psutil.Process(os.getpid()).cmdline()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    run_parser.add_argument("--oversubscribed", action="store_true")
    run_parser.add_argument(
        "--driver", choices=tuple(ARTIFACTS), default=FULL_CLONE_DRIVER
    )
    run_parser.add_argument("--output", type=Path, default=None)
    run_parser.add_argument("--report", type=Path, default=None)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--driver", choices=tuple(ARTIFACTS), default=FULL_CLONE_DRIVER
    )
    verify_parser.add_argument("--input", type=Path, default=None)
    verify_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    default_raw, default_report = ARTIFACTS[args.driver]
    if args.command == "run":
        output = args.output or default_raw
        report_path = args.report or default_report
    else:
        output = args.input or default_raw
        report_path = args.report or default_report

    if args.command == "run":
        payload = run_benchmark(
            args.profile,
            args.oversubscribed,
            driver=args.driver,
            started_at=pre_execution_started_at,
            invoked_argv=invoked_argv,
        )
        verify(payload)
        atomic_write(output, canonical_json(payload) + b"\n")
        atomic_write(report_path, render_report(payload).encode())
        print(
            json.dumps(
                {
                    "status": "complete",
                    "profile": args.profile,
                    "driver": args.driver,
                    "artifact": str(output),
                    "report": str(report_path),
                    "artifact_sha256": payload["artifact_sha256"],
                }
            )
        )
    else:
        payload = json.loads(output.read_text())
        if payload["run"]["driver"] != args.driver:
            raise RuntimeError(
                f"artifact driver {payload['run']['driver']} does not match --driver {args.driver}"
            )
        verify(payload)
        expected_report = render_report(payload)
        if report_path.read_text() != expected_report:
            raise RuntimeError("generated report is stale")
        print(
            json.dumps(
                {"status": "verified", "artifact_sha256": payload["artifact_sha256"]}
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
