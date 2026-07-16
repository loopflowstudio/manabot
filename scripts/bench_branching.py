#!/usr/bin/env python3
"""Run and verify the search-branching contract-v1 benchmark.

Invoke only through uv, for example:

    uv run scripts/bench_branching.py run-matrix
    uv run scripts/bench_branching.py verify-matrix
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
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, NamedTuple

import psutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/benchmarks/search-branching-contract-v1.md"
PREREG_PATH = ROOT / "docs/benchmarks/dense-page-cow-prereg-v1.md"
DECISION_PATH = ROOT / "docs/benchmarks/search-branching-decision-v1.md"
FULL_CLONE_DRIVER = "full_clone/current_game_v1"
CLONE_PLUS_UNDO_DRIVER = "compact_clone_undo/current_game_v1"
PAGE_COW_DRIVER = "dense_page_cow_undo/event_pages_4k_v1"

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
    PAGE_COW_DRIVER: (
        ROOT / "experiments/data/w2-199-dense-page-cow-undo-v1.json",
        ROOT / "experiments/w2-199-dense-page-cow-undo-v1.md",
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
GIT_TIMEOUT_SECONDS = 5.0
SOURCE_DIGEST_METHOD = "git-ls-tree-sha256-v1"

# These files are embedded into the production Rust crate from outside
# managym/src. The include inventory below makes this list an executable
# closure rather than a best-effort comment.
COMPILE_TIME_SOURCE_INPUTS = (
    "content/semantic/v1/coverage.evidence.json",
    "content/semantic/v1/generated/two_deck.ir.json",
    "content/semantic/v1/two_deck.source.json",
)
SOURCE_DIRECTORY_PATHS = (
    "managym/src",
    "managym/tests",
)
SOURCE_SINGLETON_PATHS = tuple(
    sorted(
        (
            *COMPILE_TIME_SOURCE_INPUTS,
            "docs/benchmarks/dense-page-cow-prereg-v1.md",
            "docs/benchmarks/search-branching-contract-v1.md",
            "managym/Cargo.lock",
            "managym/Cargo.toml",
            "pyproject.toml",
            "scripts/bench_branching.py",
            "tests/bench/test_branching_benchmark.py",
            "uv.lock",
        )
    )
)
SOURCE_PATHS = tuple(sorted((*SOURCE_DIRECTORY_PATHS, *SOURCE_SINGLETON_PATHS)))


class SourceIdentity(NamedTuple):
    sha256: str
    paths: tuple[str, ...]


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


def git_bytes(arguments: list[str], *, root: Path = ROOT, context: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{context} failed: {error}") from error
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"{context} failed ({result.returncode}): {stderr or 'no error output'}"
        )
    return result.stdout


def is_admitted_source_path(relative: str) -> bool:
    return relative in SOURCE_SINGLETON_PATHS or any(
        relative.startswith(f"{directory}/") for directory in SOURCE_DIRECTORY_PATHS
    )


def production_compile_time_inputs(*, root: Path = ROOT) -> tuple[str, ...]:
    source_root = root / "managym/src"
    if not source_root.is_dir():
        raise RuntimeError("production source root managym/src is absent")

    discovered: set[str] = set()
    macro = re.compile(
        r"\b(?:include|include_str|include_bytes)!\s*\(\s*(?P<argument>[^)]*?)\s*\)",
        re.DOTALL,
    )
    literal = re.compile(r'"(?P<path>[^"\n]+)"\s*,?')
    for rust_path in sorted(source_root.rglob("*.rs")):
        contents = rust_path.read_text()
        occurrences = re.findall(
            r"\b(?:include|include_str|include_bytes)\s*!", contents
        )
        matches = list(macro.finditer(contents))
        if len(matches) != len(occurrences):
            relative_rust = rust_path.relative_to(root).as_posix()
            raise RuntimeError(
                f"unresolvable production compile-time include in {relative_rust}"
            )
        for match in matches:
            argument = match.group("argument")
            literal_match = literal.fullmatch(argument)
            if literal_match is None:
                relative_rust = rust_path.relative_to(root).as_posix()
                raise RuntimeError(
                    f"unresolvable production compile-time include in {relative_rust}: "
                    f"{argument.strip()!r}"
                )
            target = (rust_path.parent / literal_match.group("path")).resolve()
            try:
                relative = target.relative_to(root.resolve()).as_posix()
            except ValueError as error:
                raise RuntimeError(
                    f"production compile-time include escapes the repository: {target}"
                ) from error
            if not target.is_file():
                raise RuntimeError(
                    f"production compile-time include target is absent: {relative}"
                )
            if not target.is_relative_to(source_root.resolve()):
                discovered.add(relative)

    cargo_path = root / "managym/Cargo.toml"
    try:
        cargo = tomllib.loads(cargo_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"cannot read managym/Cargo.toml: {error}") from error
    build_value = cargo.get("package", {}).get("build")
    build_path: Path | None
    if build_value is False:
        build_path = None
    elif build_value is None:
        candidate = root / "managym/build.rs"
        build_path = candidate if candidate.exists() else None
    elif isinstance(build_value, str) and build_value:
        build_path = root / "managym" / build_value
    else:
        raise RuntimeError("Cargo package.build must be a path string or false")
    if build_path is not None:
        resolved_build = build_path.resolve()
        if not resolved_build.is_file():
            raise RuntimeError(f"Cargo build script is absent: {build_path}")
        try:
            build_relative = resolved_build.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"Cargo build script escapes the repository: {resolved_build}"
            ) from error
        if not is_admitted_source_path(build_relative):
            raise RuntimeError(
                f"Cargo build script is not admitted to source provenance: {build_relative}"
            )

    actual = tuple(sorted(discovered))
    if actual != COMPILE_TIME_SOURCE_INPUTS:
        raise RuntimeError(
            "production compile-time input inventory mismatch: "
            f"declared={list(COMPILE_TIME_SOURCE_INPUTS)!r}, discovered={list(actual)!r}"
        )
    return actual


def assert_admitted_source_clean(*, root: Path = ROOT) -> None:
    status = git_bytes(
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            *SOURCE_PATHS,
        ],
        root=root,
        context="admitted-source status check",
    )
    if status:
        entries = [
            entry.decode(errors="replace") for entry in status.split(b"\0") if entry
        ]
        raise RuntimeError("admitted source inputs are dirty: " + "; ".join(entries))


def selected_tree_entries(*, root: Path = ROOT) -> tuple[bytes, tuple[str, ...]]:
    entries = git_bytes(
        ["ls-tree", "-r", "--full-tree", "-z", "HEAD", "--", *SOURCE_PATHS],
        root=root,
        context="selected source tree lookup",
    )
    if not entries:
        raise RuntimeError("selected source tree is empty")

    paths: list[str] = []
    for raw_record in entries.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            _mode, object_type, _object_name = metadata.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("malformed selected Git tree entry") from error
        if object_type != b"blob":
            raise RuntimeError(
                "selected source tree contains a non-blob entry: "
                + raw_path.decode(errors="replace")
            )
        paths.append(raw_path.decode())

    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise RuntimeError("selected source tree paths are not unique and sorted")
    path_set = set(paths)
    missing_singletons = sorted(set(SOURCE_SINGLETON_PATHS) - path_set)
    if missing_singletons:
        raise RuntimeError(
            f"required source inputs are absent from HEAD: {missing_singletons!r}"
        )
    empty_directories = [
        directory
        for directory in SOURCE_DIRECTORY_PATHS
        if not any(path.startswith(f"{directory}/") for path in paths)
    ]
    if empty_directories:
        raise RuntimeError(
            f"required source roots are absent from HEAD: {empty_directories!r}"
        )
    return entries, tuple(paths)


def source_identity(*, root: Path = ROOT, check_clean: bool = True) -> SourceIdentity:
    if check_clean:
        assert_admitted_source_clean(root=root)
    production_compile_time_inputs(root=root)
    entries, paths = selected_tree_entries(root=root)
    return SourceIdentity(hashlib.sha256(entries).hexdigest(), paths)


def source_sha256() -> str:
    return source_identity().sha256


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
        "binary_sha256": sha256_file(BINARY),
    }


def measurement_code_revision() -> str:
    revision = command_output(["git", "rev-parse", "HEAD"])
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("cannot resolve exact measurement code revision")
    return revision


def assert_binary_unchanged(build: dict[str, Any]) -> None:
    if not BINARY.is_file() or sha256_file(BINARY) != build["binary_sha256"]:
        raise RuntimeError("release benchmark binary changed during matrix execution")


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

    def simultaneous_peak(name: str) -> int | None:
        """Maximum repeat peak after summing simultaneously live workers."""
        groups = [
            [worker["metrics"][name] for worker in repeat["workers"]]
            for repeat in repeats
        ]
        if any(value is None for group in groups for value in group):
            return None
        return max(sum(group) for group in groups)

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
        "journal_peak_bytes": simultaneous_peak("journal_bytes"),
        "journal_peak_entries": simultaneous_peak("journal_peak_entries"),
        "journal_marks": counter_total("journal_marks"),
        "journal_commits": counter_total("journal_commits"),
        "journal_rollbacks": counter_total("journal_rollbacks"),
        "allocation_count": counter_total("allocation_count"),
        "allocation_bytes": counter_total("allocation_bytes"),
        "cow_bytes": simultaneous_peak("cow_bytes"),
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


def loopflow_metadata() -> dict[str, Any]:
    task_session = os.environ.get("LF_TASK_SESSION_ID")
    branch = command_output(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], required=False
    )
    fallback = {
        "task": None,
        "task_session": task_session,
        "worktree": str(ROOT),
        "branch": branch,
        "status_snapshot": None,
        "status_unavailable_reason": "local Loopflow registry unavailable during evidence run",
    }
    raw_status = command_output(["lf", "status", "--json"], required=False)
    if raw_status is None:
        return fallback
    try:
        status = json.loads(raw_status)
    except json.JSONDecodeError:
        fallback["status_unavailable_reason"] = (
            "local Loopflow registry returned malformed JSON during evidence run"
        )
        return fallback

    candidates: list[dict[str, Any]] = []
    for project_entry in status.get("projects", []):
        if not isinstance(project_entry, dict):
            continue
        for task_entry in project_entry.get("tasks", []):
            if not isinstance(task_entry, dict):
                continue
            runtime = task_entry.get("runtime", {})
            workspace = task_entry.get("reference", {}).get("workspace", {})
            if task_session and runtime.get("session_id") == task_session:
                candidates = [task_entry]
                break
            if workspace.get("worktree") == str(ROOT):
                candidates.append(task_entry)
        if task_session and len(candidates) == 1:
            break
    if len(candidates) != 1:
        fallback["status_unavailable_reason"] = (
            "ambient Loopflow Task could not be identified uniquely during evidence run"
        )
        return fallback

    task_entry = candidates[0]
    task = task_entry.get("task", {})
    runtime = task_entry.get("runtime", {})
    workspace = task_entry.get("reference", {}).get("workspace", {})
    return {
        "task": task.get("identifier"),
        "task_session": runtime.get("session_id") or task_session,
        "worktree": workspace.get("worktree") or str(ROOT),
        "branch": workspace.get("branch") or branch,
        "status_snapshot": {
            "status": runtime.get("status"),
            "reason": runtime.get("reason"),
            "project_session_id": runtime.get("project_session_id"),
        },
        "status_unavailable_reason": None,
    }


def run_benchmark(
    profile: str,
    oversubscribed: bool,
    *,
    driver: str,
    started_at: str,
    invoked_argv: list[str],
    source: SourceIdentity | None = None,
    build_metadata: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    initial_source = source or source_identity()
    build = dict(build_metadata) if build_metadata is not None else build_binary()
    measured_hardware = dict(hardware) if hardware is not None else hardware_metadata(oversubscribed)
    measured_revision = revision or measurement_code_revision()
    assert_binary_unchanged(build)
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

    control_plane = loopflow_metadata()
    completed_source = source_identity()
    if completed_source != initial_source:
        raise RuntimeError("source tree changed during benchmark execution")
    if measurement_code_revision() != measured_revision:
        raise RuntimeError("measurement code revision changed during benchmark execution")
    assert_binary_unchanged(build)
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
            "source_sha256": initial_source.sha256,
            "source_digest_method": SOURCE_DIGEST_METHOD,
            "source_paths": list(initial_source.paths),
            "measurement_code_revision": measured_revision,
            "driver": manifest["driver"],
            "profile": profile,
            "canonical": profile == "full" and not oversubscribed,
            "status": "complete",
            "loopflow": control_plane,
        },
        "hardware": measured_hardware,
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
            "| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | COW peak | journal entries | marks / commits / rollbacks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        cow_peak = (
            f"{bytes_mib(summary['cow_bytes']):.2f} MiB"
            if summary["cow_bytes"] is not None
            else "null"
        )
        lines.append(
            f"| `{cell['id']}` | {summary['eager_forks']} | "
            f"{summary['checkpoint_copies']} | {summary['fork_seconds']:.3f}s | "
            f"{summary['mark_seconds']:.3f}s | {summary['rollback_seconds']:.3f}s | "
            f"{journal_peak} | {cow_peak} | {entries} | {counts} |"
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
            "uv run scripts/bench_branching.py run-matrix",
            "uv run scripts/bench_branching.py verify-matrix",
            "```",
            "",
            f"Equivalence: `{str(payload['equivalence']['passed']).lower()}` across {len(payload['equivalence']['checks'])} fixture/seed checks, each replayed twice.",
            "Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.",
            f"Artifact SHA-256: `{payload['artifact_sha256']}`.",
            f"Source SHA-256: `{payload['run']['source_sha256']}`.",
            f"Measurement revision: `{payload['run']['measurement_code_revision']}`.",
            f"Release binary SHA-256: `{payload['build']['binary_sha256']}`.",
            f"Source method: `{payload['run']['source_digest_method']}` over {len(payload['run']['source_paths'])} tracked paths recorded in the raw receipt.",
            "",
            "`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.",
            "",
            "The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.",
            "",
            "This artifact measures one candidate. The matrix verifier compares all three candidates from the same host, source revision, and release binary and requires equal logical work before the decision record is accepted.",
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


def validate_sha256(value: Any, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"missing required {context}")


def validate_git_revision(value: Any, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError(f"missing required {context}")


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
            "source_digest_method",
            "source_paths",
            "measurement_code_revision",
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
    validate_sha256(run["source_sha256"], "run.source_sha256")
    validate_git_revision(
        run["measurement_code_revision"], "run.measurement_code_revision"
    )
    if run["source_digest_method"] != SOURCE_DIGEST_METHOD:
        raise RuntimeError("source digest method mismatch")
    if (
        not isinstance(run["source_paths"], list)
        or not run["source_paths"]
        or any(not isinstance(path, str) or not path for path in run["source_paths"])
        or run["source_paths"] != sorted(set(run["source_paths"]))
    ):
        raise RuntimeError("missing required exact run.source_paths")
    loopflow = require_keys(
        run["loopflow"],
        {
            "task",
            "task_session",
            "worktree",
            "branch",
            "status_snapshot",
            "status_unavailable_reason",
        },
        "run.loopflow",
    )
    for field in ("task", "task_session", "branch"):
        if loopflow[field] is not None and not isinstance(loopflow[field], str):
            raise RuntimeError(f"invalid run.loopflow.{field}")
    if not isinstance(loopflow["worktree"], str) or not loopflow["worktree"]:
        raise RuntimeError("missing required run.loopflow.worktree")
    if not (
        (
            loopflow["status_snapshot"] is not None
            and loopflow["status_unavailable_reason"] is None
        )
        or (
            loopflow["status_snapshot"] is None
            and isinstance(loopflow["status_unavailable_reason"], str)
            and loopflow["status_unavailable_reason"]
        )
    ):
        raise RuntimeError("invalid Loopflow availability evidence")
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
    build = require_keys(
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
            "binary_sha256",
        },
        "build",
    )
    validate_sha256(build["binary_sha256"], "build.binary_sha256")
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
    current_source = source_identity(check_clean=payload["run"]["canonical"])
    if payload["run"]["source_paths"] != list(current_source.paths):
        raise RuntimeError("source path closure mismatch")
    if (
        payload["run"]["canonical"]
        and payload["run"]["source_sha256"] != current_source.sha256
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


LOGICAL_METRIC_FIELDS = (
    "simulations",
    "transitions",
    "root_decisions",
    "resets",
    "cap_hits",
    "hero_wins",
    "villain_wins",
    "draws",
    "max_live_states",
    "result_checksum",
    "sampled_final_hashes",
)


def matched_matrix_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Project fields that must match across representation candidates."""
    manifest = dict(payload["manifest"])
    manifest.pop("driver", None)

    def group_projection(group: dict[str, Any]) -> dict[str, Any]:
        return {
            "root_seed": group["root_seed"],
            "result_checksum": group["result_checksum"],
            "workers": [
                {
                    "fixture": worker["fixture"],
                    "workload_id": worker["workload_id"],
                    "shape": worker["shape"],
                    "root_seed": worker["root_seed"],
                    "seed_path": worker["seed_path"],
                    "metrics": {
                        field: worker["metrics"][field]
                        for field in LOGICAL_METRIC_FIELDS
                    },
                }
                for worker in group["workers"]
            ],
        }

    return {
        "contract": payload["contract"],
        "source_sha256": payload["run"]["source_sha256"],
        "source_digest_method": payload["run"]["source_digest_method"],
        "source_paths": payload["run"]["source_paths"],
        "measurement_code_revision": payload["run"]["measurement_code_revision"],
        "hardware": payload["hardware"],
        "build": payload["build"],
        "manifest": manifest,
        "fixtures": payload["fixtures"],
        "equivalence": payload["equivalence"],
        "cells": [
            {
                "id": cell["id"],
                "dimensions": cell["dimensions"],
                "warmup": cell["warmup"],
                "repeats": [group_projection(group) for group in cell["repeats"]],
                "determinism_replay": group_projection(cell["determinism_replay"])
                if cell["id"] in WHOLE_CELLS
                else None,
            }
            for cell in payload["cells"]
        ],
    }


def verify_matrix_payloads(payloads: dict[str, dict[str, Any]]) -> None:
    if set(payloads) != set(ARTIFACTS):
        raise RuntimeError("three-driver artifact set mismatch")
    for driver, payload in payloads.items():
        if payload["run"]["driver"] != driver:
            raise RuntimeError(f"artifact driver mismatch for {driver}")
        verify(payload)
    reference = matched_matrix_projection(payloads[FULL_CLONE_DRIVER])
    for driver in (CLONE_PLUS_UNDO_DRIVER, PAGE_COW_DRIVER):
        if matched_matrix_projection(payloads[driver]) != reference:
            raise RuntimeError(f"matched logical/provenance matrix differs for {driver}")


def cell_summaries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cell["id"]: cell["summary"] for cell in payload["cells"]}


def decision_outcome(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = cell_summaries(payloads[FULL_CLONE_DRIVER])
    undo = cell_summaries(payloads[CLONE_PLUS_UNDO_DRIVER])
    page = cell_summaries(payloads[PAGE_COW_DRIVER])
    flat = ("flat-single-64-v1", "flat-saturated-64-v1")
    retained = ("retained-single-8-v1", "retained-saturated-16-v1")

    undo_sequential = all(
        undo[cell]["simulations_per_second"]
        >= 1.20 * full[cell]["simulations_per_second"]
        and undo[cell]["root_latency_seconds_p99"]
        <= 1.10 * full[cell]["root_latency_seconds_p99"]
        and undo[cell]["rss_peak_bytes"] <= 1.10 * full[cell]["rss_peak_bytes"]
        for cell in flat
    )
    page_retained = (
        all(
            page[cell]["simulations_per_second"]
            >= 0.90 * full[cell]["simulations_per_second"]
            and page[cell]["rss_peak_bytes"] <= full[cell]["rss_peak_bytes"]
            for cell in retained
        )
        and page["retained-saturated-16-v1"]["rss_peak_bytes"]
        <= 0.85 * full["retained-saturated-16-v1"]["rss_peak_bytes"]
        and page["retained-saturated-16-v1"]["rss_peak_delta_bytes"]
        <= 0.60 * full["retained-saturated-16-v1"]["rss_peak_delta_bytes"]
        and page["retained-single-8-v1"]["rss_peak_delta_bytes"]
        <= 0.75 * full["retained-single-8-v1"]["rss_peak_delta_bytes"]
    )
    page_general = page_retained and all(
        page[cell]["simulations_per_second"]
        >= 0.90 * full[cell]["simulations_per_second"]
        and page[cell]["root_latency_seconds_p99"]
        <= 1.10 * full[cell]["root_latency_seconds_p99"]
        and page[cell]["rss_peak_bytes"] <= 1.10 * full[cell]["rss_peak_bytes"]
        for cell in flat
    )

    if page_general:
        selection = "dense event-page COW plus undo for all measured branching shapes"
        consequence = (
            "Advance the page driver as the production candidate; W2-207 owns the "
            "subsequent runtime gate. Keep full clone as the conformance reference."
        )
    elif page_retained:
        sequential = (
            "clone plus undo" if undo_sequential else "compact full clone"
        )
        selection = (
            f"workload-specific hybrid: {sequential} for sequential flat rollouts, "
            "dense event-page COW plus undo for retained slots"
        )
        consequence = (
            "Integrate only the workload-specific paths named above; keep full clone "
            "as the conformance reference and do not generalize page storage."
        )
    elif undo_sequential:
        selection = (
            "workload-specific hybrid: clone plus undo for sequential flat rollouts, "
            "compact full clone for retained slots"
        )
        consequence = (
            "Do not integrate page COW. Retain it as a conformance/benchmark driver; "
            "W2-207 may gate the sequential undo path separately."
        )
    else:
        selection = "retain compact full clone as the production default"
        consequence = (
            "Do not integrate either optimized representation. Keep both as "
            "conformance/benchmark drivers and remove them from production hot paths."
        )
    return {
        "undo_sequential": undo_sequential,
        "page_retained": page_retained,
        "page_general": page_general,
        "selection": selection,
        "consequence": consequence,
    }


def ratio(value: float | int, baseline: float | int) -> float:
    return float(value) / float(baseline) if baseline else 0.0


def render_decision(payloads: dict[str, dict[str, Any]]) -> str:
    outcome = decision_outcome(payloads)
    full = cell_summaries(payloads[FULL_CLONE_DRIVER])
    labels = {
        FULL_CLONE_DRIVER: "full clone",
        CLONE_PLUS_UNDO_DRIVER: "clone + undo",
        PAGE_COW_DRIVER: "event-page COW + undo",
    }
    lines = [
        "# Search branching decision v1",
        "",
        f"Decision: **{outcome['selection']}**.",
        "",
        outcome["consequence"],
        "",
        "The decision applies the pre-registered whole-rollout thresholds in "
        "`dense-page-cow-prereg-v1.md`. Correctness and matched provenance are "
        "absolute gates; clone latency is diagnostic only.",
        "",
        "## Matched primary evidence",
        "",
        "| Driver | Cell | sims/s | vs full | p99 root | vs full | peak RSS | vs full | RSS delta | vs full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for driver in ARTIFACTS:
        summaries = cell_summaries(payloads[driver])
        for cell in sorted(WHOLE_CELLS):
            summary = summaries[cell]
            baseline = full[cell]
            lines.append(
                f"| {labels[driver]} | `{cell}` | "
                f"{summary['simulations_per_second']:.1f} | "
                f"{ratio(summary['simulations_per_second'], baseline['simulations_per_second']):.3f}x | "
                f"{summary['root_latency_seconds_p99']:.3f}s | "
                f"{ratio(summary['root_latency_seconds_p99'], baseline['root_latency_seconds_p99']):.3f}x | "
                f"{bytes_mib(summary['rss_peak_bytes']):.1f} MiB | "
                f"{ratio(summary['rss_peak_bytes'], baseline['rss_peak_bytes']):.3f}x | "
                f"{bytes_mib(summary['rss_peak_delta_bytes']):.1f} MiB | "
                f"{ratio(summary['rss_peak_delta_bytes'], baseline['rss_peak_delta_bytes']):.3f}x |"
            )
    lines.extend(
        [
            "",
            "## Threshold results",
            "",
            f"- Clone-plus-undo sequential bar: `{'pass' if outcome['undo_sequential'] else 'fail'}`.",
            f"- Page-COW retained-memory bar: `{'pass' if outcome['page_retained'] else 'fail'}`.",
            f"- Page-COW general-driver bar: `{'pass' if outcome['page_general'] else 'fail'}`.",
            "",
            "Clone plus undo is rejected unless both flat cells reach 1.20x full-clone "
            "throughput without exceeding 1.10x p99 latency or RSS. Page COW is "
            "rejected for retained use unless both retained cells preserve at least "
            "0.90x throughput and meet every registered RSS reduction; general use "
            "also requires the flat guardrails. The table above makes each failed "
            "comparison explicit.",
            "",
            "## Branch lifecycle diagnostics",
            "",
            "| Driver | eager full forks | full checkpoints | max journal peak | max COW peak |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for driver in ARTIFACTS:
        primary = [
            summary
            for cell, summary in cell_summaries(payloads[driver]).items()
            if cell in WHOLE_CELLS
        ]
        journal_values = [
            summary["journal_peak_bytes"]
            for summary in primary
            if summary["journal_peak_bytes"] is not None
        ]
        cow_values = [
            summary["cow_bytes"]
            for summary in primary
            if summary["cow_bytes"] is not None
        ]
        journal_peak = (
            f"{bytes_mib(max(journal_values)):.2f} MiB" if journal_values else "null"
        )
        cow_peak = f"{bytes_mib(max(cow_values)):.2f} MiB" if cow_values else "null"
        lines.append(
            f"| {labels[driver]} | {sum(summary['eager_forks'] for summary in primary)} | "
            f"{sum(summary['checkpoint_copies'] for summary in primary)} | "
            f"{journal_peak} | {cow_peak} |"
        )
    lines.extend(
        [
            "",
            "Full-clone and clone-plus-undo eager counts represent deep outer forks; "
            "only full clone takes full mark snapshots. Page COW reports zero full "
            "logical-state forks and checkpoints, with numeric copied-page peaks. "
            "Unsupported allocator totals remain null in the raw receipts.",
            "",
            "## Provenance",
            "",
        ]
    )
    for driver, (artifact, _) in ARTIFACTS.items():
        payload = payloads[driver]
        lines.append(
            f"- `{driver}`: `{artifact.relative_to(ROOT)}`; artifact "
            f"`{payload['artifact_sha256']}`."
        )
    reference = payloads[FULL_CLONE_DRIVER]
    lines.extend(
        [
            "",
            f"All three receipts use source `{reference['run']['source_sha256']}`, "
            f"revision `{reference['run']['measurement_code_revision']}`, and release "
            f"binary `{reference['build']['binary_sha256']}` on the identical recorded host.",
            "Logical fixture summaries, seeds, workload dimensions, result checksums, "
            "outcomes, caps, and sampled final hashes match exactly across candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def persist_payload(driver: str, payload: dict[str, Any]) -> None:
    raw_path, report_path = ARTIFACTS[driver]
    atomic_write(raw_path, canonical_json(payload) + b"\n")
    atomic_write(report_path, render_report(payload).encode())


def run_matrix(
    profile: str,
    oversubscribed: bool,
    *,
    started_at: str,
    invoked_argv: list[str],
) -> dict[str, dict[str, Any]]:
    source = source_identity()
    revision = measurement_code_revision()
    build = build_binary()
    hardware = hardware_metadata(oversubscribed)
    payloads: dict[str, dict[str, Any]] = {}
    for driver in ARTIFACTS:
        assert_binary_unchanged(build)
        payloads[driver] = run_benchmark(
            profile,
            oversubscribed,
            driver=driver,
            started_at=started_at,
            invoked_argv=invoked_argv,
            source=source,
            build_metadata=build,
            hardware=hardware,
            revision=revision,
        )
    verify_matrix_payloads(payloads)
    for driver, payload in payloads.items():
        persist_payload(driver, payload)
    atomic_write(DECISION_PATH, render_decision(payloads).encode())
    return payloads


def load_and_verify_matrix() -> dict[str, dict[str, Any]]:
    payloads = {
        driver: json.loads(paths[0].read_text()) for driver, paths in ARTIFACTS.items()
    }
    verify_matrix_payloads(payloads)
    for driver, payload in payloads.items():
        if ARTIFACTS[driver][1].read_text() != render_report(payload):
            raise RuntimeError(f"generated report is stale for {driver}")
    if DECISION_PATH.read_text() != render_decision(payloads):
        raise RuntimeError("generated branching decision is stale")
    return payloads


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
    matrix_parser = subparsers.add_parser("run-matrix")
    matrix_parser.add_argument(
        "--profile", choices=("full", "smoke"), default="full"
    )
    matrix_parser.add_argument("--oversubscribed", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--driver", choices=tuple(ARTIFACTS), default=FULL_CLONE_DRIVER
    )
    verify_parser.add_argument("--input", type=Path, default=None)
    verify_parser.add_argument("--report", type=Path, default=None)
    subparsers.add_parser("verify-matrix")
    args = parser.parse_args()

    if args.command == "run-matrix":
        payloads = run_matrix(
            args.profile,
            args.oversubscribed,
            started_at=pre_execution_started_at,
            invoked_argv=invoked_argv,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "profile": args.profile,
                    "artifacts": {
                        driver: payload["artifact_sha256"]
                        for driver, payload in payloads.items()
                    },
                    "decision": decision_outcome(payloads)["selection"],
                }
            )
        )
        return 0
    if args.command == "verify-matrix":
        payloads = load_and_verify_matrix()
        print(
            json.dumps(
                {
                    "status": "verified",
                    "artifacts": {
                        driver: payload["artifact_sha256"]
                        for driver, payload in payloads.items()
                    },
                    "decision": decision_outcome(payloads)["selection"],
                }
            )
        )
        return 0

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
