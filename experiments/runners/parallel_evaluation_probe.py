"""Measure serial/parallel evaluation on the same checkpoint and rental.

The bootstrap uses only the standard library. Ordinary evaluation CLIs own all
games and replay. This probe compares retained rows and full decision traces,
excluding only their measured latencies and hashes that include those latencies.
Its scores never select a training treatment.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from experiments.runners.batching_calibration import (
    ANCHORS,
    DEAL_SEED,
    SEED,
    evaluation_timing,
)
from experiments.runners.gpu_job import (
    Deadlines,
    Worker,
    project_interpreter,
    write_json,
    write_retained_manifest,
)
from experiments.runners.parallel_commands import Command, run_parallel

PROTOCOL = Path("experiments/protocols/parallel-evaluation-probe-009.md")


@dataclass(frozen=True)
class PanelEvidence:
    decision_digest: str
    result_sha256: str
    evaluator_seconds: float
    games: int


def cpu_allowance(quota: str, affinity: int) -> float:
    """Bound inference concurrency by both the CPU quota and allowed processors."""
    amount, period = quota.split()
    if int(period) <= 0 or affinity < 1:
        raise ValueError("Invalid CPU capacity receipt")
    cores = (
        float(affinity) if amount == "max" else min(affinity, int(amount) / int(period))
    )
    if cores < 4:
        raise ValueError("Parallel probe requires at least four CPU cores of quota")
    return cores


def panel_evidence(panel: Path, anchor: str) -> PanelEvidence:
    """Require complete replayed eight-leg evidence and retain all decision data.

    JSON is narrowed once here at the retained-file boundary. The ordinary arena
    evaluator has already validated registrations, identities and cohort layout.
    This additional audit refuses partial rows or traces before comparing them.
    """
    path = panel / "transitions-0/monitor.json"
    timing = evaluation_timing(path, "large", anchor)
    result = json.loads(path.read_text())
    rows = result["rows"]
    expected = {(DEAL_SEED, leg) for leg in range(8)}
    if {(r["deal_seed"], r["leg"]) for r in rows} != expected or any(
        not r["replay_passed"] or not r["terminated"] or r["failure"] is not None
        for r in rows
    ):
        raise ValueError("Parallel probe requires complete replayed cohorts")
    trace_paths = {r["trace_path"] for r in rows}
    games: list[dict[str, object]] = []
    for relative in sorted(trace_paths):
        trace = path.parent / relative
        if not trace.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Trace resolves outside its panel")
        for line in gzip.decompress(trace.read_bytes()).splitlines():
            game = json.loads(line)
            if not isinstance(game, dict) or not isinstance(
                game.get("decisions"), list
            ):
                raise ValueError("Malformed retained decision trace")
            game.pop("game_trace_sha256")
            for decision in game["decisions"]:
                if not isinstance(decision, dict):
                    raise ValueError("Malformed retained decision")
                decision.pop("latency_seconds")
            games.append(game)
    if len(games) != 8 or {(g["deal_seed"], g["leg"]) for g in games} != expected:
        raise ValueError("Parallel probe requires all eight game traces")
    cost_fields = {
        "latency",
        "game_seconds",
        "game_trace_sha256",
        "trace_sha256",
        "trace_shard_sha256",
    }
    normalized_rows = [
        {k: v for k, v in row.items() if k not in cost_fields} for row in rows
    ]
    identity_fields = (
        "run_id",
        "regime_digest",
        "training_seed",
        "artifact",
        "coordinates",
        "protocol",
        "key",
        "candidate",
        "opponent",
    )
    canonical = json.dumps(
        {
            "identity": {key: result[key] for key in identity_fields},
            "rows": sorted(normalized_rows, key=lambda r: (r["deal_seed"], r["leg"])),
            "games": sorted(games, key=lambda g: (g["deal_seed"], g["leg"])),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return PanelEvidence(
        hashlib.sha256(canonical).hexdigest(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        timing.coordinator_seconds,
        timing.games,
    )


def run(worker: Worker, output: Path) -> None:
    interpreter = project_interpreter(Path.cwd())
    cgroup = Path("/sys/fs/cgroup")
    if (cgroup / "cpu.max").exists():
        quota = (cgroup / "cpu.max").read_text()
    else:
        candidates = sorted(cgroup.glob("cpu*/cpu.cfs_quota_us"))
        if len(candidates) != 1:
            raise ValueError("CPU quota is unavailable; parallel probe not admitted")
        amount = int(candidates[0].read_text())
        period = candidates[0].with_name("cpu.cfs_period_us").read_text().strip()
        quota = f"{'max' if amount < 0 else amount} {period}"
    affinity = len(os.sched_getaffinity(0))
    allowance = cpu_allowance(quota, affinity)
    write_json(
        output / "cpu-admission.json",
        {
            "quota": quota,
            "affinity_cpus": affinity,
            "allowed_cores": allowance,
            "evaluation_workers": 4,
        },
    )

    def command(name: str, argv: list[str], seconds: float) -> None:
        code = worker.command(name, argv, seconds)
        if code:
            raise RuntimeError(f"{name} exited {code}; parallel probe incomplete")

    command("setup", ["bash", "scripts/benchmarks/setup-worker-gpu.sh"], 600)
    command(
        "cuda-admission",
        [
            interpreter,
            "-m",
            "experiments.runners.cuda_admission",
            "--out",
            str(output / "cuda-admission.json"),
        ],
        60,
    )
    frozen = PROTOCOL.read_bytes()
    (output / PROTOCOL.name).write_bytes(frozen)
    digest = hashlib.sha256(frozen).hexdigest()
    command(
        "training",
        [
            interpreter,
            "-m",
            "experiments.runners.batching_grid",
            "--size",
            "large",
            "--streams",
            "64",
            "--transitions",
            "4096",
            "--seed",
            str(SEED),
            "--seconds",
            "120",
            "--out",
            str(output / "study"),
        ],
        180,
    )
    for anchor in ANCHORS:
        write_json(
            output / f"protocol-{anchor}.json",
            {
                "purpose": "frozen-study-evaluation",
                "study_protocol_sha256": digest,
                "deal_seeds": [DEAL_SEED],
                "include_mirrors": True,
                "opponent": anchor,
                "game_seconds": 120,
                "max_commands": 10000,
            },
        )

    def panel_argv(name: str, anchor: str) -> tuple[str, ...]:
        return (
            interpreter,
            "-m",
            "experiments.runners.batching_evaluation",
            "--run",
            str(output / "study/training/run"),
            "--transitions",
            "0",
            "--protocol",
            str(output / f"protocol-{anchor}.json"),
            "--out",
            str(output / name),
        )

    start = time.monotonic()
    for anchor in ANCHORS:
        name = f"serial-{anchor}"
        command(name, list(panel_argv(name, anchor)), 600)
    serial_seconds = time.monotonic() - start
    # Compare command/coordinator wall time in both phases. Canonical trace
    # auditing belongs outside both measured intervals.
    serial = {
        anchor: panel_evidence(output / f"serial-{anchor}", anchor)
        for anchor in ANCHORS
    }
    write_json(
        output / "serial.json",
        {
            "seconds": serial_seconds,
            "panels": {k: asdict(v) for k, v in serial.items()},
        },
    )
    commands = tuple(
        Command(
            f"parallel-{copy}-{anchor}",
            panel_argv(f"parallel-{copy}-{anchor}", anchor),
            600,
        )
        for copy in range(2)
        for anchor in ANCHORS
    )
    start = time.monotonic()
    code = run_parallel(worker, commands, workers=4)
    parallel_seconds = time.monotonic() - start
    if code:
        raise RuntimeError(f"Parallel panels exited {code}; probe incomplete")
    parallel: dict[str, PanelEvidence] = {}
    for copy in range(2):
        for anchor in ANCHORS:
            name = f"parallel-{copy}-{anchor}"
            evidence = panel_evidence(output / name, anchor)
            parallel[name] = evidence
            if evidence.decision_digest != serial[anchor].decision_digest:
                raise ValueError(f"Decision or identity mismatch in {name}")
    write_json(
        output / "comparison.json",
        {
            "status": "completed",
            "purpose": "equivalence and timing only",
            "protocol_sha256": digest,
            "serial_seconds": serial_seconds,
            "parallel_seconds": parallel_seconds,
            # The parallel phase repeats twice the serial phase's game count.
            "equal_work_speedup": 2 * serial_seconds / parallel_seconds,
            "serial_games": 16,
            "parallel_games": 32,
            "parallel_workers": 4,
            "all_decisions_and_identities_equal": True,
            "parallel_panels": {k: asdict(v) for k, v in parallel.items()},
            "limitation": "One host, initialization and deal block; no strength or universal scaling claim",
        },
    )


def main() -> None:
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-status.json").exists():
        raise ValueError("Parallel probe already has a receipt")
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    worker = Worker(output, Deadlines(deadline))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.phase("starting", outer_deadline=deadline, retention_seconds=240)
    error: str | None = None
    try:
        run(worker, output)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        worker.phase("failed" if error else "completed", error=error)
        write_retained_manifest(output)


if __name__ == "__main__":
    main()
