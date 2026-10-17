"""Bound evaluation cost across all grid shapes before freezing a learning panel.

This stdlib-only worker installs the pinned environment once, then isolates each
training and evaluation command. Tiny training runs produce ordinarily admitted
initial policies; only their initialization is scored. These scores never select
a model. All outputs and failures stay under Etude's retained output directory.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal

from experiments.runners.gpu_job import (
    Deadlines,
    Worker,
    project_interpreter,
    write_json,
    write_retained_manifest,
)

PROTOCOL = Path("experiments/protocols/batching-evaluation-calibration-008.md")
SIZES = ("small", "middle", "large", "small-deep", "large-shallow")
ANCHORS = ("scripted_greedy", "random")
SEED = 13598
DEAL_SEED = 1359800000


@dataclass(frozen=True)
class EvaluationTiming:
    size: str
    anchor: str
    games: int
    coordinator_seconds: float
    slowest_game_seconds: float
    reserved_seconds_per_game: float


def _positive(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing numeric evaluation timing")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Evaluation timing must be finite and positive")
    return float(value)


def evaluation_timing(path: Path, size: str, anchor: str) -> EvaluationTiming:
    """Read costs only after the ordinary evaluator admitted a complete cohort.

    Reserve 1.5 times the larger of full coordinator/replay cost per game or the
    slowest observed game. This is an extrapolation, not a runtime guarantee for
    learned policies. The actual learning allocation keeps its hard deadline.
    """
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("Calibration requires completed evaluation")
    rows = result.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 8
        or result.get("expected_games") != 8
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("Calibration requires all eight games")
    seconds = _positive(result.get("evaluation_seconds"))
    slowest = max(_positive(row.get("game_seconds")) for row in rows)
    return EvaluationTiming(
        size,
        anchor,
        len(rows),
        seconds,
        slowest,
        1.5 * max(seconds / len(rows), slowest),
    )


def run(worker: Worker, interpreter: str, output: Path) -> None:
    """Fail the allocation on any interrupted setup, trajectory or arena panel."""

    def command(name: str, args: list[str], seconds: float) -> None:
        code = worker.command(name, args, seconds)
        if code != 0:
            raise RuntimeError(f"{name} exited {code}; calibration incomplete")

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
    protocol_bytes = PROTOCOL.read_bytes()
    (output / PROTOCOL.name).write_bytes(protocol_bytes)
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    timings: list[EvaluationTiming] = []
    for size in SIZES:
        study = output / size
        command(
            f"train-{size}",
            [
                interpreter,
                "-m",
                "experiments.runners.batching_grid",
                "--size",
                size,
                "--streams",
                "64",
                "--transitions",
                "4096",
                "--seed",
                str(SEED),
                "--seconds",
                "120",
                "--out",
                str(study),
            ],
            180,
        )
        for anchor in ANCHORS:
            protocol_path = output / f"protocol-{anchor}.json"
            write_json(
                protocol_path,
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
            panel = output / f"{size}-{anchor}"
            command(
                f"eval-{size}-{anchor}",
                [
                    interpreter,
                    "-m",
                    "experiments.runners.batching_evaluation",
                    "--run",
                    str(study / "training/run"),
                    "--transitions",
                    "0",
                    "--protocol",
                    str(protocol_path),
                    "--out",
                    str(panel),
                ],
                600,
            )
            timing = evaluation_timing(
                panel / "transitions-0/monitor.json", size, anchor
            )
            timings.append(timing)
            write_json(
                output / "timings.json",
                {
                    "purpose": "timing-only; no strength or model selection",
                    "complete": len(timings) == len(SIZES) * len(ANCHORS),
                    "protocol_sha256": digest,
                    "timings": [asdict(t) for t in timings],
                },
            )


def main() -> None:
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-status.json").exists():
        raise ValueError("Calibration attempt already has a receipt")
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    worker = Worker(output, Deadlines(deadline))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.phase("starting", outer_deadline=deadline, retention_seconds=240)
    error: str | None = None
    try:
        run(worker, project_interpreter(Path.cwd()), output)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        worker.phase("failed" if error else "completed", error=error)
        write_retained_manifest(output)


if __name__ == "__main__":
    main()
