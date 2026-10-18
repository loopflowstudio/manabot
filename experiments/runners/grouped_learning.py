"""Run a declared learning comparison after one environment setup per seed.

Each PlannedCell gets a fresh training process, an explicit learner deadline,
and its own retained directory. Common experience checkpoints are validated
before setup; a failed or incomplete cell prevents subsequent admissions.
The launcher owns the frozen allocation and its rental/budget reservation.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

from experiments.runners.gpu_job import Deadlines, Worker, project_interpreter
from experiments.runners.grouped_grid import Cell, publish_progress, reporting


@dataclass(frozen=True)
class PlannedCell:
    cell: Cell
    seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Learner seconds must be finite and positive")

    @classmethod
    def parse(cls, value: str) -> "PlannedCell":
        cell, separator, limit = value.rpartition(":")
        if not separator:
            raise ValueError("Cell must be size:streams:learner-seconds")
        seconds = float(limit)
        return cls(Cell.parse(cell), seconds)


def validate_plan(
    cells: list[PlannedCell], seed: int, transitions: int, checkpoints: tuple[int, ...]
) -> None:
    if seed < 0 or not 1 <= len(cells) <= 25:
        raise ValueError("Declare a nonnegative seed and one to 25 cells")
    if len({planned.cell for planned in cells}) != len(cells):
        raise ValueError("A cell may occur only once within a seed")
    if (
        not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[0] <= 0
        or checkpoints[-1] != transitions
    ):
        raise ValueError(
            "Checkpoints must increase and include the transition endpoint"
        )
    for planned in cells:
        stride = planned.cell.streams * 64
        if any(point % stride for point in checkpoints):
            raise ValueError("Every checkpoint must be an exact iteration boundary")


def verify_completed(
    root: Path,
    planned: PlannedCell,
    *,
    seed: int,
    transitions: int,
    checkpoints: tuple[int, ...],
) -> None:
    """Verify this cell's seed, endpoint, milestones and complete artifact bytes.

    JSON is a boundary here: records are consumed from the just-exited child,
    and missing fields or malformed values fail before the next child starts.
    Full model/source/optimizer auditing remains a separate offline gate.
    """
    run = json.loads((root / "training/run/run.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        run["status"] != "completed"
        or run["seed"] != seed
        or run["parent_run_id"] is not None
        or len(run["stages"]) != 1
        or run["stages"][0]["learner_transitions"] != transitions
        or run["stages"][0].get("learning_state_origin") is not None
        or summary["checkpoint_finite"] is not True
        or summary["learner_transitions"] != transitions
        or summary["updates"] != transitions // (planned.cell.streams * 64)
    ):
        raise ValueError(
            f"{planned.cell.name} did not complete its declared trajectory"
        )
    saved = run["monitoring_checkpoints"]
    for point in checkpoints[:-1]:
        matches = [row for row in saved if row["learner_transitions"] == point]
        if (
            len(matches) != 1
            or matches[0]["artifact"] is None
            or matches[0]["error"] is not None
        ):
            raise ValueError(
                f"Missing unique retained checkpoint at {point} transitions"
            )
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Missing retained study manifest")
    seen: set[str] = set()
    for entry in manifest:
        relative = entry["path"]
        path = root / relative
        if relative in seen or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Repeated or escaping study artifact path")
        seen.add(relative)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Retained artifact differs: {relative}")
    artifacts = [
        run["stages"][0]["artifacts"][name]
        for name in ("initial_raw", "raw", "optimizer")
    ]
    artifacts.extend(
        next(row["artifact"] for row in saved if row["learner_transitions"] == point)
        for point in checkpoints[:-1]
    )
    for artifact in artifacts:
        path = root / "training/run" / Path(artifact["path"]).name
        if str(path.relative_to(root)) not in seen:
            raise ValueError("Checkpoint is not present in the retained manifest")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != artifact["bytes"] or digest != artifact["sha256"]:
            raise ValueError("Retained checkpoint differs from its recorded identity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", type=PlannedCell.parse, action="append", required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--transitions", type=int, required=True)
    parser.add_argument("--checkpoint-transitions", type=int, nargs="+", required=True)
    parser.add_argument("--locked-setup", type=Path, required=True)
    args = parser.parse_args()
    planned: list[PlannedCell] = args.cell
    checkpoints = tuple(args.checkpoint_transitions)
    validate_plan(planned, args.seed, args.transitions, checkpoints)
    cells = [item.cell for item in planned]
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-status.json").exists():
        raise ValueError("This attempt already has a worker receipt")
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    worker = Worker(output, Deadlines(deadline))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.phase(
        "starting",
        seed=args.seed,
        transitions=args.transitions,
        checkpoint_transitions=checkpoints,
        cells=[cell.name for cell in cells],
        outer_deadline=deadline,
        retention_seconds=240,
    )
    completed: list[str] = []
    code = 1
    error: str | None = None
    try:
        code = worker.command("nvidia-smi", ["nvidia-smi", "-q"], 15)
        if code == 0:
            code = worker.command(
                "setup", [sys.executable, str(args.locked_setup.resolve())], 1200
            )
        interpreter = project_interpreter(Path.cwd())
        if code == 0:
            code = worker.command(
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
        for item in planned:
            if code != 0 or worker.stopping:
                break
            cell = item.cell
            cell_output = output / cell.name
            cell_output.mkdir()
            with reporting(worker, cells, len(completed)):
                code = worker.command(
                    cell.name,
                    [
                        "env",
                        f"ETUDE_OUTPUT_DIR={cell_output}",
                        interpreter,
                        "-m",
                        "experiments.runners.batching_grid",
                        "--size",
                        cell.size,
                        "--streams",
                        str(cell.streams),
                        "--seed",
                        str(args.seed),
                        "--transitions",
                        str(args.transitions),
                        "--checkpoint-transitions",
                        *(str(point) for point in checkpoints),
                        "--seconds",
                        str(item.seconds),
                        "--out",
                        str(cell_output / "study"),
                    ],
                    item.seconds + 60,
                )
            if code == 0:
                verify_completed(
                    cell_output / "study",
                    item,
                    seed=args.seed,
                    transitions=args.transitions,
                    checkpoints=checkpoints,
                )
                completed.append(cell.name)
                worker.phase("cell_completed", cell=cell.name, completed=completed)
                publish_progress(output, cells, len(completed))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        code = 1
        raise
    finally:
        if code == 0 and len(completed) != len(cells):
            code = 124
        worker.phase(
            "completed" if code == 0 else "incomplete",
            completed=completed,
            remaining=[cell.name for cell in cells if cell.name not in completed],
            exit_code=code,
            error=error,
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
