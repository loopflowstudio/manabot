"""Run declared feasibility cells sequentially after one GPU environment setup.

The launcher embeds this stdlib wrapper in an immutable job command against one
frozen scientific source. Every cell gets a fresh training process and retained
directory. Original transition targets and learner deadlines stay unchanged;
failure stops the group, leaving unstarted cells pending for operator review.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from experiments.runners.gpu_job import (
    Deadlines,
    Worker,
    project_interpreter,
    write_json,
)

_SIZES = {"small", "middle", "large", "small-deep", "large-shallow"}
_STREAMS = {4, 64, 128, 256, 512}


@dataclass(frozen=True)
class Cell:
    size: str
    streams: int

    @property
    def name(self) -> str:
        return f"{self.size}-s{self.streams}"

    @classmethod
    def parse(cls, value: str) -> "Cell":
        size, separator, streams = value.partition(":")
        if not separator or size not in _SIZES or not streams.isdecimal():
            raise ValueError("Cell must be a declared size:stream-count pair")
        count = int(streams)
        if count not in _STREAMS:
            raise ValueError("Undeclared stream count")
        return cls(size, count)


def verify_completed(root: Path, cell: Cell) -> None:
    """Verify retained completion before allowing the next cell to consume time."""
    run = json.loads((root / "training/run/run.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        run["status"] != "completed"
        or run["seed"] != 13571
        or summary["checkpoint_finite"] is not True
        or summary["learner_transitions"] != 131072
        or summary["updates"] != 131072 // (cell.streams * 64)
    ):
        raise ValueError(f"{cell.name} did not finish with a finite checkpoint")
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Missing retained study manifest")
    for entry in manifest:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Study manifest escapes its cell directory")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Retained artifact differs: {entry['path']}")


def publish_progress(output: Path, cells: list[Cell], completed: int) -> None:
    """Aggregate persisted clocks without letting a new cell reset group progress."""
    clocks: dict[str, float] = {"cells_completed": float(completed)}
    checkpoint: str | None = None
    for index, cell in enumerate(cells):
        path = output / cell.name / "etude-result.json"
        if not path.exists():
            continue
        progress = json.loads(path.read_text())
        for key in ("updates", "learner_transitions", "training_seconds"):
            value = progress["clocks"][key]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Invalid persisted cell clock")
            clocks[key] = clocks.get(key, 0) + value
            clocks[f"cell_{index}_{key}"] = value
        relative = progress.get("checkpoint")
        if relative is not None:
            candidate = output / cell.name / relative
            if not candidate.resolve().is_relative_to((output / cell.name).resolve()):
                raise ValueError("Checkpoint escapes its cell directory")
            if candidate.is_file():
                checkpoint = candidate.relative_to(output).as_posix()
    clocks["observed_unix_seconds"] = time.time()
    write_json(
        output / "etude-result.json", {"clocks": clocks, "checkpoint": checkpoint}
    )


@contextmanager
def reporting(worker: Worker, cells: list[Cell], completed: int) -> Iterator[None]:
    """A reporting failure stops work; the last valid retained clocks remain readable."""
    stopped = threading.Event()
    errors: list[Exception] = []

    def report() -> None:
        while not stopped.is_set():
            try:
                publish_progress(worker.output, cells, completed)
            except Exception as error:
                errors.append(error)
                worker.stopping = True
                return
            stopped.wait(5)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stopped.set()
        reporter.join()
        if errors:
            raise errors[0]
        publish_progress(worker.output, cells, completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Cell.parse, action="append", required=True)
    args = parser.parse_args()
    cells: list[Cell] = args.cell
    if len(cells) > 5 or len(set(cells)) != len(cells):
        parser.error("Declare one to five unique cells")
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
        cells=[cell.name for cell in cells],
        outer_deadline=deadline,
        retention_seconds=240,
    )
    code = worker.command("nvidia-smi", ["nvidia-smi", "-q"], 15)
    if code == 0:
        code = worker.command(
            "setup", ["bash", "scripts/benchmarks/setup-worker-gpu.sh"], 1200
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
    completed: list[str] = []
    for cell in cells:
        if code != 0 or worker.stopping:
            break
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
                    "13571",
                    "--transitions",
                    "131072",
                    "--out",
                    str(cell_output / "study"),
                ],
                1860,
            )
        if code == 0:
            verify_completed(cell_output / "study", cell)
            completed.append(cell.name)
            worker.phase("cell_completed", cell=cell.name, completed=completed)
            publish_progress(output, cells, len(completed))
    worker.phase(
        "completed" if len(completed) == len(cells) else "incomplete",
        completed=completed,
        remaining=[cell.name for cell in cells if cell.name not in completed],
        exit_code=code,
    )
    raise SystemExit(0 if len(completed) == len(cells) else (code or 124))


if __name__ == "__main__":
    main()
