"""Run one study attempt under Etude's deadline and retained-output contract.

This stdlib-only entry point can run before dependencies are installed. Setup,
CUDA admission and training have distinct logs. Training writes directly into
retained outputs; stopping never depends on a final archive or shell copy.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import FrameType


@dataclass(frozen=True)
class Deadlines:
    outer: float
    retention_seconds: float = 240

    def work(self, seconds: float) -> float:
        deadline = min(time.time() + seconds, self.outer - self.retention_seconds)
        if deadline <= time.time():
            raise TimeoutError("No execution time remains before retention reserve")
        return deadline


def project_interpreter(root: Path) -> str:
    # Resolving the executable symlink bypasses pyvenv.cfg and loses dependencies.
    return str(root.absolute() / ".venv" / "bin" / "python")


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Worker:
    def __init__(self, output: Path, deadlines: Deadlines) -> None:
        self.output, self.deadlines = output, deadlines
        self.stopping = False

    def stop(self, signum: int, frame: FrameType | None) -> None:
        self.stopping = True

    def phase(self, phase: str, **details: object) -> None:
        value = {"phase": phase, "observed_unix_seconds": time.time(), **details}
        write_json(self.output / "worker-status.json", value)
        print(json.dumps(value), flush=True)

    def command(self, name: str, args: list[str], seconds: float) -> int:
        deadline = self.deadlines.work(seconds)
        self.phase(name, deadline=deadline)
        with (self.output / f"{name}.log").open("wb") as log:
            child = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                while child.poll() is None:
                    if self.stopping or time.time() >= deadline:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=2 if self.stopping else 30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=2)
                        self.phase(name + "_interrupted", exit_code=child.returncode)
                        return 124
                    time.sleep(0.2)
                return child.returncode
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        choices=("small", "middle", "large", "small-deep", "large-shallow"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument(
        "--study", choices=("size-frontier", "batching-grid"), default="size-frontier"
    )
    parser.add_argument(
        "--streams", type=int, choices=(4, 64, 128, 256, 512), default=4
    )
    parser.add_argument("--transitions", type=int, default=131_072)
    parser.add_argument("--checkpoint-transitions", type=int, nargs="+")
    args = parser.parse_args()
    if args.study == "size-frontier" and args.size not in ("small", "middle", "large"):
        parser.error("The added width/depth contrasts require --study batching-grid")
    if args.study != "batching-grid" and args.checkpoint_transitions is not None:
        parser.error("Transition checkpoints require --study batching-grid")
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    worker = Worker(output, Deadlines(deadline))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    if (output / "worker-status.json").exists():
        raise ValueError("This attempt already has a worker receipt")
    worker.phase("starting", outer_deadline=deadline, retention_seconds=240)
    # Preserve device/driver evidence even if dependency installation fails.
    try:
        worker.command("nvidia-smi", ["nvidia-smi", "-q"], 15)
    except FileNotFoundError:
        (output / "nvidia-smi.log").write_text("nvidia-smi unavailable\n")
    code = worker.command(
        "setup", ["bash", "scripts/benchmarks/setup-worker-gpu.sh"], 600
    )
    if code == 0:
        interpreter = project_interpreter(Path.cwd())
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
        if code == 0:
            command = [
                interpreter,
                "-m",
                "experiments.runners.size_frontier",
                "learning",
                "--size",
                args.size,
                "--seed",
                str(args.seed),
                "--out",
                str(output / "study"),
            ]
            if args.canary:
                command.append("--canary")
            if args.study == "batching-grid":
                if args.canary:
                    raise ValueError("Use the separately declared lifecycle canary")
                command = [
                    interpreter,
                    "-m",
                    "experiments.runners.batching_grid",
                    "--size",
                    args.size,
                    "--seed",
                    str(args.seed),
                    "--streams",
                    str(args.streams),
                    "--transitions",
                    str(args.transitions),
                    "--out",
                    str(output / "study"),
                ]
                if args.checkpoint_transitions is not None:
                    command += [
                        "--checkpoint-transitions",
                        *(str(n) for n in args.checkpoint_transitions),
                    ]
            code = worker.command("training", command, 3600)
    worker.phase("completed" if code == 0 else "failed", exit_code=code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
