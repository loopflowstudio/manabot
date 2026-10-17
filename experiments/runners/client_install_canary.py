"""Exercise prebuilt Etude startup with the existing two-update GPU canary.

The launcher embeds this stdlib entry point in the job command while retaining
the original scientific source archive. Project setup gets 1,200 seconds for
CUDA dependency downloads; the outer deadline and retention reserve still win.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import sys

from experiments.runners.gpu_job import Deadlines, Worker, project_interpreter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-setup", type=Path)
    args = parser.parse_args()
    setup = (
        [sys.executable, str(args.locked_setup.resolve())]
        if args.locked_setup is not None
        else ["bash", "scripts/benchmarks/setup-worker-gpu.sh"]
    )
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-status.json").exists():
        raise ValueError("This attempt already has a worker receipt")
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    worker = Worker(output, Deadlines(deadline))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.phase("starting", outer_deadline=deadline, retention_seconds=240)
    code = worker.command("nvidia-smi", ["nvidia-smi", "-q"], 15)
    if code == 0:
        code = worker.command("setup", setup, 1200)
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
    if code == 0:
        code = worker.command(
            "training",
            [
                interpreter,
                "-m",
                "experiments.runners.size_frontier",
                "learning",
                "--size",
                "small",
                "--seed",
                "13599",
                "--canary",
                "--out",
                str(output / "study"),
            ],
            180,
        )
    worker.phase("completed" if code == 0 else "failed", exit_code=code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
