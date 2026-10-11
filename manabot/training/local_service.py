"""Run a bounded local command independently of the submitting terminal.

launchd owns process lifetime on macOS. One immutable allocation supplies the
absolute deadline; the subprocess retains ordinary TrainingRun/Experiment recovery.
No crash retry, budget extension or remote machine operation is implicit.
"""

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from typing import Literal

import psutil
from pydantic import Field

from manabot.training.clock import watchdog_seconds
from manabot.training.execution import atomic_json
from manabot.training.models import Strict
from manabot.training.recovery import attempt_lock


class LocalAllocation(Strict):
    id: str
    started_unix: float
    deadline_unix: float
    seconds: float = Field(gt=0)
    cpu_threads: int = Field(ge=1)
    max_evaluators: int = Field(ge=0, le=1)
    paid_compute: bool = False


class LocalCommand(Strict):
    allocation: str
    cwd: str
    argv: list[str]
    report_argv: list[str] = []
    # Standard avoids macOS Background QoS throttling; thread bounds still apply.
    process_type: Literal["Background", "Standard"] = "Background"
    report_every_seconds: float = Field(default=180, ge=60)


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(descendants, timeout=5)


def supervise(path: Path) -> None:
    command = LocalCommand.model_validate_json(path.read_text())
    allocation = LocalAllocation.model_validate_json(
        Path(command.allocation).read_text()
    )
    remaining = allocation.deadline_unix - time.time()
    if (
        remaining <= 0
        or allocation.paid_compute
        or allocation.deadline_unix - allocation.started_unix > allocation.seconds
    ):
        raise ValueError("invalid or expired local allocation")
    continuous_deadline = watchdog_seconds() + remaining
    directory = path.parent
    with attempt_lock(directory / "owner.lock"):
        if (directory / "status.json").exists():
            raise ValueError(
                "command already attempted; explicit new recovery command required"
            )
        # -i only prevents idle sleep and lasts at most the original allocation.
        assertion = subprocess.Popen(
            [
                "/usr/bin/caffeinate",
                "-i",
                "-t",
                str(int(remaining)),
                "-w",
                str(os.getpid()),
            ]
        )
        child: subprocess.Popen[bytes] | None = None
        reporter: subprocess.Popen[bytes] | None = None
        state = {
            "pid": os.getpid(),
            "started_unix": time.time(),
            "deadline_unix": allocation.deadline_unix,
            "allocation_id": allocation.id,
            "status": "running",
            "command": command.argv,
        }

        def interrupted(signum: int, frame: object) -> None:
            raise InterruptedError(f"local service interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            with (directory / "worker.log").open("ab") as log:
                child = subprocess.Popen(
                    command.argv,
                    cwd=command.cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                    },
                )
                state["child_pid"] = child.pid
                last_report = time.time()
                while child.poll() is None:
                    if (
                        time.time() >= allocation.deadline_unix
                        or watchdog_seconds() >= continuous_deadline
                    ):
                        raise TimeoutError(
                            "absolute local allocation deadline exhausted"
                        )
                    state["heartbeat_unix"] = time.time()
                    state["elapsed_allocation_seconds"] = (
                        time.time() - allocation.started_unix
                    )
                    atomic_json(directory / "status.json", state)
                    if reporter is not None and reporter.poll() is not None:
                        state["report_exit_code"] = reporter.returncode
                        reporter = None
                    if (
                        command.report_argv
                        and reporter is None
                        and time.time() - last_report >= command.report_every_seconds
                    ):
                        reporter = subprocess.Popen(
                            command.report_argv,
                            cwd=command.cwd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        last_report = time.time()
                    time.sleep(2)
                state["exit_code"] = child.returncode
                state["status"] = "completed" if child.returncode == 0 else "failed"
        except BaseException as error:
            state["status"], state["error"] = (
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            for process in (reporter, child, assertion):
                if process is not None:
                    _stop(process)
            state["finished_unix"] = time.time()
            state["elapsed_allocation_seconds"] = time.time() - allocation.started_unix
            atomic_json(directory / "status.json", state)


def install(path: Path, label: str) -> None:
    if (
        sys.platform != "darwin"
        or not label.startswith("com.manabot.")
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in label)
    ):
        raise ValueError("local installation requires macOS and a com.manabot label")
    command = LocalCommand.model_validate_json(path.read_text())
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required")
    target = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if target.exists():
        raise ValueError("service label already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": label,
        "ProgramArguments": [
            uv,
            "run",
            "--no-sync",
            "python",
            "-m",
            "manabot.training.local_service",
            "--command",
            str(path.resolve()),
        ],
        "WorkingDirectory": command.cwd,
        "RunAtLoad": True,
        "ProcessType": command.process_type,
        "Nice": 10,
        "StandardOutPath": str(path.parent / "service.log"),
        "StandardErrorPath": str(path.parent / "service.log"),
        "EnvironmentVariables": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }
    with target.open("xb") as stream:
        plistlib.dump(payload, stream)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(target)], check=True
    )
    atomic_json(
        path.parent / "launch.json",
        {
            "label": label,
            "plist": str(target),
            "command": command.model_dump(),
            "launched_unix": time.time(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--install", metavar="LABEL")
    args = parser.parse_args()
    if args.install:
        install(args.command.resolve(), args.install)
    else:
        supervise(args.command.resolve())


if __name__ == "__main__":
    main()
