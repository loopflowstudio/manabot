"""A named CUDA admission gate inside the existing supervised job process group.

The frozen numerical suite and overhead probe run before the declared learner.
Receipts/logs use ordinary snapshots; interruption never authorizes a retry or
turns a partially checked job into a passed admission. No arbitrary test command
is accepted from a deployment plan.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Timer
import time
from typing import Literal

from pydantic import Field
import torch

from manabot.training.execution import atomic_json
from managym import _managym

from .deploy import current_source
from .jobs import Job
from .plan import Frozen, Source, digest


class ValidationCheck(Frozen):
    name: str
    exit_code: int
    seconds: float


class ValidationReceipt(Frozen):
    contract: Literal["policy-numerics-cuda-v1"] = "policy-numerics-cuda-v1"
    status: Literal["running", "completed", "failed", "deadline"] = "running"
    started_at: float
    deadline: float
    finished_at: float | None = None
    source: Source | None = None
    native_sha256: str | None = None
    torch_version: str = torch.__version__
    cuda_version: str | None = torch.version.cuda
    device: str | None = None
    checks: tuple[ValidationCheck, ...] = ()
    error_type: str | None = None
    seconds: float = Field(default=0, ge=0)


def validation_command(spec: Job, root: Path, learner: list[str]) -> list[str]:
    """The gate spends at most five minutes within the existing work allowance."""
    return [
        # Use the current uv-managed interpreter directly so this process, rather
        # than uv's launcher, owns the session that timeout cleanup terminates.
        sys.executable,
        "-m",
        "manabot.remote.validation",
        "--root",
        str(root),
        "--deadline",
        str(min(time.time() + 300, spec.work_deadline)),
        "--",
        *learner,
    ]


def _checks() -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "numerical-contract",
            [
                "uv",
                "run",
                "--no-sync",
                "pytest",
                "tests/training/test_numerical_health.py",
                "tests/training/test_learning_state.py::test_segments_keep_adam_ema_and_absolute_coordinates",
                "-q",
            ],
        ),
        (
            "optimizer-overhead",
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/benchmark_numerical_health.py",
                "--device",
                "cuda",
            ],
        ),
    )


def validate(root: Path, deadline: float) -> bool:
    """Fail closed without CUDA; a timeout kills this private process group."""
    if os.getpid() != os.getpgrp():
        raise RuntimeError("validation requires the supervisor's private process group")
    root.mkdir(parents=True, exist_ok=True)
    receipt = ValidationReceipt(started_at=time.time(), deadline=deadline)
    path = root / "numerical-validation.json"
    started = time.monotonic()
    atomic_json(path, receipt.model_dump(mode="json"))
    # Bound device discovery, filesystem calls and descendants too. The extra
    # second lets a normal command timeout persist its receipt before this last
    # resort; an unresponsive process retains an explicitly incomplete receipt.
    timer = Timer(
        max(0.01, deadline - time.time() + 1),
        lambda: os.killpg(os.getpgrp(), signal.SIGKILL),
    )
    timer.daemon = True
    timer.start()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA validation requested without CUDA")
        native = _managym.__file__
        if native is None:
            raise RuntimeError("native module identity unavailable")
        receipt = receipt.model_copy(
            update={
                "source": current_source(Path.cwd()),
                "native_sha256": digest(Path(native).read_bytes()),
                "device": torch.cuda.get_device_name(0),
            }
        )
        atomic_json(path, receipt.model_dump(mode="json"))
        environment = {**os.environ, "MANABOT_NUMERICS_DEVICE": "cuda"}
        with (root / "numerical-validation.log").open("ab") as log:
            for name, command in _checks():
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 0)
                before = time.monotonic()
                log.write(f"\n{name}\n".encode())
                log.flush()
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    timeout=remaining,
                    check=False,
                )
                receipt = receipt.model_copy(
                    update={
                        "checks": (
                            *receipt.checks,
                            ValidationCheck(
                                name=name,
                                exit_code=result.returncode,
                                seconds=time.monotonic() - before,
                            ),
                        )
                    }
                )
                atomic_json(path, receipt.model_dump(mode="json"))
                if result.returncode != 0:
                    raise RuntimeError("named numerical check failed")
        receipt = receipt.model_copy(update={"status": "completed"})
    except subprocess.TimeoutExpired:
        receipt = receipt.model_copy(update={"status": "deadline"})
    except Exception as error:
        receipt = receipt.model_copy(
            update={
                "status": "failed",
                "error_type": type(error).__name__,
            }
        )
    receipt = receipt.model_copy(
        update={
            "finished_at": time.time(),
            "seconds": time.monotonic() - started,
        }
    )
    atomic_json(path, receipt.model_dump(mode="json"))
    timer.cancel()
    if receipt.status == "deadline":
        # uv may have spawned descendants. This group belongs exclusively to this
        # supervised job; kill all of it after preserving the timeout receipt.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    return receipt.status == "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("learner", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    learner = args.learner[1:] if args.learner[:1] == ["--"] else args.learner
    if not learner:
        parser.error("declared learner command is required")
    if not validate(args.root, args.deadline):
        raise SystemExit(1)
    # Preserve the supervisor's process group and ordinary learner lifecycle.
    os.execvpe(learner[0], learner, os.environ)


if __name__ == "__main__":
    main()
