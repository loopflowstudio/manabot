"""A named CUDA admission gate that a training job runs before its learner.

The frozen numerical suite and the optimizer overhead probe run on the job's own
machine. Failure, a timeout or missing CUDA refuses training; the receipt and log
stay in the job's output either way. No arbitrary test command is accepted.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

from manabot.infra.files import atomic_json
from managym import _managym


class NumericsCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    exit_code: int
    seconds: float


class NumericsReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: Literal["policy-numerics-cuda-v1"] = "policy-numerics-cuda-v1"
    status: Literal["running", "completed", "failed", "deadline"] = "running"
    started_at: float
    deadline: float
    finished_at: float | None = None
    native_sha256: str | None = None
    torch_version: str = torch.__version__
    cuda_version: str | None = torch.version.cuda
    device: str | None = None
    checks: tuple[NumericsCheck, ...] = ()
    error_type: str | None = None
    seconds: float = Field(default=0, ge=0)


_CHECKS: tuple[tuple[str, list[str]], ...] = (
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


def check_numerics(output: Path, deadline: float) -> NumericsReceipt:
    """Run the named checks by `deadline` (Unix seconds); fail closed without CUDA.

    Writes `numerical-validation.json` and `.log` under `output` as it goes, so an
    interrupted gate leaves an explicitly incomplete receipt.
    """
    output.mkdir(parents=True, exist_ok=True)
    path = output / "numerical-validation.json"
    receipt = NumericsReceipt(started_at=time.time(), deadline=deadline)
    started = time.monotonic()
    atomic_json(path, receipt.model_dump(mode="json"))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA validation requested without CUDA")
        native = _managym.__file__
        if native is None:
            raise RuntimeError("native module identity unavailable")
        receipt = receipt.model_copy(
            update={
                "native_sha256": hashlib.sha256(Path(native).read_bytes()).hexdigest(),
                "device": torch.cuda.get_device_name(0),
            }
        )
        atomic_json(path, receipt.model_dump(mode="json"))
        environment = {**os.environ, "MANABOT_NUMERICS_DEVICE": "cuda"}
        with (output / "numerical-validation.log").open("ab") as log:
            for name, command in _CHECKS:
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
                check = NumericsCheck(
                    name=name,
                    exit_code=result.returncode,
                    seconds=time.monotonic() - before,
                )
                receipt = receipt.model_copy(
                    update={"checks": (*receipt.checks, check)}
                )
                atomic_json(path, receipt.model_dump(mode="json"))
                if result.returncode != 0:
                    raise RuntimeError("named numerical check failed")
        receipt = receipt.model_copy(update={"status": "completed"})
    except subprocess.TimeoutExpired:
        receipt = receipt.model_copy(update={"status": "deadline"})
    except Exception as error:
        receipt = receipt.model_copy(
            update={"status": "failed", "error_type": type(error).__name__}
        )
    receipt = receipt.model_copy(
        update={"finished_at": time.time(), "seconds": time.monotonic() - started}
    )
    atomic_json(path, receipt.model_dump(mode="json"))
    return receipt
