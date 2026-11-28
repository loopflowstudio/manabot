"""Etude's model-loading entry point, selecting the framework and device.

Device order is cuda -> mlx -> mps -> cpu, overridable by MANABOT_EVAL_DEVICE.
Precision is declared explicitly (`load`, `load_tf32`, `load_bf16`) because it
alters logits and compute identity. Each returned forward accepts Prepared
batches of any positive batch size and returns float32 masked logits of shape
[batch, A] in input order. Etude owns lifetime, residency, request batching
and process allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import platform
from typing import TYPE_CHECKING, Literal, cast

from etude.checkpoints import CheckpointRef
import numpy as np

from manabot.backends import resolve
from manabot.policy.batching import concatenate
from manabot.policy.prepare import Prepared
from manabot.sim.registration import EvalPrecision

if TYPE_CHECKING:
    import torch

    from manabot.backends.torch.policy import ActorCritic

EvalDevice = Literal["cuda", "mlx", "mps", "cpu"]
EVAL_DEVICES: tuple[EvalDevice, ...] = ("cuda", "mlx", "mps", "cpu")
Logits = Callable[[Prepared[np.ndarray]], np.ndarray]


def mlx_available() -> bool:
    """Check for the optional MLX extra and usable Metal on Apple Silicon."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        return resolve("mlx").gpu_available()
    except (ImportError, RuntimeError):
        return False


def device_available(device: EvalDevice) -> bool:
    if device == "cpu":
        return True
    if device == "mlx":
        return mlx_available()

    if device == "cuda" and platform.system() == "Darwin":
        return False

    import torch

    if device == "cuda":
        return torch.cuda.is_available()
    return torch.backends.mps.is_available()


def select_device() -> EvalDevice:
    """Choose the available evaluation device in order, unless explicitly set."""
    override = os.environ.get("MANABOT_EVAL_DEVICE")
    if override:
        if override not in EVAL_DEVICES:
            accepted = ", ".join(EVAL_DEVICES)
            raise ValueError(
                f"unknown MANABOT_EVAL_DEVICE {override!r}; expected one of: {accepted}"
            )
        device = cast(EvalDevice, override)
        if not device_available(device):
            extra = (
                " (requires Apple Silicon and uv sync --extra mlx)"
                if device == "mlx"
                else ""
            )
            raise ValueError(
                f"MANABOT_EVAL_DEVICE={device!r} requested, but {device.upper()} is unavailable{extra}"
            )
        return device
    return next(device for device in EVAL_DEVICES if device_available(device))


def mlx_logits(path: Path, precision: EvalPrecision) -> Logits:
    if precision == "tf32":
        raise ValueError("precision tf32 requires CUDA")
    backend = resolve("mlx", precision="bfloat16" if precision == "bf16" else precision)
    model, _ = backend.load_policy(path)
    model.to("gpu").eval()
    model.prepare_inference()
    return model.evaluate_logits


@contextmanager
def torch_forward_context(
    precision: EvalPrecision, device: torch.device
) -> Iterator[None]:
    import torch

    from manabot.backends.torch.learner import apply_matmul_precision, forward_precision

    if device.type == "cuda":
        apply_matmul_precision(precision)
    with torch.inference_mode(), forward_precision(precision, device):
        yield


def load_torch_policy(
    path: Path,
    precision: EvalPrecision,
    device: torch.device,
) -> ActorCritic:
    """Load the padded evaluation policy and cache its catalog under the declared precision."""
    import torch

    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.backends.torch.learner import (
        apply_matmul_precision,
        validate_precision,
    )

    validate_precision(precision, device)
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("precision bf16 is not supported on this CUDA device")
        if device.type == "mps" and not torch.backends.mps.is_macos_or_newer(14, 0):
            raise ValueError("precision bf16 on MPS requires macOS 14.0 or newer")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    apply_matmul_precision(precision)
    model, _ = load_checkpoint(str(path))
    model.to(device).eval()
    with torch_forward_context(precision, device):
        model.cache_programs()
    return model


def torch_logits(
    path: Path,
    precision: EvalPrecision,
    device_name: Literal["cuda", "mps", "cpu"],
) -> Logits:
    import torch

    from manabot.backends.torch.arrays import on_device

    device = torch.device(device_name)
    model = load_torch_policy(path, precision, device)

    def logits(batch: Prepared[np.ndarray]) -> np.ndarray:
        with torch_forward_context(precision, device):
            scores, _ = model(on_device(batch, device))
        return scores.float().cpu().numpy()

    return logits


def load_with_precision(
    checkpoint: CheckpointRef,
    precision: EvalPrecision,
) -> Callable[[list[object]], list[object]]:
    device = select_device()
    path = Path(checkpoint.path)
    logits = (
        mlx_logits(path, precision)
        if device == "mlx"
        else torch_logits(path, precision, device)
    )

    def forward(inputs: list[object]) -> list[object]:
        if not inputs:
            return []
        if any(not isinstance(row, Prepared) or row.batch < 1 for row in inputs):
            raise TypeError("Manabot inference takes nonempty host Prepared inputs")
        rows = cast(list[Prepared[np.ndarray]], inputs)
        batch = rows[0] if len(rows) == 1 else concatenate(rows)
        values = logits(batch)
        values[~batch.legal] = -np.inf
        found: list[object] = []
        offset = 0
        for row in rows:
            end = offset + row.batch
            found.append(values[offset:end].copy())
            offset = end
        return found

    return forward


def load(checkpoint: CheckpointRef) -> Callable[[list[object]], list[object]]:
    """Load an Etude checkpoint under float32 precision."""
    return load_with_precision(checkpoint, "float32")


def load_tf32(checkpoint: CheckpointRef) -> Callable[[list[object]], list[object]]:
    """Load an Etude checkpoint under declared CUDA TF32 precision."""
    return load_with_precision(checkpoint, "tf32")


def load_bf16(checkpoint: CheckpointRef) -> Callable[[list[object]], list[object]]:
    """Load an Etude checkpoint under declared bfloat16 autocast precision."""
    return load_with_precision(checkpoint, "bf16")
