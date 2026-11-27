"""Etude's model-loading entry point; only this module chooses Torch and device.

Device order is cuda -> mps -> cpu, overridable by MANABOT_EVAL_DEVICE.
Precision is declared explicitly (`load`, `load_tf32`, `load_bf16`) because it
alters logits and compute identity. Each returned forward accepts Prepared
batches of any positive batch size and returns float32 masked logits of shape
[batch, A] in input order. Etude owns lifetime, residency, request batching
and process allocation.
"""

from collections.abc import Callable
import os
from typing import Literal, cast

from etude.checkpoints import CheckpointRef
import numpy as np

from manabot.policy.batching import concatenate
from manabot.policy.prepare import Prepared
from manabot.sim.registration import EvalPrecision

EvalDevice = Literal["cuda", "mps", "cpu"]
EVAL_DEVICES: tuple[EvalDevice, ...] = ("cuda", "mps", "cpu")


def select_device() -> EvalDevice:
    """Choose the evaluation Torch device: MANABOT_EVAL_DEVICE, else cuda -> mps -> cpu."""
    import torch

    override = os.environ.get("MANABOT_EVAL_DEVICE")
    if override is not None and override != "":
        if override not in EVAL_DEVICES:
            accepted = ", ".join(EVAL_DEVICES)
            raise ValueError(
                f"unknown MANABOT_EVAL_DEVICE {override!r}; expected one of: {accepted}"
            )
        if override == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "MANABOT_EVAL_DEVICE='cuda' requested, but CUDA is unavailable"
            )
        if override == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "MANABOT_EVAL_DEVICE='mps' requested, but MPS is unavailable"
            )
        return cast(EvalDevice, override)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_with_precision(
    checkpoint: CheckpointRef,
    precision: EvalPrecision,
) -> Callable[[list[object]], list[object]]:
    import torch

    from manabot.backends.torch.arrays import on_device
    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.backends.torch.learner import (
        apply_matmul_precision,
        forward_precision,
        validate_precision,
    )

    device = torch.device(select_device())
    validate_precision(precision, device)
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("precision bf16 is not supported on this CUDA device")
        if device.type == "mps" and not torch.backends.mps.is_macos_or_newer(14, 0):
            raise ValueError("precision bf16 on MPS requires macOS 14.0 or newer")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    apply_matmul_precision(precision)
    model, _ = load_checkpoint(str(checkpoint.path))
    model.to(device).eval()
    with torch.inference_mode():
        model.cache_programs()

    def forward(inputs: list[object]) -> list[object]:
        if not inputs:
            return []
        if any(not isinstance(row, Prepared) or row.batch < 1 for row in inputs):
            raise TypeError("Manabot inference takes nonempty host Prepared inputs")
        rows = cast(list[Prepared[np.ndarray]], inputs)
        batch = rows[0] if len(rows) == 1 else concatenate(rows)
        if device.type == "cuda":
            apply_matmul_precision(precision)
        with torch.inference_mode(), forward_precision(precision, device):
            logits, _ = model(on_device(batch, device))
        values = logits.float().cpu().numpy()
        values[~batch.legal] = -np.inf
        if len(rows) == 1:
            return [values.copy()]
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
