"""Numeric precision a training stage declares in `Execution.precision`.

- `float32`: nothing changes; every helper here is a no-op.
- `tf32`: for the stage, CUDA matmul and cuDNN may use TF32 tensor cores
  (`allow_tf32`). Weights, activations and losses stay float32 tensors.
- `bf16`: `torch.autocast(cuda, bfloat16)` around the policy forwards only, in
  collection (behavior sampling and stored values) and in the learner (bootstrap
  targets and the loss forward). Callers cast forward outputs back to float32,
  so behavior log-probabilities, targets, losses and their reductions are float32;
  weights and Adam state stay float32 and no gradient scaler is used, since bf16
  keeps float32's exponent range.

Both reduced precisions change numerics, so they are declared regime variants,
never edits of a float32 regime; `Execution` rejects them off CUDA.
"""

from contextlib import AbstractContextManager, nullcontext

import torch

from manabot.training.regime import Precision

# The process's switches before any stage set them. float32 keeps them, so float32
# runs train exactly as before these options existed (cuDNN's default allows TF32).
_PROCESS_TF32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def apply_matmul_precision(precision: Precision) -> None:
    """Set the TF32 switches a stage runs under: on for tf32, else the process's own."""
    matmul, cudnn = (True, True) if precision == "tf32" else _PROCESS_TF32
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def forward_precision(
    precision: Precision, device: torch.device
) -> AbstractContextManager[object]:
    """The context one policy forward runs in: bf16 autocast, else nothing."""
    if precision != "bf16":
        return nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)
