"""Torch admission for Apple MPS training.

MPS training requires Torch ``MPS_MINIMUM_TORCH`` or newer: Torch 2.10 computed
finite but wrong, or nonfinite, MPS gradients on some learner minibatches whose
losses matched CPU, and gradient clipping applied the finite wrong ones.
Execution admission rejects MPS stages on older Torch; fresh recipes skip MPS
during automatic device selection.
"""

import torch
from torch.torch_version import TorchVersion

MPS_MINIMUM_TORCH = "2.15"


def mps_torch_error() -> str | None:
    """Why the installed Torch cannot train on MPS, or None when it can.

    Pre-releases of the minimum (``2.15.0a0``) are not admitted.
    """
    if TorchVersion(torch.__version__) >= MPS_MINIMUM_TORCH:
        return None
    return (
        f"MPS training requires Torch {MPS_MINIMUM_TORCH} or newer; installed Torch "
        f"{torch.__version__} computes incorrect MPS gradients (seen on 2.10). "
        "Upgrade Torch (uv sync --locked) or train on CPU with --recipe local"
    )
