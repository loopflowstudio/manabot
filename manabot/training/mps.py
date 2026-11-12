"""Torch admission for Apple MPS training.

MPS training requires Torch ``MPS_MINIMUM_TORCH`` or newer. On an Apple M2
(macOS 26.7), Torch 2.10–2.13 compute a wrong MPS weight gradient for
``F.linear`` with a large reduced dimension (``[160, 44550]^T @ [44550, 160]``,
float32, seeded random inputs: 0/5 fresh processes match CPU, relative L2 error
2.25); 2.14 and 2.15 pass 5/5 at 5e-7. An M4 Max does not reproduce it. In
manabot it appears as finite-but-wrong or nonfinite gradients in the w160l8
attention output projection on real learner minibatches whose losses match CPU,
and gradient clipping applies the finite wrong ones. 2.14 would suffice; the
minimum is the pinned 2.15. Execution admission rejects MPS stages on older
Torch; fresh recipes skip MPS during automatic device selection.
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
        f"{torch.__version__} is below that minimum (Torch 2.10–2.13 compute incorrect "
        "MPS linear-layer weight gradients on Apple M2). Upgrade Torch (uv sync --locked) "
        "or train on CPU"
    )
