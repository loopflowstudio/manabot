"""Hardware selection for fresh recipes; saved execution defaults stay CPU."""

from typing import Literal

import torch


def accelerator() -> Literal["cuda", "mps"] | None:
    """Prefer CUDA, then Apple MPS; return None on CPU-only hosts."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None
