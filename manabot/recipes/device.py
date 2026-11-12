"""Hardware selection for fresh recipes; saved execution defaults stay CPU."""

from typing import Literal

import torch

from manabot.training.mps import mps_torch_error


def accelerator() -> Literal["cuda", "mps"] | None:
    """Prefer CUDA, then Apple MPS on admitted Torch; None on CPU-only hosts."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and mps_torch_error() is None:
        return "mps"
    return None
