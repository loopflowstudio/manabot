"""Hardware selection for fresh recipes; saved execution defaults stay CPU."""

from typing import Literal

import torch


def accelerator() -> Literal["cuda"] | None:
    """CUDA when available, otherwise None.

    Apple MPS is never selected automatically. On a quiet Apple M2 it was slower
    than CPU for collection at every measured batch size, only tied 8-thread
    CPU for learning, and its driver memory grew past 20 GB within a few
    learner steps. Callers request MPS explicitly with `device="mps"`.
    """
    return "cuda" if torch.cuda.is_available() else None
