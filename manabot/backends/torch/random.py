"""Initialize the sampling streams shared by Torch training and simulation."""

import random

import numpy as np
import torch


def seed_random(seed: int, *, torch_deterministic: bool = True) -> None:
    """Reset training RNGs without constructing a tracking run.

    Call at the declared sampling boundary: evaluation and checkpoint admission
    can consume RNG state before collection begins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
