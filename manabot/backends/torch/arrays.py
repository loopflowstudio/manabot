"""NumPy at the torch backend's boundary: conversion, dtype and device.

`from_numpy` shares memory with its input on CPU and copies to another
device; `to_numpy` detaches and copies to the host (a CPU tensor is viewed).
Neither changes a dtype, so a float64 operand stays float64 and an integer
index keeps its width.
"""

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn


def from_numpy(array: np.ndarray, device: torch.device | str = "cpu") -> Tensor:
    return torch.as_tensor(array, device=device)


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def float32(tensor: Tensor) -> Tensor:
    """A forward output in float32, whatever precision the forward ran in."""
    return tensor.float()


def flat_numpy(named: Iterable[tuple[str, Tensor]]) -> dict[str, np.ndarray]:
    """Named parameters, buffers or gradients as host arrays under the same names."""
    return {name: to_numpy(value) for name, value in named}


def device_of(module: nn.Module) -> torch.device:
    """Where a module computes: the device of its first parameter."""
    return next(module.parameters()).device
