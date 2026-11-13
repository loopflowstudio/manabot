"""NumPy at the torch backend's boundary: conversion, dtype and device.

`from_numpy` shares memory with its input on CPU and copies to another
device; `to_numpy` detaches and copies to the host (a CPU tensor is viewed).
Neither changes a dtype, so a float64 operand stays float64 and an integer
index keeps its width.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from manabot.policy.prepare import Prepared


def from_numpy(array: np.ndarray, device: torch.device | str = "cpu") -> Tensor:
    return torch.as_tensor(array, device=device)


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def float32(tensor: Tensor) -> Tensor:
    """A forward output in float32, whatever precision the forward ran in."""
    return tensor.float()


def device_of(module: nn.Module) -> torch.device:
    """Where a module computes: the device of its first parameter."""
    return next(module.parameters()).device


def _array_fields[A](value: object, leaf: type[A], prefix: str = "") -> dict[str, A]:
    """Flatten the dataclass tree; tuples preserve their field coordinates."""
    if isinstance(value, leaf):
        return {prefix: value}
    if is_dataclass(value) and not isinstance(value, type):
        children = ((f.name, getattr(value, f.name)) for f in fields(value))
    elif isinstance(value, tuple):
        children = ((str(i), child) for i, child in enumerate(value))
    else:
        return {}
    arrays: dict[str, A] = {}
    for name, child in children:
        arrays.update(_array_fields(child, leaf, f"{prefix}/{name}"))
    return arrays


def _replace_arrays(value: Any, moved: dict[str, Tensor], prefix: str = "") -> Any:
    # The dataclass structure is unchanged; only the generic array leaf changes.
    if prefix in moved:
        return moved[prefix]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                f.name: _replace_arrays(
                    getattr(value, f.name), moved, f"{prefix}/{f.name}"
                )
                for f in fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(
            _replace_arrays(child, moved, f"{prefix}/{i}")
            for i, child in enumerate(value)
        )
    return value


def on_device(
    batch: Prepared[np.ndarray], device: torch.device | str
) -> Prepared[Tensor]:
    """Move a prepared batch with one contiguous, pinned CUDA copy per dtype.

    CPU tensors share writable NumPy memory. Scalar metadata and packed sequence
    lengths remain on the host; all array leaves retain their shape and dtype.
    """
    device = torch.device(device)
    arrays = _array_fields(batch, np.ndarray)
    tensors = {
        key: torch.from_numpy(value if value.flags.writeable else value.copy())
        for key, value in arrays.items()
    }
    if device.type != "cpu":
        groups: dict[torch.dtype, list[str]] = {}
        for key, tensor in tensors.items():
            groups.setdefault(tensor.dtype, []).append(key)
        moved: dict[str, Tensor] = {}
        for dtype, keys in groups.items():
            sizes = [tensors[key].numel() for key in keys]
            block = torch.empty(
                sum(sizes), dtype=dtype, pin_memory=device.type == "cuda"
            )
            torch.cat([tensors[key].reshape(-1) for key in keys], out=block)
            block = block.to(device, non_blocking=device.type == "cuda")
            for key, part in zip(keys, block.split(sizes), strict=True):
                moved[key] = part.view(tensors[key].shape)
        tensors = moved
    return cast("Prepared[Tensor]", _replace_arrays(batch, tensors))
