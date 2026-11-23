"""NumPy at the MLX backend's boundary: conversion and dtype.

Every array crossing into the backend is float32, an integer index or a bool
mask; MLX's GPU has no float64, so `from_numpy` narrows float64 input. Arrays
live in unified memory, so conversion copies but never transfers between
devices, and the `device` argument the updates pass is ignored.
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from manabot.infra.config import BF16_MIN_HIDDEN_DIM
from manabot.policy.protocol import Precision

if TYPE_CHECKING:
    from manabot.training.learning import Precision as TrainingPrecision

FLOAT = mx.float32


def mlx_device(name: str) -> mx.DeviceType:
    if name == "cpu":
        return mx.cpu
    if name in {"mps", "gpu"}:
        return mx.gpu
    if name == "cuda":
        raise ValueError("MLX does not support CUDA")
    raise ValueError(f"MLX does not support device {name!r}")


def training_precision(precision: "TrainingPrecision") -> Precision:
    """Translate the recipe's precision to the MLX forward dtype name."""
    if precision == "tf32":
        raise ValueError("precision tf32 requires CUDA")
    return "bfloat16" if precision == "bf16" else precision


def precision_dtype(precision: Precision) -> mx.Dtype:
    """The compute dtype; parameters and optimizer state remain float32."""
    dtypes = {"float32": mx.float32, "bfloat16": mx.bfloat16}
    try:
        return dtypes[precision]
    except KeyError as error:
        raise ValueError(
            f"unsupported MLX precision: {precision!r}; expected one of "
            f"{', '.join(dtypes)}"
        ) from error


def default_precision(hidden_dim: int) -> Precision:
    """The compute precision a policy of this width runs in when none is given:
    bfloat16 from `BF16_MIN_HIDDEN_DIM` and float32 below it. An explicit
    precision always wins."""
    return "bfloat16" if hidden_dim >= BF16_MIN_HIDDEN_DIM else "float32"


def to_mlx(array: np.ndarray) -> mx.array:
    return mx.array(array)


def from_numpy(array: np.ndarray, device: object = None) -> mx.array:
    """`array` as an MLX array; float64 becomes float32."""
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return mx.array(array)


def to_numpy(array: mx.array) -> np.ndarray:
    if array.dtype == mx.bfloat16:
        array = array.astype(mx.float32)
    return np.array(array)


def float32(array: mx.array) -> mx.array:
    """A forward output in float32, whatever precision the forward ran in."""
    return array.astype(FLOAT)


def all_finite(arrays: Iterable[mx.array]) -> mx.array:
    """Reduce each tensor before stacking, so the check never packs a second model."""
    return mx.stack([mx.isfinite(value).all() for value in arrays]).all()


def flat_numpy(tree: Mapping[str, object] | list[object]) -> dict[str, np.ndarray]:
    """A parameter or gradient tree as NumPy arrays under dotted names."""
    return {name: np.array(value) for name, value in tree_flatten(tree)}
