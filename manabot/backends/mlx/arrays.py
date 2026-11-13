"""NumPy at the MLX backend's boundary: conversion and dtype.

Every array crossing into the backend is float32, an integer index or a bool
mask; MLX's GPU has no float64, so `from_numpy` narrows float64 input. Arrays
live in unified memory, so conversion copies but never transfers between
devices, and the `device` argument the updates pass is ignored.
"""

from collections.abc import Mapping

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from manabot.infra.config import BF16_MIN_HIDDEN_DIM
from manabot.policy.protocol import Precision

FLOAT = mx.float32


def precision_dtype(precision: Precision) -> mx.Dtype:
    """The compute dtype; parameters and optimizer state remain float32."""
    dtypes = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
    try:
        return dtypes[precision]
    except KeyError as error:
        raise ValueError(
            f"unsupported MLX precision: {precision!r}; expected one of "
            f"{', '.join(dtypes)}"
        ) from error


def default_precision(hidden_dim: int) -> Precision:
    """The compute precision a policy of this width runs in when none is given:
    bfloat16 from `BF16_MIN_HIDDEN_DIM`, the width rule torch's GPU devices
    share, and float32 below it. An explicit precision always wins."""
    return "bfloat16" if hidden_dim >= BF16_MIN_HIDDEN_DIM else "float32"


def to_mlx(array: np.ndarray) -> mx.array:
    return mx.array(array)


def from_numpy(array: np.ndarray, device: object = None) -> mx.array:
    """`array` as an MLX array; float64 becomes float32."""
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return mx.array(array)


def to_numpy(array: mx.array) -> np.ndarray:
    return np.array(array)


def float32(array: mx.array) -> mx.array:
    """A forward output in float32, whatever precision the forward ran in."""
    return array.astype(FLOAT)


def flat_numpy(tree: Mapping[str, object] | list[object]) -> dict[str, np.ndarray]:
    """A parameter or gradient tree as NumPy arrays under dotted names."""
    return {name: np.array(value) for name, value in tree_flatten(tree)}
