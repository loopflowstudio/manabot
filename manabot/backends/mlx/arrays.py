"""NumPy at the MLX backend's boundary: conversion and dtype.

Every array crossing into the backend is float32 or an integer index; MLX's
GPU has no float64. Arrays live in unified memory, so conversion copies but
never transfers between devices.
"""

from collections.abc import Mapping

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

FLOAT = mx.float32


def to_mlx(array: np.ndarray) -> mx.array:
    return mx.array(array)


def to_numpy(array: mx.array) -> np.ndarray:
    return np.array(array)


def flat_numpy(tree: Mapping[str, object] | list[object]) -> dict[str, np.ndarray]:
    """A parameter or gradient tree as NumPy arrays under dotted names."""
    return {name: np.array(value) for name, value in tree_flatten(tree)}
