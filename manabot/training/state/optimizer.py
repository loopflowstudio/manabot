"""Portable Adam moments, addressed by the same parameter names as a policy.

The global step counts accepted optimizer updates. Each parameter counts the
updates on which it received a gradient, including a gradient of zero.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdamState:
    step: int
    param_steps: dict[str, int]
    exp_avg: dict[str, np.ndarray]
    exp_avg_sq: dict[str, np.ndarray]


def admit_adam(
    adam: AdamState, shapes: Mapping[str, tuple[int, ...]], *, step: int
) -> None:
    """Check the complete optimizer against the policy and saved update count."""
    if type(adam.step) is not int or adam.step != step or step < 0:
        raise ValueError(f"adam/step {adam.step!r} differs from optimizer count {step}")
    expected = set(shapes) if step else set()
    if missing := expected ^ adam.param_steps.keys():
        raise ValueError(f"adam/param_steps keys differ: {sorted(missing)}")
    for name, count in adam.param_steps.items():
        if type(count) is not int or not 0 <= count <= step:
            raise ValueError(f"adam/param_steps/{name}: invalid step {count!r}")
    stepped = {name for name, count in adam.param_steps.items() if count}
    for field, arrays in (("exp_avg", adam.exp_avg), ("exp_avg_sq", adam.exp_avg_sq)):
        if missing := stepped ^ arrays.keys():
            raise ValueError(f"adam/{field} keys differ: {sorted(missing)}")
        for name, value in arrays.items():
            if value.shape != shapes[name] or value.dtype != np.float32:
                raise ValueError(f"adam/{field}/{name}: incompatible shape or dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"adam/{field}/{name}: nonfinite moment")
            if field == "exp_avg_sq" and (value < 0).any():
                raise ValueError(f"adam/{field}/{name}: negative second moment")


def admit_policy_arrays(
    arrays: Mapping[str, np.ndarray], shapes: Mapping[str, tuple[int, ...]]
) -> None:
    """Refuse a partial, reshaped or damaged set before replacing any weight."""
    if differing := arrays.keys() ^ shapes.keys():
        raise ValueError(f"policy parameter keys differ: {sorted(differing)}")
    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype != np.float32:
            raise ValueError(f"parameter {name}: incompatible shape or dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"parameter {name}: nonfinite weight")
