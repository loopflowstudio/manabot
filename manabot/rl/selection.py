"""Advantage selection: which rows a policy loss reads, and its masked mean.

Both filters rank rows by advantage magnitude. A top-count filter keeps a
fixed share (stable ties); a quantile filter keeps every row at or above an
inclusive magnitude quantile, so ties are never split.
"""

import math
from typing import Literal

import torch
from torch import Tensor

from manabot.rl.numerics import float64_operand


def selected_rows(
    advantages: Tensor,
    fraction: float,
    minimum: float,
    *,
    kind: Literal["top_count", "quantile"] = "top_count",
) -> Tensor:
    """Select [B] indexes, preserving each learner's pre-shuffle order.

    Top-count sorts by magnitude with stable ties; quantiles retain row order.
    """
    if kind == "quantile":
        return selected_moves(advantages, 1 - fraction, minimum).nonzero().flatten()
    magnitude = advantages.abs()
    count = max(1, math.ceil(len(magnitude) * fraction))
    indices = torch.argsort(magnitude, descending=True, stable=True)[:count]
    return indices[magnitude[indices] >= minimum]


def magnitude_quantile(advantages: Tensor, quantile: float) -> float:
    """The inclusive `quantile` of |advantages| over every row."""
    return float(torch.quantile(advantages.detach().abs(), quantile))


def selected_moves(advantages: Tensor, quantile: float, minimum: float) -> Tensor:
    """Inclusive magnitude quantile over all [T,E] rows; retain every tie."""
    threshold = max(magnitude_quantile(advantages, quantile), minimum)
    return advantages.detach().abs() >= threshold


def selection_mask(
    advantages: Tensor,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> Tensor:
    """Boolean mask shaped like `advantages` keeping `fraction` of rows, |A| >= minimum."""
    mask = torch.zeros_like(advantages, dtype=torch.bool).flatten()
    mask[selected_rows(advantages.flatten(), fraction, minimum, kind=kind)] = True
    return mask.reshape_as(advantages)


def selected_mean(values: Tensor, selected: Tensor) -> Tensor:
    """Selected mean, on the host for MPS; empty support has zero gradient."""
    values, selected = float64_operand(values), float64_operand(selected)
    return values[selected].sum() / selected.sum().clamp_min(1)
