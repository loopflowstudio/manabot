"""Advantage selection and censored rollout diagnostics shared by both learners.

Distances count same-viewer transitions to a terminal observed in this batch.
Unfinished tails are unknown, never labeled distant or terminal. Residuals are
lambda-target minus collection prediction, not independent ground-truth error.
"""

import math
from typing import Literal, TypedDict

import torch
from torch import Tensor

from manabot.training.health import NumericalHealth


class SelectionGroup(TypedDict):
    action_type: int
    terminal_distance: str
    rows: int
    retained: int
    advantage_quantiles: list[float]
    retained_advantage_quantiles: list[float]
    residual_abs_mean: float
    retained_residual_abs_mean: float | None


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


def selected_moves(advantages: Tensor, quantile: float, minimum: float) -> Tensor:
    """Inclusive magnitude quantile over all [T,E] rows; retain every tie."""
    magnitude = advantages.detach().abs()
    threshold = max(float(torch.quantile(magnitude, quantile)), minimum)
    return magnitude >= threshold


def selection_mask(
    advantages: Tensor,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> Tensor:
    if kind == "quantile":
        return selected_moves(advantages, 1 - fraction, minimum)
    mask = torch.zeros_like(advantages, dtype=torch.bool).flatten()
    mask[selected_rows(advantages.flatten(), fraction, minimum)] = True
    return mask.reshape_as(advantages)


def selected_mean(values: Tensor, selected: Tensor) -> Tensor:
    """Mean over selected rows; empty support yields zero with zero gradient."""
    return values[selected].sum() / selected.sum().clamp_min(1)


def terminal_distances(ends: Tensor) -> Tensor:
    """Return [T,E] distances (terminal=0, censored=-1), reset at each end."""
    distances = torch.full_like(ends, -1, dtype=torch.long)
    tail = distances[-1].clone()
    for step in reversed(range(len(ends))):
        tail = torch.where(ends[step], 0, torch.where(tail >= 0, tail + 1, -1))
        distances[step] = tail
    return distances


@torch.no_grad()
def selection_diagnostics(
    advantages: Tensor,
    residuals: Tensor,
    selected: Tensor,
    ends: Tensor,
    action_types: Tensor,
) -> list[SelectionGroup]:
    """Raw/retained signed quartiles and residual magnitudes per type/distance.

    Fixed descriptive bins are not a scientific short/long horizon definition.
    Cross-stratification permits comparisons within distance and action type;
    selection-correlated residuals alone cannot establish critic error causality.
    """
    distances = terminal_distances(ends)
    result: list[SelectionGroup] = []
    bins = {
        "terminal": distances == 0,
        "1-4": (distances >= 1) & (distances <= 4),
        "5+": distances >= 5,
        "censored": distances < 0,
    }
    quantiles = advantages.new_tensor([0, 0.25, 0.5, 0.75, 1])
    for kind in action_types.unique().tolist():
        for label, distance_mask in bins.items():
            mask = distance_mask & (action_types == kind)
            if not mask.any():
                continue
            retained = mask & selected
            result.append(
                SelectionGroup(
                    action_type=int(kind),
                    terminal_distance=label,
                    rows=int(mask.sum()),
                    retained=int(retained.sum()),
                    advantage_quantiles=torch.quantile(
                        advantages[mask], quantiles
                    ).tolist(),
                    retained_advantage_quantiles=(
                        torch.quantile(advantages[retained], quantiles).tolist()
                        if retained.any()
                        else []
                    ),
                    residual_abs_mean=float(residuals[mask].abs().mean()),
                    retained_residual_abs_mean=(
                        float(residuals[retained].abs().mean())
                        if retained.any()
                        else None
                    ),
                )
            )
    return result


class UpdateDiagnostics(TypedDict, total=False):
    """Per-iteration measurements persisted in the TrainingRun stage record."""

    numerical: NumericalHealth
    gradient: str
    value_kind: str
    rows: int
    retained: int
    optimizer_exposures: int
    actor_exposures: int
    critic_exposures: int
    iteration: int
    learning_rate: float
    tau: float
    schedule_progress: float
    schedule_clock: str
    gamma: float
    policy_lambda: float
    value_lambda: float
    reference: str
    collection_kl_coefficient: float
    retained_fraction: float
    return_mean: float
    value_residual_abs_mean: float
    advantage_abs_mean: float
    bootstrapped_tail_fraction: float
    action_types: list[int]
    selected_action_types: list[int]
    selection_groups: list[SelectionGroup]
    skipped: str
    loss: float
    policy_loss: float
    value_loss: float
    gradient_norm: float
    entropy: float
    reference_kl: float
    collection_kl: float
    behavior: str
    behavior_iteration: int
    coordinates: dict[str, int | float]
