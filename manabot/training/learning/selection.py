"""Advantage selection and censored rollout diagnostics shared by both learners.

Distances count same-viewer transitions to a terminal observed in this batch.
Unfinished tails are unknown, never labeled distant or terminal. Residuals are
lambda-target minus collection prediction, not independent ground-truth error.
"""

import math
from typing import Literal, TypedDict

import torch
from torch import Tensor

from manabot.training.learning.health import NumericalHealth
from manabot.training.learning.returns import terminal_distances
from manabot.training.rollout.adaptive import AdaptiveDiagnostics
from manabot.training.rollout.self_play import SelfPlayOutcome


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


def _magnitude_quantile(advantages: Tensor, quantile: float) -> float:
    return float(torch.quantile(advantages.detach().abs(), quantile))


def selected_moves(advantages: Tensor, quantile: float, minimum: float) -> Tensor:
    """Inclusive magnitude quantile over all [T,E] rows; retain every tie."""
    threshold = max(_magnitude_quantile(advantages, quantile), minimum)
    return advantages.detach().abs() >= threshold


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


class ValueScale(TypedDict):
    """Critic and advantage scale over every collected row [T,E] of one batch.

    Predictions are collection-time critic outputs; targets are the update's
    lambda returns (WDL expectations for categorical critics). Explained variance
    is 1 - Var(target - prediction) / Var(target), a bootstrapped-target fit,
    not error against final outcomes; None when targets are constant.
    `selection_threshold` is the magnitude a quantile filter actually applied;
    `floor_binds` says the absolute floor, not the quantile, set it. `quantile`
    is the magnitude quantile, as in selected_moves.
    """

    prediction_mean: float
    prediction_std: float
    target_mean: float
    target_std: float
    explained_variance: float | None
    advantage_std: float
    selection_threshold: float | None
    floor_binds: bool | None


@torch.no_grad()
def value_scale(
    predictions: Tensor,
    targets: Tensor,
    advantages: Tensor,
    *,
    kind: Literal["top_count", "quantile"],
    quantile: float,
    minimum: float,
) -> ValueScale:
    """Summarize critic/advantage scale; see ValueScale for target semantics."""
    predictions = predictions.double()
    targets = targets.double()
    target_variance = float(targets.var(correction=0))
    threshold: float | None = None
    floor_binds: bool | None = None
    if kind == "quantile":
        empirical = _magnitude_quantile(advantages, quantile)
        threshold = max(empirical, minimum)
        floor_binds = minimum > empirical
    return ValueScale(
        prediction_mean=float(predictions.mean()),
        prediction_std=float(predictions.std(correction=0)),
        target_mean=float(targets.mean()),
        target_std=target_variance**0.5,
        explained_variance=(
            1 - float((targets - predictions).var(correction=0)) / target_variance
            if target_variance > 0
            else None
        ),
        advantage_std=float(advantages.double().std(correction=0)),
        selection_threshold=threshold,
        floor_binds=floor_binds,
    )


class UpdateDiagnostics(TypedDict, total=False):
    """Per-iteration measurements, logged once as the update's Etude metrics."""

    adaptive: AdaptiveDiagnostics

    self_play_outcomes: list[SelfPlayOutcome]
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
    value_scale: ValueScale
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
