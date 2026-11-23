"""Censored rollout diagnostics of the player's advantage selection.

Each source applies the backend's magnitude filter to its eligible rows.
Host diagnostics group the result by source, action type and terminal distance.

Distances count same-viewer transitions to a terminal observed in this batch.
Unfinished tails are unknown, never labeled distant or terminal. Residuals are
lambda-target minus collection prediction, not independent ground-truth error.
"""

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike

from manabot.policy.returns import magnitude_quantile, quantile, terminal_distances
from manabot.training.rollout.adaptive import AdaptiveDiagnostics
from manabot.training.updates.backend import Array, Device, UpdateBackend
from manabot.training.updates.health import NumericalHealth

_QUARTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


class SourceDiagnostics(TypedDict):
    eligible: int
    retained: int
    retention_rate: NotRequired[float]
    threshold: NotRequired[float]


@dataclass(frozen=True)
class SourceSelection:
    source: str
    eligible: int
    retained: int
    threshold: float | None
    floor_binds: bool | None

    def diagnostics(self) -> SourceDiagnostics:
        result = SourceDiagnostics(eligible=self.eligible, retained=self.retained)
        if self.threshold is not None:
            result.update(
                retention_rate=self.retained / self.eligible, threshold=self.threshold
            )
        return result


def select_sources(
    backend: UpdateBackend,
    device: Device,
    advantages: Array,
    eligible: np.ndarray,
    sources: np.ndarray | None,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> tuple[np.ndarray, tuple[SourceSelection, ...]]:
    """Filter eligible flat rows per source; return one stable pre-shuffle order."""
    strata = (0,) if sources is None else np.unique(sources)
    parts: list[np.ndarray] = []
    summaries: list[SourceSelection] = []
    flat = advantages.flatten()
    for source in strata:
        rows = np.flatnonzero(
            eligible if sources is None else eligible & (sources == source)
        )
        cutoff = None
        kept = rows
        if len(rows):
            values = (
                flat
                if len(rows) == eligible.size
                else flat[backend.arrays.from_numpy(rows, device)]
            )
            indexes, cutoff = backend.rl.selected_magnitudes(
                values, kind, fraction, minimum
            )
            kept = rows[backend.arrays.to_numpy(indexes)]
            parts.append(kept)
        summaries.append(
            SourceSelection(
                ("fresh", "snapshot")[source],
                len(rows),
                len(kept),
                None if cutoff is None else max(cutoff, minimum),
                None if cutoff is None else minimum > cutoff,
            )
        )
    if not parts:
        return np.empty(0, dtype=np.intp), tuple(summaries)
    order = parts[0]
    if len(parts) > 1:
        order = np.sort(np.concatenate(parts))
        if kind == "top_count":
            order = order[
                np.argsort(-np.abs(backend.arrays.to_numpy(flat)[order]), kind="stable")
            ]
    return order, tuple(summaries)


class SelectionGroup(TypedDict):
    action_type: int
    terminal_distance: str
    rows: int
    retained: int
    advantage_quantiles: list[float]
    retained_advantage_quantiles: list[float]
    residual_abs_mean: float
    retained_residual_abs_mean: float | None


def selection_diagnostics(
    advantages: ArrayLike,
    residuals: ArrayLike,
    selected: ArrayLike,
    ends: ArrayLike,
    action_types: ArrayLike,
    segment_ends: np.ndarray | None = None,
) -> list[SelectionGroup]:
    """Raw/retained signed quartiles and residual magnitudes per type/distance.

    Every argument is [T,E] host rows. Fixed descriptive bins are not a
    scientific short/long horizon definition. Cross-stratification permits
    comparisons within distance and action type; selection-correlated
    residuals alone cannot establish critic error causality.
    """
    advantages = np.asarray(advantages, np.float32)
    residuals = np.asarray(residuals, np.float32)
    selected = np.asarray(selected, bool)
    action_types = np.asarray(action_types)
    distances = terminal_distances(np.asarray(ends, bool), segment_ends)
    result: list[SelectionGroup] = []
    bins = {
        "terminal": distances == 0,
        "1-4": (distances >= 1) & (distances <= 4),
        "5+": distances >= 5,
        "censored": distances < 0,
    }
    for kind in np.unique(action_types).tolist():
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
                    advantage_quantiles=[
                        quantile(advantages[mask], q) for q in _QUARTILES
                    ],
                    retained_advantage_quantiles=(
                        [quantile(advantages[retained], q) for q in _QUARTILES]
                        if retained.any()
                        else []
                    ),
                    residual_abs_mean=float(np.abs(residuals[mask]).mean()),
                    retained_residual_abs_mean=(
                        float(np.abs(residuals[retained]).mean())
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


def value_scale(
    predictions: ArrayLike,
    targets: ArrayLike,
    advantages: ArrayLike,
    *,
    kind: Literal["top_count", "quantile"],
    quantile: float,
    minimum: float,
    sources: tuple[SourceSelection, ...] | None = None,
) -> ValueScale:
    """Summarize critic/advantage scale in float64 over host rows; see
    ValueScale for target semantics."""
    predictions = np.asarray(predictions, np.float64)
    targets = np.asarray(targets, np.float64)
    target_variance = float(targets.var())
    threshold: float | None = None
    floor_binds: bool | None = None
    if kind == "quantile" and sources is not None:
        eligible_sources = [s for s in sources if s.eligible]
        if len(eligible_sources) == 1:
            threshold = eligible_sources[0].threshold
            floor_binds = eligible_sources[0].floor_binds
    elif kind == "quantile":
        empirical = magnitude_quantile(np.asarray(advantages, np.float32), quantile)
        threshold = max(empirical, minimum)
        floor_binds = minimum > empirical
    return ValueScale(
        prediction_mean=float(predictions.mean()),
        prediction_std=float(predictions.std()),
        target_mean=float(targets.mean()),
        target_std=target_variance**0.5,
        explained_variance=(
            1 - float((targets - predictions).var()) / target_variance
            if target_variance > 0
            else None
        ),
        advantage_std=float(np.asarray(advantages, np.float64).std()),
        selection_threshold=threshold,
        floor_binds=floor_binds,
    )


class UpdateDiagnostics(TypedDict, total=False):
    """Per-iteration measurements, logged once as the update's Etude metrics."""

    decision_memory: dict[str, float]
    adaptive: AdaptiveDiagnostics

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
    reference_forward_kl: float
    reference_forward_kl_coefficient: float
    collection_kl_coefficient: float
    retained_fraction: float
    return_mean: float
    value_residual_abs_mean: float
    advantage_abs_mean: float
    bootstrapped_tail_fraction: float
    action_types: list[int]
    selected_action_types: list[int]
    source_selection: dict[str, SourceDiagnostics]
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
