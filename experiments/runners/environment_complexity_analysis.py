"""Within-world learning contrasts for MTG-122's frozen-control experiment.

This module reads admitted arena results, never executes games. Each seed's
candidate and initialization use identical deal/role/seat cohorts. Whole deals
stay together in the conditional bootstrap; the three-seed method interval is
separate. No cross-world raw-score ranking or candidate selection occurs here.
"""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from manabot.arena.models import canonical_sha256
from manabot.training.models import ArtifactReference, TrainingCoordinates
from manabot.training.monitor_evaluation import MonitorResult


@dataclass(frozen=True)
class Interval:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class CellEffect:
    candidate_role: str
    opponent_role: str
    candidate_score: float
    control_score: float
    improvement: Interval
    games: int


@dataclass(frozen=True)
class SeedEffect:
    seed: int
    comparison_identity: str
    candidate_sha256: str
    initialization_sha256: str
    coordinates: TrainingCoordinates
    cells: tuple[CellEffect, ...]
    improvement: Interval
    deal_blocks: int
    candidate_result_sha256: str
    control_result_sha256: str
    series_identity: str


@dataclass(frozen=True)
class CohortEffect:
    available: bool
    reason: str | None
    seed_effects: tuple[SeedEffect, ...]
    improvement: Interval | None
    method: str = (
        "three paired training seeds; 95% t interval, df=2; assumption-sensitive"
    )


def _admit(result: MonitorResult) -> MonitorResult:
    checked = result.model_copy(deep=True)
    checked.validate_cohort()
    if checked.status != "completed" or result.status != "completed":
        raise ValueError("learning contrast requires complete replayed cohorts")
    if (
        not checked.protocol.include_mirrors
        or checked.protocol.authored_matchup is None
    ):
        raise ValueError("learning contrast requires an explicit four-cell matrix")
    if checked.expected_games != 8 * len(checked.protocol.deal_seeds):
        raise ValueError("learning contrast has inconsistent expected game count")
    for name in ("win", "draw", "score"):
        if getattr(checked, name) != getattr(result, name):
            raise ValueError("learning contrast has inconsistent saved estimates")
    return checked


def paired_improvement(
    candidate: MonitorResult,
    control: MonitorResult,
    *,
    initialization: ArtifactReference,
    bootstrap_seed: int = 12290,
    bootstrap_replicates: int = 10000,
) -> SeedEffect:
    """Subtract before resampling whole deals; intervals condition on this seed.

    Control coordinates must identify zero-update initialization, not an EMA or
    zero-learning-rate training stage. initialization must come from the
    producer's retained initial_raw receipt, never a selected later checkpoint.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    candidate, control = _admit(candidate), _admit(control)
    for name in (
        "run_id",
        "regime_digest",
        "training_seed",
        "key",
        "protocol",
        "opponent",
    ):
        if getattr(candidate, name) != getattr(control, name):
            raise ValueError(f"candidate/control pairing differs: {name}")
    if control.coordinates.updates != 0 or control.coordinates.optimizer_exposures != 0:
        raise ValueError("control must be the frozen zero-update initialization")
    for field in (
        "player_id",
        "player_spec",
        "compute_class_id",
        "world",
        "content_suite",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
    ):
        if getattr(candidate.candidate, field) != getattr(control.candidate, field):
            raise ValueError(f"candidate/control inference binding differs: {field}")
    if candidate.evaluation_identities != control.evaluation_identities:
        raise ValueError("candidate/control evaluator identities differ")
    if any(
        control.artifact[name] != initialization[name] for name in ("sha256", "bytes")
    ):
        raise ValueError("control differs from producer initialization receipt")
    matrix = candidate.protocol.authored_matchup
    assert matrix is not None
    if candidate.key.content_suite != matrix.content_suite:
        raise ValueError("evaluation suite differs from declared deck-role matrix")
    deals = candidate.protocol.deal_seeds
    scores = []
    for result in (candidate, control):
        by_coordinate = {(row.deal_seed, row.leg): row.score_a for row in result.rows}
        scores.append(
            np.array(
                [[by_coordinate[(seed, leg)] for leg in range(8)] for seed in deals],
                dtype=np.float64,
            )
        )
    difference = scores[0] - scores[1]
    draws = np.random.default_rng(bootstrap_seed).integers(
        len(deals), size=(bootstrap_replicates, len(deals))
    )

    def interval(legs: list[int]) -> Interval:
        per_deal = difference[:, legs].mean(axis=1)
        lower, upper = np.quantile(per_deal[draws].mean(axis=1), [0.025, 0.975])
        return Interval(float(per_deal.mean()), float(lower), float(upper))

    cells = []
    for a, b in (
        (matrix.role_a, matrix.role_a),
        (matrix.role_a, matrix.role_b),
        (matrix.role_b, matrix.role_a),
        (matrix.role_b, matrix.role_b),
    ):
        legs = [
            leg
            for leg in range(8)
            if (matrix.decks(leg)[leg % 2], matrix.decks(leg)[1 - leg % 2]) == (a, b)
        ]
        assert len(legs) == 2
        cells.append(
            CellEffect(
                a,
                b,
                float(scores[0][:, legs].mean()),
                float(scores[1][:, legs].mean()),
                interval(legs),
                len(deals) * len(legs),
            )
        )
    identity = canonical_sha256(
        {
            "regime_digest": candidate.regime_digest,
            "protocol": candidate.protocol.model_dump(mode="json"),
            "key": candidate.key.model_dump(mode="json"),
            "opponent": candidate.opponent.model_dump(mode="json"),
            "updates": candidate.coordinates.updates,
            "inference": candidate.candidate.player_spec,
        }
    )
    return SeedEffect(
        candidate.training_seed,
        identity,
        candidate.artifact["sha256"],
        control.artifact["sha256"],
        candidate.coordinates.model_copy(deep=True),
        tuple(cells),
        interval(list(range(8))),
        len(deals),
        canonical_sha256(candidate.model_dump(mode="json")),
        canonical_sha256(control.model_dump(mode="json")),
        canonical_sha256(
            {
                "regime_digest": candidate.regime_digest,
                "protocol": candidate.protocol.model_dump(
                    mode="json", exclude={"deal_seeds"}
                ),
                "key": candidate.key.model_dump(mode="json"),
                "opponent": candidate.opponent.model_dump(mode="json"),
                "inference": candidate.candidate.player_spec,
            }
        ),
    )


def three_seed_summary(
    effects: tuple[SeedEffect, ...], expected_seeds: tuple[int, int, int]
) -> CohortEffect:
    """The predeclared MTG-122 three-seed interval, never a pooled-game CI."""
    if len(set(expected_seeds)) != 3:
        raise ValueError("expected seeds must be three distinct independent fits")
    actual = [effect.seed for effect in effects]
    if len(set(actual)) != len(actual) or any(
        seed not in expected_seeds for seed in actual
    ):
        raise ValueError("duplicate or unplanned training seed")
    if set(actual) != set(expected_seeds):
        return CohortEffect(False, "missing training-seed cohorts", effects, None)
    if len({effect.comparison_identity for effect in effects}) != 1:
        raise ValueError("seed effects differ in world, protocol, recipe or milestone")
    values = np.array([effect.improvement.mean for effect in effects])
    # t(.975, df=2): fixed by the declared three-seed protocol, not a normal
    # approximation or a game-level confidence interval. Do not clip at ±1.
    half_width = 4.302652729911275 * float(values.std(ddof=1)) / math.sqrt(3)
    mean = float(values.mean())
    return CohortEffect(
        True, None, effects, Interval(mean, mean - half_width, mean + half_width)
    )


ProgressClock = Literal[
    "updates",
    "learner_transitions",
    "optimizer_exposures",
    "active_training_seconds",
    "training_seconds",
]


@dataclass(frozen=True)
class ObservedProgress:
    status: Literal["interval_censored", "right_censored", "unconfirmed", "unavailable"]
    clock: ProgressClock
    threshold: float
    lower: float | None
    upper: float | None
    first_observed_update: int | None
    reason: str | None


def observed_progress(
    effects: tuple[SeedEffect, ...],
    planned_updates: tuple[int, ...],
    *,
    clock: ProgressClock = "active_training_seconds",
    threshold: float = 0.10,
) -> ObservedProgress:
    """First observed threshold crossing, confirmed at the planned endpoint.

    Bounds are observation intervals, not confidence intervals. Missing planned
    checkpoints suppress the result. training_seconds is executor elapsed time,
    not inclusive campaign cost; evaluator/transfer costs must be supplied by the
    campaign's separate cost accounting, never inferred from this clock.
    """
    if (
        len(planned_updates) < 2
        or planned_updates[0] != 0
        or tuple(sorted(set(planned_updates))) != planned_updates
        or any(type(n) is not int or n < 0 for n in planned_updates)
    ):
        raise ValueError("planned updates must start at zero and strictly increase")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("progress threshold must be finite and in (0, 1]")
    by_update = {effect.coordinates.updates: effect for effect in effects}
    if len(by_update) != len(effects) or not set(by_update).issubset(planned_updates):
        raise ValueError("duplicate or unplanned checkpoint")
    if (
        effects
        and len({(e.seed, e.series_identity, e.initialization_sha256) for e in effects})
        != 1
    ):
        raise ValueError("progress curve mixes seeds, initializations or experiments")
    if set(by_update) != set(planned_updates):
        return ObservedProgress(
            "unavailable",
            clock,
            threshold,
            None,
            None,
            None,
            "missing planned checkpoint cohorts",
        )
    ordered = [by_update[n] for n in planned_updates]
    if ordered[0].improvement.mean != 0:
        raise ValueError("initialization must have zero paired improvement")
    positions = [getattr(e.coordinates, clock) for e in ordered]
    if any(value is None for value in positions):
        return ObservedProgress(
            "unavailable",
            clock,
            threshold,
            None,
            None,
            None,
            "clock unavailable at a planned checkpoint",
        )
    values = [float(value) for value in positions if value is not None]
    if any(not math.isfinite(value) or value < 0 for value in values) or any(
        after < before for before, after in zip(values, values[1:])
    ):
        raise ValueError("progress clock is not finite and monotonic")
    crossings = [i for i, e in enumerate(ordered) if e.improvement.mean >= threshold]
    if not crossings:
        return ObservedProgress(
            "right_censored",
            clock,
            threshold,
            values[-1],
            None,
            None,
            "threshold not observed by planned endpoint",
        )
    if ordered[-1].improvement.mean < threshold:
        return ObservedProgress(
            "unconfirmed",
            clock,
            threshold,
            None,
            None,
            planned_updates[crossings[0]],
            "earlier crossing not confirmed at endpoint",
        )
    first = crossings[0]
    return ObservedProgress(
        "interval_censored",
        clock,
        threshold,
        values[first - 1],
        values[first],
        planned_updates[first],
        None,
    )
