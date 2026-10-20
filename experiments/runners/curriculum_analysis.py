"""Pre-registered analysis for the MTG-122 curriculum follow-up.

Reads admitted arena results and never plays games. Written and frozen before
any curriculum cell was scored; the rules here are the design's
(`scratch/measure-learning-progress-across-simple.md`), not fitted to data.

- `paired_difference`: two checkpoints scored on the identical protocol,
  deals and opponent, subtracted per deal and leg before resampling whole
  deals. Used for arm A − arm B and for each arm minus its frozen control.
- `six_seed_summary` and `verdict`: the cohort interval and the frozen
  useful / no-effect / harmful / inconclusive rule.
- `anchor_validation`: whether `heuristic` replaces greedy as the decision
  anchor, decided from complexity evidence before curriculum scoring.

Scores are win fractions in [0, 1]; "5 points" is 0.05.
"""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from experiments.runners.environment_complexity_analysis import (
    CellEffect,
    Interval,
    _admit,
)
from manabot.training.monitor_evaluation import MonitorResult

# t(.975, df=5) for six independent training seeds.
T_975_DF5 = 2.570581835636314
THRESHOLD = 0.05
PROGRESS = 0.10


@dataclass(frozen=True)
class PairedDifference:
    """`a` minus `b` on one protocol; intervals condition on this seed."""

    seed: int
    milestone: int
    opponent: str
    cells: tuple[CellEffect, ...]
    difference: Interval
    deal_blocks: int


def paired_difference(
    a: MonitorResult,
    b: MonitorResult,
    *,
    bootstrap_seed: int = 12290,
    bootstrap_replicates: int = 10000,
) -> PairedDifference:
    """Subtract before resampling whole deals; the four cells weigh equally.

    Unlike the complexity contrast, `a` and `b` may come from different
    TrainingRuns (two arms, or an arm and the shared initialization), so
    pairing is on everything the games depend on: protocol (deals, matrix,
    anchor), evaluation key, evaluator source, training seed, the inference
    binding, and absolute iteration. Different seeds or milestones never pair.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    a, b = _admit(a), _admit(b)
    for name in ("training_seed", "key", "protocol", "opponent"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"paired results differ: {name}")
    if a.evaluation_identities != b.evaluation_identities:
        raise ValueError("paired results differ: evaluator identities")
    for field in (
        "player_spec",
        "compute_class_id",
        "world",
        "content_suite",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
    ):
        if getattr(a.candidate, field) != getattr(b.candidate, field):
            raise ValueError(f"paired inference binding differs: {field}")
    controls = [r.coordinates.updates == 0 for r in (a, b)]
    if not any(controls) and a.coordinates.updates != b.coordinates.updates:
        raise ValueError("paired arms differ in absolute iteration")
    matrix = a.protocol.authored_matchup
    assert matrix is not None
    deals = a.protocol.deal_seeds
    scores = []
    for result in (a, b):
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
    for x, y in (
        (matrix.role_a, matrix.role_a),
        (matrix.role_a, matrix.role_b),
        (matrix.role_b, matrix.role_a),
        (matrix.role_b, matrix.role_b),
    ):
        legs = [
            leg
            for leg in range(8)
            if (matrix.decks(leg)[leg % 2], matrix.decks(leg)[1 - leg % 2]) == (x, y)
        ]
        assert len(legs) == 2
        cells.append(
            CellEffect(
                x,
                y,
                float(scores[0][:, legs].mean()),
                float(scores[1][:, legs].mean()),
                interval(legs),
                len(deals) * len(legs),
            )
        )
    return PairedDifference(
        seed=a.training_seed,
        milestone=max(a.coordinates.updates, b.coordinates.updates),
        opponent=a.protocol.opponent,
        cells=tuple(cells),
        # Each cell has two legs, so the all-leg mean weighs cells equally.
        difference=interval(list(range(8))),
        deal_blocks=len(deals),
    )


# ---- Cohort ----------------------------------------------------------------


@dataclass(frozen=True)
class SeedSummary:
    available: bool
    reason: str | None
    effects: tuple[PairedDifference, ...]
    interval: Interval | None
    positive_seeds: int
    method: str = "six paired training seeds; 95% t interval, df=5"


def six_seed_summary(
    effects: tuple[PairedDifference, ...], expected_seeds: tuple[int, ...]
) -> SeedSummary:
    """The predeclared six-seed interval, never a pooled-game interval.

    A missing seed leaves the cohort unavailable rather than shrinking it.
    """
    if len(set(expected_seeds)) != 6:
        raise ValueError("expected seeds must be six distinct independent fits")
    actual = [effect.seed for effect in effects]
    if len(set(actual)) != len(actual) or not set(actual) <= set(expected_seeds):
        raise ValueError("duplicate or unplanned training seed")
    if len({(e.milestone, e.opponent) for e in effects}) > 1:
        raise ValueError("seed effects differ in milestone or anchor")
    positive = sum(e.difference.mean > 0 for e in effects)
    if set(actual) != set(expected_seeds):
        return SeedSummary(False, "missing training seeds", effects, None, positive)
    values = np.array([e.difference.mean for e in effects])
    half_width = T_975_DF5 * float(values.std(ddof=1)) / math.sqrt(len(values))
    mean = float(values.mean())
    return SeedSummary(
        True,
        None,
        effects,
        Interval(mean, mean - half_width, mean + half_width),
        positive,
    )


Verdict = Literal["useful", "no useful effect", "harmful", "inconclusive"]


def verdict(primary: SeedSummary, random_anchor: SeedSummary) -> Verdict:
    """The frozen rule for A − B at iteration 64.

    Useful: mean ≥ +5 points, lower bound > 0, at least five of six seeds
    positive, and the random-anchor A − B interval not entirely below zero.
    No useful effect: the interval lies inside (−5, +5). Harmful: mean ≤ −5
    with upper bound < 0. Anything else, including a missing seed, is
    inconclusive and is never rerun with more seeds until it passes.
    """
    if (
        not primary.available
        or not random_anchor.available
        or primary.interval is None
        or random_anchor.interval is None
    ):
        return "inconclusive"
    effect, competence = primary.interval, random_anchor.interval
    if (
        effect.mean >= THRESHOLD
        and effect.lower > 0
        and primary.positive_seeds >= 5
        and competence.upper >= 0
    ):
        return "useful"
    if effect.mean <= -THRESHOLD and effect.upper < 0:
        return "harmful"
    if -THRESHOLD < effect.lower and effect.upper < THRESHOLD:
        return "no useful effect"
    return "inconclusive"


def time_to_progress(gains: dict[int, float]) -> int | None:
    """First scored iteration whose gain over initialization reaches +10.

    `gains` maps absolute iteration to one seed's arm-minus-control mean. None
    means right-censored at the last scored iteration, not "never".
    """
    for milestone in sorted(gains):
        if gains[milestone] >= PROGRESS:
            return milestone
    return None


# ---- Decision anchor ---------------------------------------------------------


@dataclass(frozen=True)
class AnchorEvidence:
    """Complexity L3 evidence for seeds 12251-12253, both anchors, same deals.

    `heuristic_vs_greedy` is MTG-153's 200-game anchor pairing on the
    product Allies vs Lessons decks, the heuristic's score. `scores` maps (seed, iteration) to the checkpoint's mean score against
    each anchor; `gains_128` is each seed's paired endpoint gain over its
    initialization against heuristic.
    """

    heuristic_vs_greedy: Interval
    heuristic_scores: dict[tuple[int, int], float]
    greedy_scores: dict[tuple[int, int], float]
    heuristic_gains_128: dict[int, float]


@dataclass(frozen=True)
class AnchorDecision:
    anchor: Literal["heuristic", "scripted_greedy"]
    failures: tuple[str, ...]


def anchor_validation(evidence: AnchorEvidence) -> AnchorDecision:
    """Heuristic decides only if it is stronger, has headroom and resolves learning.

    Frozen before any curriculum cell is scored, so the curriculum result
    cannot choose its own anchor.
    """
    failures = []
    if evidence.heuristic_vs_greedy.lower <= 0.5:
        failures.append("heuristic does not beat greedy head-to-head")
    if set(evidence.heuristic_scores) != set(evidence.greedy_scores):
        raise ValueError("anchors were scored on different checkpoints")
    if any(
        evidence.heuristic_scores[k] >= evidence.greedy_scores[k]
        for k in evidence.heuristic_scores
    ):
        failures.append("a checkpoint scores no lower against heuristic")
    if any(
        score > 0.8
        for (_, milestone), score in evidence.heuristic_scores.items()
        if milestone == 128
    ):
        failures.append("an endpoint exceeds 80% against heuristic")
    if len(evidence.heuristic_gains_128) != 3 or any(
        gain <= 0 for gain in evidence.heuristic_gains_128.values()
    ):
        failures.append("heuristic does not show all three seeds improving")

    def ordering(scores: dict[tuple[int, int], float]) -> list[float]:
        return [
            float(np.mean([s for (_, m), s in scores.items() if m == milestone]))
            for milestone in (0, 64, 128)
        ]

    h, g = ordering(evidence.heuristic_scores), ordering(evidence.greedy_scores)
    if not (h[0] < h[1] <= h[2]) or (g[0] < g[1] <= g[2]) != (h[0] < h[1] <= h[2]):
        failures.append("heuristic milestone ordering differs from greedy's")
    return AnchorDecision(
        "scripted_greedy" if failures else "heuristic", tuple(failures)
    )
