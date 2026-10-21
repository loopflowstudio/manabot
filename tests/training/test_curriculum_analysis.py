"""The curriculum's pre-registered rules, on synthetic seed effects."""

import pytest

from experiments.runners.curriculum_analysis import (
    AnchorEvidence,
    Interval,
    PairedDifference,
    SeedSummary,
    anchor_validation,
    six_seed_summary,
    time_to_progress,
    verdict,
)
from experiments.runners.curriculum_campaign import SEEDS


def _summary(values: list[float], opponent: str = "heuristic") -> SeedSummary:
    effects = tuple(
        PairedDifference(seed, 64, opponent, (), Interval(v, v, v), 50)
        for seed, v in zip(SEEDS, values)
    )
    return six_seed_summary(effects, SEEDS)


NEUTRAL = [0.0, 0.01, -0.01, 0.0, 0.01, -0.01]


def test_six_seed_interval_and_missing_seeds() -> None:
    summary = _summary([0.06, 0.08, 0.07, 0.09, 0.05, 0.07])
    assert summary.available and summary.positive_seeds == 6
    assert summary.interval is not None
    assert summary.interval.mean == pytest.approx(0.07)
    assert summary.interval.lower > 0.05
    partial = _summary([0.06, 0.08, 0.07])
    assert not partial.available and verdict(partial, _summary(NEUTRAL)) == (
        "inconclusive"
    )
    with pytest.raises(ValueError, match="milestone or anchor"):
        six_seed_summary(
            (
                PairedDifference(SEEDS[0], 64, "heuristic", (), Interval(0, 0, 0), 50),
                PairedDifference(SEEDS[1], 128, "heuristic", (), Interval(0, 0, 0), 50),
            ),
            SEEDS,
        )


@pytest.mark.parametrize(
    ("values", "random_anchor", "expected"),
    [
        ([0.06, 0.08, 0.07, 0.09, 0.05, 0.07], NEUTRAL, "useful"),
        # Strong mean but only four seeds positive.
        ([0.2, 0.2, 0.2, 0.2, -0.01, -0.01], NEUTRAL, "inconclusive"),
        # Competence regresses against random.
        (
            [0.06, 0.08, 0.07, 0.09, 0.05, 0.07],
            [-0.05, -0.06, -0.05, -0.07, -0.05, -0.06],
            "inconclusive",
        ),
        (NEUTRAL, NEUTRAL, "no useful effect"),
        ([-0.06, -0.08, -0.07, -0.09, -0.05, -0.07], NEUTRAL, "harmful"),
        ([0.1, -0.1, 0.12, -0.08, 0.0, 0.03], NEUTRAL, "inconclusive"),
    ],
)
def test_frozen_verdict(
    values: list[float], random_anchor: list[float], expected: str
) -> None:
    assert verdict(_summary(values), _summary(random_anchor, "random")) == expected


def test_time_to_progress_is_right_censored() -> None:
    assert time_to_progress({32: 0.04, 64: 0.12, 128: 0.2}) == 64
    assert time_to_progress({32: 0.04, 64: 0.08, 128: 0.09}) is None


def _evidence(**update: object) -> AnchorEvidence:
    milestones = {0: (0.1, 0.2), 64: (0.35, 0.5), 128: (0.4, 0.55)}
    fields: dict[str, object] = {
        "heuristic_vs_greedy": Interval(0.62, 0.56, 0.68),
        "heuristic_scores": {
            (seed, m): h for seed in SEEDS[:3] for m, (h, _) in milestones.items()
        },
        "greedy_scores": {
            (seed, m): g for seed in SEEDS[:3] for m, (_, g) in milestones.items()
        },
        "heuristic_gains_128": {seed: 0.3 for seed in SEEDS[:3]},
    }
    return AnchorEvidence(**{**fields, **update})  # type: ignore[arg-type]


def test_heuristic_decides_only_when_validated() -> None:
    assert anchor_validation(_evidence()).anchor == "heuristic"
    weaker = anchor_validation(
        _evidence(heuristic_vs_greedy=Interval(0.52, 0.46, 0.58))
    )
    assert weaker.anchor == "scripted_greedy"
    assert weaker.failures == ("heuristic does not beat greedy head-to-head",)
    saturated = _evidence()
    saturated.heuristic_scores[(SEEDS[0], 128)] = 0.85
    saturated.greedy_scores[(SEEDS[0], 128)] = 0.9
    assert anchor_validation(saturated).failures == (
        "an endpoint exceeds 80% against heuristic",
    )
    flat = anchor_validation(_evidence(heuristic_gains_128={SEEDS[0]: 0.3}))
    assert flat.failures == ("heuristic does not show all three seeds improving",)
    with pytest.raises(ValueError, match="different checkpoints"):
        anchor_validation(_evidence(greedy_scores={}))
