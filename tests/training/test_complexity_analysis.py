"""Paired-control arithmetic and independent-seed uncertainty, without training."""

from dataclasses import replace

import pytest

from experiments.runners.environment_complexity_analysis import (
    observed_progress,
    paired_improvement,
    three_seed_summary,
)
from manabot.arena.matchup import AuthoredMatchup
from manabot.training.monitor_evaluation import ArenaRow, MonitorResult
from tests.training.test_monitor_evaluation import _manifest, _rows


def cohort(seed: int = 12251) -> MonitorResult:
    result = _manifest()
    matrix = AuthoredMatchup(pack_key="fixture", role_a="swift", role_b="heavy")
    result.protocol.authored_matchup = matrix
    result.protocol.include_mirrors = True
    result.key = result.key.model_copy(update={"content_suite": matrix.content_suite})
    result.training_seed = seed
    result.coordinates.updates = 0
    result.coordinates.optimizer_exposures = 0
    result.expected_games = 16
    result.rows = []
    for raw in _rows(result):
        if raw["leg"] != 0:
            continue
        for leg in range(8):
            result.rows.append(
                ArenaRow.model_validate(
                    {
                        **raw,
                        "leg": leg,
                        "score_a": 0.0,
                        "seat_decks": list(matrix.decks(leg)),
                        "player_a_seat": leg % 2,
                    }
                )
            )
    result.validate_cohort()
    return result


def test_paired_difference_resamples_whole_deals_and_names_actual_cells() -> None:
    control = cohort()
    candidate = control.model_copy(deep=True)
    candidate.coordinates.updates = 1000
    for row in candidate.rows:
        # Candidate swift versus opponent heavy: legs 0 and 3, not 0 and 1.
        if row.deal_seed == 102 and row.leg in (0, 3):
            row.score_a = 1.0
    candidate.validate_cohort()
    effect = paired_improvement(candidate, control, initialization=control.artifact)
    assert effect.improvement.mean == 0.125
    assert (effect.improvement.lower, effect.improvement.upper) == (0, 0.25)
    assert len(effect.cells) == 4
    positive = next(c for c in effect.cells if c.improvement.mean > 0)
    assert (positive.candidate_role, positive.opponent_role) == ("swift", "heavy")
    assert positive.improvement.mean == 0.5
    assert positive.games == 4
    assert effect.deal_blocks == 2


def test_missing_and_duplicate_seeds_do_not_become_a_method_claim() -> None:
    control = cohort()
    effect = paired_improvement(control, control, initialization=control.artifact)
    expected = (12251, 12252, 12253)
    incomplete = three_seed_summary((effect,), expected)
    assert not incomplete.available and incomplete.improvement is None
    with pytest.raises(ValueError, match="duplicate"):
        three_seed_summary((effect, effect), expected)
    effects = tuple(
        replace(effect, seed=seed, improvement=replace(effect.improvement, mean=value))
        for seed, value in zip(expected, (0, 0.5, 1), strict=True)
    )
    result = three_seed_summary(effects, expected)
    assert result.improvement is not None
    assert result.improvement.mean == 0.5
    assert result.improvement.lower == pytest.approx(-0.742068855859773)
    assert result.improvement.upper == pytest.approx(1.742068855859773)


@pytest.mark.parametrize(
    "defect", ["missing", "different-deals", "updated-control", "wrong-initialization"]
)
def test_invalid_pair_is_not_scored(defect: str) -> None:
    control = cohort()
    candidate = control.model_copy(deep=True)
    initialization = dict(control.artifact)
    if defect == "missing":
        candidate.rows.pop()
    elif defect == "different-deals":
        candidate.protocol = candidate.protocol.model_copy(
            update={"deal_seeds": (101, 999)}
        )
    elif defect == "updated-control":
        control.coordinates.updates = 1
    else:
        initialization["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        paired_improvement(candidate, control, initialization=initialization)


@pytest.mark.parametrize(
    "gains,status,bounds",
    [
        ((0.0, 0.12, 0.15), "interval_censored", (0, 10)),
        ((0.0, 0.05, 0.15), "interval_censored", (10, 20)),
        ((0.0, 0.05, 0.08), "right_censored", (20, None)),
        ((0.0, 0.12, 0.05), "unconfirmed", (None, None)),
    ],
)
def test_progress_uses_observed_intervals_not_interpolation(
    gains: tuple[float, float, float],
    status: str,
    bounds: tuple[int | None, int | None],
) -> None:
    control = cohort()
    baseline = paired_improvement(control, control, initialization=control.artifact)
    points = tuple(
        replace(
            baseline,
            coordinates=baseline.coordinates.model_copy(
                update={"updates": update, "active_training_seconds": seconds}
            ),
            improvement=replace(baseline.improvement, mean=gain),
        )
        for update, seconds, gain in zip(
            (0, 500, 1000), (0, 10, 20), gains, strict=True
        )
    )
    progress = observed_progress(points, (0, 500, 1000))
    assert progress.status == status
    assert (progress.lower, progress.upper) == bounds
    assert observed_progress(points[:2], (0, 500, 1000)).status == "unavailable"
    with pytest.raises(ValueError, match="mixes seeds"):
        observed_progress((*points[:2], replace(points[2], seed=999)), (0, 500, 1000))
