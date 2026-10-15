"""Scientific declaration preserves all cells, seeds and frozen controls."""

import pytest

from experiments.runners.complexity_campaign import ComplexityPlan


def plan() -> ComplexityPlan:
    return ComplexityPlan(
        updates=1000,
        training_seconds_per_run=7200,
        total_seconds=47 * 3600,
        evaluation_reserve_seconds=90000,
        calibration_hashes={str(n): "a" * 64 for n in range(9)},
    )


def test_declaration_and_complete_game_counts() -> None:
    frozen = plan()
    recipes = frozen.recipes()
    assert tuple(recipes) == ("l1", "l2", "l3")
    assert all(r.stages[0].updates == 1000 for r in recipes.values())
    families = [set(frozen.deals(n)) for n in (0, 500, 1000)]
    assert len(set.union(*families)) == 150
    # Per fit: one initial, two candidate and two control cohorts; both anchors,
    # fifty whole deals, eight seat/role legs. Nine independent fits overall.
    assert 9 * 5 * 2 * 50 * 8 == 36000
    assert frozen.experiment().resolve().cases[0].receipt()


def test_allocation_or_cohort_cannot_silently_shrink() -> None:
    for change in (
        {"updates": 100},
        {"updates": 1050},
        {"seed_order": (1, 2, 3)},
        {"total_seconds": 3600},
        {"calibration_hashes": {}},
    ):
        with pytest.raises(ValueError):
            ComplexityPlan.model_validate({**plan().model_dump(), **change})
