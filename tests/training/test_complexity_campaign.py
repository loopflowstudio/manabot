"""Scientific declaration preserves all cells, seeds and frozen controls."""

import pytest

from experiments.runners.complexity_campaign import (
    ComplexityPlan,
    GPUComplexityPlan,
    execute,
)


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


def gpu_plan() -> GPUComplexityPlan:
    return GPUComplexityPlan.model_validate(
        {
            **plan().model_dump(),
            "schema_version": 2,
            "total_seconds": 43 * 3600,
            "evaluation_reserve_seconds": 85000,
            "max_usd_per_hour": 0.69,
            "calibration_usd": 2,
        }
    )


def test_gpu_plan_preserves_cohort_but_changes_bound_recipe() -> None:
    frozen = gpu_plan()
    for recipe in frozen.recipes().values():
        assert recipe.stages[0].streams == 64
        assert recipe.stages[0].execution.device == "cuda"
        assert recipe.stages[0].transitions == 64
    assert frozen.deals(1000) == plan().deals(1000)
    assert frozen.digest != plan().digest
    for recipe in plan().recipes().values():
        assert recipe.stages[0].streams == 4
        assert recipe.stages[0].execution.device == "cpu"


def test_gpu_plan_rejects_overspending_and_unapproved_settings() -> None:
    for change in (
        {"max_usd_per_hour": 0.80},
        {"calibration_usd": 2.01},
        {"inclusive_cap_usd": 36},
        {"settlement_reserve_usd": 0},
        {"streams": 512},
        {"device": "cpu"},
    ):
        with pytest.raises(ValueError):
            GPUComplexityPlan.model_validate({**gpu_plan().model_dump(), **change})


def test_gpu_campaign_rejects_cpu_before_starting_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        execute(gpu_plan())
