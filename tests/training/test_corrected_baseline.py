"""Baseline admission changes the world explicitly, not frozen evidence."""

import pytest

from experiments.runners.corrected_baseline import MATCHUP, recipe, resolve_counts


def test_baseline_recipe_and_budget() -> None:
    plan = resolve_counts(
        slowest_update=3.7,
        panel_costs=(2, 2),
        remaining=53 * 3600,
        hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
    )
    assert plan.updates == 10000
    recipes = plan.recipes()
    assert tuple(recipes) == ("l3",)
    r = recipes["l3"]
    assert r.world == "w6"
    assert r.agent.semantic_pack == "ur-lessons-vs-gw-allies"
    assert r.agent.value_aggregation == "masked_mean"
    assert r.stages[0].matchup_curriculum == "cross-balanced"
    for update in plan.evaluation_updates:
        protocol = plan.protocol("l3", "random", update)
        assert protocol.authored_matchup == MATCHUP
        assert protocol.include_mirrors and len(protocol.deal_seeds) == 25
    assert 3 * 3 * 2 * 25 * 8 == 3600
    assert recipe(100, 1200).stages[0].updates == 100


def test_insufficient_capacity_does_not_shrink_cohort() -> None:
    with pytest.raises(ValueError):
        resolve_counts(
            slowest_update=50,
            panel_costs=(2, 2),
            remaining=36000,
            hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
        )


def test_midpoint_is_export_only() -> None:
    plan = resolve_counts(
        slowest_update=3.7,
        panel_costs=(2, 2),
        remaining=53 * 3600,
        hashes={k: "a" * 64 for k in ("training", "random", "scripted_greedy")},
    )
    assert plan.evaluation_updates == (0, 10000)
    with pytest.raises(ValueError):
        plan.protocol("l3", "random", 5000)
