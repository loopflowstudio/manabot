"""The Elemental closure and lambda studies vary the frozen recipe one field at a time."""

import pytest

from experiments.elemental_grid import DEAL_BASE, DECKS, GAMES_PER_CELL
from experiments.elemental_long import H2H_DEAL_BASE, H2H_GAMES_PER_CELL, REGIME
from experiments.elemental_variants import (
    CLOSURE_COMPLETE_FLOOR,
    closure_comparison,
    lambda_sweep,
)
from experiments.learning_signal import DEALS, ELEMENTAL_PAIRS
from manabot.infra.identity import canonical_sha256
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

# The protocol's pairings, fixed before scoring.
PAIRINGS_SHA256 = "9bff729243f7962406520412e37d023fb6a6619e2eb996de43ef85ec5859503f"


def _changed(first: dict[str, object], second: dict[str, object]) -> set[str]:
    return {
        key for key in first.keys() | second.keys() if first.get(key) != second.get(key)
    }


def _stage(regime: TrainingRegime) -> TrainSelfPlay:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return stage


def test_variants_relabel_the_frozen_recipe_to_the_current_world() -> None:
    frozen = TrainingRegime.model_validate_json(REGIME.read_bytes()).model_dump(
        mode="json"
    )
    current = validate_regime(lambda_sweep(0.5, 0.8, 75)).model_dump(mode="json")
    assert current["world"] == managym.WORLD_VERSION
    assert _changed(frozen, current) == {"id", "world", "stages"}
    assert _changed(frozen["stages"][0], current["stages"][0]) == {
        "updates",
        "game_errors",
    }
    assert _changed(
        frozen["stages"][0]["game_errors"], current["stages"][0]["game_errors"]
    ) == {"recover_capacity"}


def test_closure_arms_learn_from_one_seat_and_differ_only_in_closure() -> None:
    exact = validate_regime(closure_comparison("exact-quota", 150, quota=90))
    complete = validate_regime(closure_comparison("complete-games", 150))
    for arm in (exact, complete):
        stage = _stage(arm)
        assert stage.learning_seats == "one"
        assert stage.game_errors is None
        assert stage.matchup_curriculum == "elemental-uniform"
    assert _stage(exact).transitions == 90
    assert _stage(complete).transitions == CLOSURE_COMPLETE_FLOOR
    first, second = (arm.model_dump(mode="json") for arm in (exact, complete))
    assert _changed(first, second) == {"id", "stages"}
    assert _changed(first["stages"][0], second["stages"][0]) == {
        "batch_closure",
        "transitions",
    }


def test_exact_arm_needs_its_calibrated_quota() -> None:
    with pytest.raises(ValueError, match="calibration"):
        closure_comparison("exact-quota", 150, quota=None)


def test_lambda_arms_keep_two_seats_and_differ_only_in_their_traces() -> None:
    first = validate_regime(lambda_sweep(0.5, 0.8, 75))
    second = validate_regime(lambda_sweep(0.95, 1.0, 75))
    assert _stage(first).learning_seats == "both"
    assert _stage(first).game_errors is not None
    a, b = (arm.model_dump(mode="json") for arm in (first, second))
    assert _changed(a, b) == {"id", "stages"}
    assert _changed(a["stages"][0], b["stages"][0]) == {"learning"}
    assert _changed(a["stages"][0]["learning"], b["stages"][0]["learning"]) == {
        "policy_lambda",
        "value_lambda",
    }


def test_elemental_pairings_are_fixed_distinct_pairs_on_reserved_deals() -> None:
    pairings = ELEMENTAL_PAIRS.pairings
    assert len(pairings) == 50
    assert canonical_sha256([p.model_dump() for p in pairings]) == PAIRINGS_SHA256
    assert ELEMENTAL_PAIRS.legs == 4
    deals = [pairing.deal for pairing in pairings]
    assert len(set(deals)) == len(deals)
    for pairing in pairings:
        assert pairing.role_a != pairing.role_b
        assert {pairing.role_a, pairing.role_b} <= set(DECKS)
    grid = set(range(DEAL_BASE, DEAL_BASE + 100 * 1_000 + GAMES_PER_CELL))
    pairs = set(range(H2H_DEAL_BASE, H2H_DEAL_BASE + 100 * 1_000 + H2H_GAMES_PER_CELL))
    monitoring = set(range(1_910_101_000, 1_910_101_025))
    assert not set(deals) & (grid | pairs | monitoring | set(DEALS))
