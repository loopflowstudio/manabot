"""Combat metrics on collected decisions and boards with known log-odds."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.policy.protocol import BackendPolicy
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.combat_probe import (
    CombatColumns,
    combat_columns,
    combat_probe,
    combat_rows,
    gap_shifts,
)
from manabot.training.loops.self_play import (
    create,
    device_settings,
    self_play_collector,
)
from manabot.training.rollout.collection import RolloutBatch, close_collector, collect


@pytest.fixture(scope="module")
def rollout() -> tuple[RolloutBatch, ObservationSpace]:
    settings = gray_ogre_self_play(horizon=2)
    with device_settings(settings):
        learner, _ = create(settings, seed=1)
        collector = self_play_collector(settings, seed=1, start=0)
        try:
            batch = collect(collector, learner.agent, settings.transitions)
        finally:
            close_collector(collector)
    return batch, learner.agent.observation_space


def scoring_policy(columns: CombatColumns, board_aware: bool) -> Mock:
    """Declare with gap 3, optionally adding my power minus their untapped power."""

    def evaluate(rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        declared = rows["actions"][..., columns.declared]
        gap = np.full(len(declared), 3.0)
        if board_aware:
            for side, cols, sign in (
                ("agent", columns.mine, 1),
                ("opponent", columns.theirs, -1),
            ):
                board = rows[f"{side}_permanents"]
                power = board[..., cols.power] * cols.power_scale
                available = board[..., cols.is_creature] > 0
                if side == "opponent":
                    available &= board[..., cols.tapped] == 0
                gap += sign * (available * power).sum(axis=1)
        return declared * gap[:, None], np.zeros(len(gap))

    policy = Mock(spec=BackendPolicy)
    policy.prepare.side_effect = lambda rows: rows
    policy.evaluate.side_effect = evaluate
    return policy


def test_played_combat_distinguishes_lock_from_board_dependence(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    columns = combat_columns(space)
    assert columns is not None
    blind = combat_probe(scoring_policy(columns, False), batch, space, limit=16)
    aware = combat_probe(scoring_policy(columns, True), batch, space, limit=16)
    for kind in ("attack", "block"):
        prefix = f"combat/{kind}/"
        assert blind[prefix + "decisions"] == 16
        assert blind[prefix + "gap"] == 3
        assert 0 < blind[prefix + "shift_decisions/theirs"] <= 16
        assert blind[prefix + "shift_decisions/mine"] == 16
        for label, sign in (
            ("my_creatures_0_1", -1),
            ("my_creatures_9_9", 1),
            ("their_creatures_0_1_tapped", 1),
            ("their_creatures_9_9_untapped", -1),
        ):
            key = prefix + "shift/" + label
            assert blind[key] == 0
            assert aware[key] * sign > 0


def test_gap_uses_original_slots_and_omits_empty_side(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    columns = combat_columns(space)
    assert columns is not None
    attack = combat_rows(
        batch.action_data, columns.declare_attacker, columns.declared, 1
    )
    rows = batch.observations(attack).observation
    rows["opponent_permanents"].fill(0)
    rows["opponent_permanents_valid"].fill(0)
    rows["actions_valid"].fill(0)
    rows["actions_valid"][:, [2, 4]] = 1
    rows["actions"][..., columns.declared] = 0
    rows["actions"][:, 4, columns.declared] = 1
    original = rows["agent_permanents"].copy()
    metrics = gap_shifts(scoring_policy(columns, False), rows, columns)
    assert metrics == {
        "gap": 3,
        "decisions": 1,
        "shift_decisions/mine": 1,
        "shift/my_creatures_0_1": 0,
        "shift/my_creatures_9_9": 0,
    }
    np.testing.assert_array_equal(rows["agent_permanents"], original)


def test_no_combat_or_legacy_layout_omits_metrics(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    packed = batch.action_data
    noncombat = replace(
        packed,
        fields={**packed.fields, "actions": np.zeros_like(packed.fields["actions"])},
    )
    assert combat_probe(Mock(), replace(batch, action_data=noncombat), space) == {}
    for version in (9, 10):
        with pytest.warns(DeprecationWarning):
            legacy = ObservationSpace(ObservationConfig(policy_input_version=version))
        assert combat_probe(Mock(), batch, legacy) == {}


def test_shifts_in_game_units_condition_on_boards_with_creatures(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    cols = combat_columns(space)
    assert cols is not None
    indexes = combat_rows(batch.action_data, cols.declare_attacker, cols.declared, 2)
    rows = batch.observations(indexes).observation
    for key, side in (
        ("agent_permanents", cols.mine),
        ("opponent_permanents", cols.theirs),
    ):
        board = rows[key]
        board.fill(0)
        rows[f"{key}_valid"].fill(0)
        rows[f"{key}_valid"][:, :2] = 1
        # One creature and one land on each side. The second opposing board is empty.
        board[:, 0, side.is_creature] = 1
        board[:, 0, side.power] = 2 / side.power_scale
        board[:, 0, side.toughness] = 3 / side.toughness_scale
        board[:, 0, side.tapped] = 1
    rows["opponent_permanents"][1] = 0
    rows["opponent_permanents_valid"][1] = 0

    def evaluate(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Sum all slots so accidental edits to lands or padding change the result.
        gap = np.zeros(2)
        for key, side in (
            ("agent_permanents", cols.mine),
            ("opponent_permanents", cols.theirs),
        ):
            board = obs[key]
            gap += (
                board[..., side.power] * side.power_scale
                + board[..., side.toughness] * side.toughness_scale
                + board[..., side.tapped]
            ).sum(axis=1)
        return obs["actions"][..., cols.declared] * gap[:, None], np.zeros(2)

    policy = scoring_policy(cols, False)
    policy.evaluate.side_effect = evaluate
    assert gap_shifts(policy, rows, cols) == pytest.approx(
        {
            "gap": 9,
            "decisions": 2,
            "shift_decisions/mine": 2,
            "shift_decisions/theirs": 1,
            "shift/my_creatures_0_1": -4,
            "shift/my_creatures_9_9": 13,
            "shift/their_creatures_0_1_tapped": -4,
            "shift/their_creatures_9_9_untapped": 12,
        }
    )
