"""Tests for the determinized PUCT Teacher-1 reference."""

from pathlib import Path

import numpy as np
import pytest

from manabot.sim.distill import (
    ROOT_VALUE_KEY,
    VISIT_COUNT_KEY,
    generate_selfplay_shard,
    load_shards,
)
from manabot.sim.mcts import DeterminizedPuctPlayer, determinized_puct
from manabot.verify.util import INTERACTIVE_DECK
import managym


def _fresh_engine(seed: int = 0) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True)
    env.reset(
        [
            managym.PlayerConfig("hero", dict(INTERACTIVE_DECK)),
            managym.PlayerConfig("villain", dict(INTERACTIVE_DECK)),
        ]
    )
    return env


def test_puct_is_seeded_budgeted_and_does_not_mutate_root() -> None:
    first_root = _fresh_engine(seed=31)
    second_root = _fresh_engine(seed=31)
    action_count = first_root.action_count()
    acting = first_root.current_agent_index()

    first = determinized_puct(
        first_root, simulations=12, worlds=3, seed=99, max_steps=2000
    )
    second = determinized_puct(
        second_root, simulations=12, worlds=3, seed=99, max_steps=2000
    )

    assert first.simulations == 12
    assert first.worlds == 3
    assert int(first.visit_counts.sum()) == 12
    assert len(first.visit_counts) == len(first.q_values) == action_count
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert np.array_equal(first.q_values, second.q_values)
    assert first.root_value == second.root_value
    assert 0.0 <= first.root_value <= 1.0
    assert np.all((first.q_values >= 0.0) & (first.q_values <= 1.0))
    assert first.tree_nodes > first.worlds
    assert first.max_depth >= 1

    assert first_root.action_count() == action_count
    assert first_root.current_agent_index() == acting
    assert not first_root.is_game_over()


def test_puct_rejects_more_worlds_than_simulations() -> None:
    with pytest.raises(ValueError, match="worlds"):
        determinized_puct(_fresh_engine(), simulations=2, worlds=3, seed=0)
    with pytest.raises(ValueError, match="worlds"):
        DeterminizedPuctPlayer(2, worlds=3)


def test_player_publishes_visit_and_value_targets() -> None:
    root = _fresh_engine(seed=37)
    player = DeterminizedPuctPlayer(8, worlds=2, seed=41)

    class _Wrapper:
        _engine = root

    action = player.act(_Wrapper(), {})
    assert 0 <= action < root.action_count()
    assert player.last_visit_counts is not None
    assert player.last_scores is not None
    assert player.last_root_value is not None
    assert int(player.last_visit_counts.sum()) == 8
    assert player.last_visit_counts[action] == player.last_visit_counts.max()
    assert player.stats.simulations == 8
    assert player.stats.decisions == 1
    assert player.stats.tree_nodes > player.worlds
    assert player.stats.worlds_sampled == player.worlds
    assert player.stats.max_depth_sum >= 1
    assert player.stats.max_depth_max >= 1


def test_selfplay_shard_records_real_tree_targets(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00.npz"
    summary = generate_selfplay_shard(
        num_games=1,
        teacher_spec={
            "kind": "determinized_puct",
            "sims": 4,
            "worlds": 1,
            "c_puct": 1.5,
        },
        seed=43,
        out_path=shard,
    )
    dataset = load_shards([shard])

    assert summary["policy_target_kind"] == "visit_distribution"
    assert summary["value_target_kind"] == "root_value"
    assert VISIT_COUNT_KEY in dataset
    assert ROOT_VALUE_KEY in dataset
    assert np.all(dataset[VISIT_COUNT_KEY].sum(axis=1) == 4)
    assert np.all((dataset[ROOT_VALUE_KEY] >= 0) & (dataset[ROOT_VALUE_KEY] <= 1))
    assert np.all(dataset[VISIT_COUNT_KEY][dataset["actions_valid"] <= 0] == 0)
    chosen = dataset[VISIT_COUNT_KEY][
        np.arange(len(dataset["action"])), dataset["action"].astype(np.int64)
    ]
    assert np.all(chosen == dataset[VISIT_COUNT_KEY].max(axis=1))
    assert summary["search"]["tree_nodes"] > summary["search"]["worlds_sampled"]
    assert summary["search"]["max_depth_max"] >= 1
