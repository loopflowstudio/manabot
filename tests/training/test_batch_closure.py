"""The collector closes a batch at the exact quota or at complete games."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.env.match import Match, Reward
from manabot.infra.config import RewardConfig
from manabot.training.learning import BatchClosure
from manabot.training.rollout.collection import Collector, collect, open_collector
from manabot.training.state.game_errors import GameErrorPolicy
from tests.fixtures.dense_collector import dense_collector
from tests.training.test_two_seat_learning import ScriptedGames, _agent

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "fields",
    [
        {"learning_seats": "both"},
        {"learning_seats": "one", "games_per_stream": 2},
        {
            "learning_seats": "one",
            "game_errors": GameErrorPolicy(),
            "record_attempt": lambda state, _: state,
        },
    ],
)
def test_collector_rejects_exact_quota_with_whole_game_settings(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="exact quota"):
        open_collector(
            _agent().observation_space,
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode="self",
            batch_closure="exact-quota",
            **fields,
            numerics=TorchCollectionSession(device="cpu", seed=0, precision="float32"),
        )


def _one_seat(closure: BatchClosure) -> Collector:
    agent = _agent()
    collector = open_collector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="one",
        batch_closure=closure,
        numerics=TorchCollectionSession(device="cpu", seed=0, precision="float32"),
    )
    collector._env = ScriptedGames(dense_collector(collector))
    return collector


def test_exact_quota_pauses_mid_game_at_the_quota() -> None:
    agent = _agent()
    batch = collect(_one_seat("exact-quota"), agent, 2)
    assert batch.dones.shape == (2, 2)
    assert not batch.next_done.all()


def test_exact_quota_counts_trajectories_within_each_column() -> None:
    batch = collect(_one_seat("exact-quota"), _agent(), 4)
    # One three-choice game and an unfinished tail in stream 0, two whole
    # two-choice games in stream 1. Rows are interleaved in flat batch order.
    assert [list(rows) for rows in batch.trajectory_rows()] == [
        [0, 2, 4],
        [6],
        [1, 3],
        [5, 7],
    ]
    assert batch.bootstrapped_tail_fraction() == 0.25


def test_one_seat_complete_games_end_every_trajectory_at_a_result() -> None:
    agent = _agent()
    collector = _one_seat("complete-games")
    batch = collect(collector, agent, 2)
    assert batch.ordered_obs()["agent_player"].shape[1] == 1
    assert bool(batch.next_done.all())
    # Every packed trajectory closes on a terminal row, so nothing bootstraps.
    assert bool(batch.dones[-1].all())
    assert np.count_nonzero(batch.dones) >= 1
    assert collector.stats.games >= 2
    # Stream 1's shorter game leaves it idle while stream 0 finishes.
    assert collector.stats.idle_stream_steps > 0
