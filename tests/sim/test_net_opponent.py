"""Tests for the seat-routed net-opponent training path (exp-11 / C8)."""

import numpy as np
import pytest

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.verify.util import INTERACTIVE_DECK


def _make_collector(opponent_mode, opponent_agent=None, num_envs=4, seed=7):
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="hero",
            villain="villain",
            hero_deck=INTERACTIVE_DECK,
            villain_deck=INTERACTIVE_DECK,
        )
    )
    reward = Reward(RewardHypers())
    return SeatRoutedCollector(
        obs_space,
        match,
        reward,
        num_envs=num_envs,
        seed=seed,
        opponent_mode=opponent_mode,
        opponent_agent=opponent_agent,
    )


def _make_agent():
    return Agent(ObservationSpace(), AgentHypers(attention_on=False))


def _check_batch(batch, num_steps, num_envs):
    assert batch.actions.shape == (num_steps, num_envs)
    assert batch.logprobs.shape == (num_steps, num_envs)
    assert batch.rewards.shape == (num_steps, num_envs)
    assert batch.dones.shape == (num_steps, num_envs)
    assert batch.values.shape == (num_steps, num_envs)
    assert batch.next_done.shape == (num_envs,)
    assert batch.obs["agent_player"].shape[:2] == (num_steps, num_envs)
    assert batch.next_obs["agent_player"].shape[0] == num_envs

    # Terminal-only reward: nonzero rewards only on done transitions, and
    # every nonzero reward is +/- 1.
    nonzero = batch.rewards != 0.0
    assert not np.any(nonzero & ~batch.dones)
    assert set(np.unique(batch.rewards[nonzero])).issubset({1.0, -1.0})

    # Every stored learner observation has at least one valid action.
    assert np.all(batch.obs["actions_valid"].sum(axis=-1) >= 1)
    # Chosen actions were valid at the time.
    steps, envs = np.meshgrid(np.arange(num_steps), np.arange(num_envs), indexing="ij")
    assert np.all(batch.obs["actions_valid"][steps, envs, batch.actions] > 0)


@pytest.mark.parametrize("mode", ["random", "self"])
def test_collector_batch_shapes_and_reward_semantics(mode):
    collector = _make_collector(mode)
    agent = _make_agent()
    batch = collector.collect(agent, num_steps=32)
    _check_batch(batch, 32, 4)
    # Seat balance: streams alternate learner seats.
    assert list(collector.learner_seat) == [0, 1, 0, 1]


def test_collector_frozen_opponent_and_streaming_continuity():
    opponent = _make_agent()
    collector = _make_collector("frozen", opponent_agent=opponent)
    agent = _make_agent()
    first = collector.collect(agent, num_steps=16)
    second = collector.collect(agent, num_steps=16)
    _check_batch(first, 16, 4)
    _check_batch(second, 16, 4)
    assert collector.stats.opponent_decisions > 0
    assert collector.stats.learner_transitions >= 2 * 16 * 4
    # The frozen opponent produced a fingerprint histogram.
    assert sum(collector.stats.opponent_action_types.values()) == (
        collector.stats.opponent_decisions
    )


def test_collector_requires_opponent_agent_for_frozen():
    with pytest.raises(ValueError):
        _make_collector("frozen", opponent_agent=None)
