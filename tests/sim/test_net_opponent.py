"""Tests for the seat-routed net-opponent training path (exp-11 / C8)."""

import numpy as np
import pytest

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentSpec, MatchHypers, RewardHypers
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
    return Agent(ObservationSpace(), AgentSpec(attention_on=False))


def _check_batch(batch, num_steps, num_envs):
    shapes = ObservationSpace().shapes
    assert batch.obs.keys() == batch.next_obs.keys() == shapes.keys()
    for key, shape in shapes.items():
        assert batch.obs[key].shape == (num_steps, num_envs, *shape)
        assert batch.next_obs[key].shape == (num_envs, *shape)
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


def test_terminal_credit_and_bootstrap():
    import torch

    from manabot.sim.net_opponent import transition_gae

    rewards = torch.tensor([[0.0], [1.0], [-1.0], [0.0]])
    advantages, returns = transition_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.tensor([[False], [True], [True], [False]]),
        torch.tensor([0.4]),
        1.0,
        1.0,
    )
    torch.testing.assert_close(returns[:, 0], torch.tensor([1.0, 1.0, -1.0, 0.4]))


def test_update_boundary_has_no_banked_or_sampled_actions():
    collector = _make_collector("self")
    agent = _make_agent()
    first = collector.collect(agent, 7)
    assert collector.stats.learner_transitions == 28
    assert all(not stream for stream in collector._streams)
    assert all(pending is None for pending in collector._pending)
    for key in first.next_obs:
        np.testing.assert_array_equal(first.next_obs[key], collector._buffers[key])
    collector.collect(agent, 7)
    assert collector.stats.learner_transitions == 56


def test_paused_buffer_rows_are_unchanged_and_clear_step_flags():
    collector = _make_collector("self", num_envs=2)
    before = {k: v[0].copy() for k, v in collector._buffers.items()}
    collector._buffers["terminated"][0] = 1
    collector._buffers["rewards"][0] = 7
    collector._env.step_into_buffers([-999, 0], [False, True])
    for key in collector._buffers:
        if key in {"terminated", "truncated", "rewards"}:
            assert collector._buffers[key][0] == 0
        else:
            np.testing.assert_array_equal(collector._buffers[key][0], before[key])


def test_collection_probabilities_and_bootstrap_survive_weight_change():
    import torch

    collector = _make_collector("self")
    agent = _make_agent()
    first = collector.collect(agent, 7)
    first_probabilities = first.probabilities.copy()
    with torch.no_grad():
        agent.policy_head[-1].weight.mul_(-5)
        agent.value_head[-1].bias.add_(3.0)
    second = collector.collect(agent, 7)
    # Nonterminal streams resume at precisely the bootstrap state, under new
    # weights. Terminal streams may route the new episode's opponent first.
    for key in second.obs:
        np.testing.assert_array_equal(
            second.obs[key][0, ~first.next_done], first.next_obs[key][~first.next_done]
        )
    flat = {
        k: torch.as_tensor(v.reshape((-1,) + v.shape[2:]))
        for k, v in second.obs.items()
    }
    with torch.no_grad():
        logits, values = agent(flat)
    np.testing.assert_allclose(
        second.values.flatten(), values.numpy().flatten(), atol=1e-5
    )
    np.testing.assert_allclose(
        second.probabilities.reshape(logits.shape),
        logits.softmax(-1).numpy(),
        atol=1e-6,
    )
    selected = np.take_along_axis(
        second.probabilities, second.actions[..., None], axis=-1
    )[..., 0]
    np.testing.assert_allclose(second.logprobs, np.log(selected), atol=1e-6)
    assert np.all(second.probabilities[second.obs["actions_valid"] == 0] == 0)
    np.testing.assert_array_equal(first.probabilities, first_probabilities)
    assert collector.stats.learner_transitions == 56
