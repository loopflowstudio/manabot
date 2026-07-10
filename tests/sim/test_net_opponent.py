"""Tests for the seat-routed net-opponent training path (exp-11 / C8)."""

import numpy as np
import pytest
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import (
    LEGACY_PERMANENT_DIM,
    LEGACY_PLAYER_DIM,
    SeatRoutedCollector,
    port_legacy_state_dict,
)
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
    steps, envs = np.meshgrid(
        np.arange(num_steps), np.arange(num_envs), indexing="ij"
    )
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


def test_port_legacy_state_dict_maps_columns():
    encoder = ObservationSpace().encoder
    agent = _make_agent()
    state = agent.state_dict()

    # Fabricate a legacy checkpoint by shrinking the input embeddings to the
    # old dims: player loses its trailing column, permanent keeps features
    # 0..5 and its validity column (old index 6 = new last index).
    legacy = {k: v.clone() for k, v in state.items()}
    player_w = state["player_embedding.projection.0.weight"]
    perm_w = state["perm_embedding.projection.0.weight"]
    legacy["player_embedding.projection.0.weight"] = player_w[:, :LEGACY_PLAYER_DIM]
    legacy["perm_embedding.projection.0.weight"] = torch.cat(
        [
            perm_w[:, : LEGACY_PERMANENT_DIM - 1],
            perm_w[:, encoder.permanent_dim - 1 :],
        ],
        dim=1,
    )

    ported = port_legacy_state_dict(legacy, encoder)
    new_player = ported["player_embedding.projection.0.weight"]
    new_perm = ported["perm_embedding.projection.0.weight"]

    assert new_player.shape == (player_w.shape[0], encoder.player_dim)
    assert torch.equal(new_player[:, :LEGACY_PLAYER_DIM], player_w[:, :LEGACY_PLAYER_DIM])
    assert torch.all(new_player[:, LEGACY_PLAYER_DIM:] == 0)

    assert new_perm.shape == (perm_w.shape[0], encoder.permanent_dim)
    assert torch.equal(new_perm[:, :6], perm_w[:, :6])
    assert torch.all(new_perm[:, 6 : encoder.permanent_dim - 1] == 0)
    assert torch.equal(
        new_perm[:, encoder.permanent_dim - 1],
        perm_w[:, encoder.permanent_dim - 1],
    )

    # A current-world Agent accepts the ported dict and can run a forward.
    fresh = _make_agent()
    fresh.load_state_dict(ported)
    obs = {
        key: torch.zeros((1,) + value.shape[1:], dtype=torch.float32)
        for key, value in encoder.allocate(1).items()
    }
    obs["actions_valid"][0, 0] = 1.0
    obs["action_focus"] = obs["action_focus"].float()
    logits, value = fresh.forward(obs)
    assert logits.shape == (1, encoder.max_actions)


def test_port_rejects_current_world_checkpoint():
    encoder = ObservationSpace().encoder
    agent = _make_agent()
    with pytest.raises(ValueError):
        port_legacy_state_dict(agent.state_dict(), encoder)
