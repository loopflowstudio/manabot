"""Distributional critics retain outcome information and ordinary serving ABI."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import SeatRoutedCollector
import managym


def _agent(kind: Literal["scalar", "categorical_wdl"]) -> Agent:
    return Agent(
        ObservationSpace(),
        AgentHypers(hidden_dim=8, attention_on=False, value_kind=kind),
    )


def _collector(agent: Agent) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        agent.observation_space, Match(), Reward(RewardHypers()), num_envs=2, seed=17
    )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_value_interface_and_collection_likelihoods(
    kind: Literal["scalar", "categorical_wdl"], tmp_path: Path
) -> None:
    agent = _agent(kind)
    collector = _collector(agent)
    batch = collector.collect(agent, 4)
    obs = {
        key: torch.from_numpy(value.reshape((-1,) + value.shape[2:]))
        for key, value in batch.obs.items()
    }
    policy, raw_value = agent.forward_distribution(obs)
    ordinary_policy, value = agent(obs)
    assert torch.equal(policy, ordinary_policy)
    if kind == "categorical_wdl":
        assert raw_value.shape == (8, 3)
        probabilities = raw_value.softmax(-1)
        torch.testing.assert_close(value, probabilities[:, 2] - probabilities[:, 0])
        assert batch.outcome_probabilities is not None
        np.testing.assert_allclose(
            batch.outcome_probabilities.reshape(-1, 3),
            probabilities.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        assert raw_value.shape == (8, 1)
        torch.testing.assert_close(value, raw_value[:, 0])
        assert batch.outcome_probabilities is None
    np.testing.assert_allclose(value.detach().numpy(), batch.values.ravel(), atol=1e-6)
    behavior = torch.distributions.Categorical(logits=policy)
    np.testing.assert_allclose(
        behavior.log_prob(torch.from_numpy(batch.actions.ravel())).detach().numpy(),
        batch.logprobs.ravel(),
        atol=1e-6,
    )
    torch.testing.assert_close(agent.get_value(obs), value)

    path = tmp_path / "critic.pt"
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "hypers": {
                "agent_hypers": agent.hypers.model_dump(),
                "observation_hypers": agent.observation_space.encoder.hypers.model_dump(),
            },
            "world_binding": checkpoint_world(
                Match().to_rust(), agent.observation_space
            ),
        },
        path,
    )
    loaded, _ = load_checkpoint_agent(str(path))
    assert loaded.hypers.value_kind == kind
    loaded_policy, loaded_value = loaded.forward_distribution(obs)
    torch.testing.assert_close(loaded_policy, policy, rtol=0, atol=0)
    torch.testing.assert_close(loaded_value, raw_value, rtol=0, atol=0)


def test_outcome_supervision_distinguishes_draw_from_balanced_win_loss() -> None:
    # Both targets have signed mean zero; categorical cross-entropy must still
    # train different distributions rather than collapse to a scalar target.
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
    F.cross_entropy(logits, targets, reduction="sum").backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, logits.softmax(-1).detach() - targets)
    assert not torch.equal(logits.grad[0], logits.grad[1])


def test_distributional_supervision_reaches_shared_encoder() -> None:
    agent = _agent("categorical_wdl")
    collector = _collector(agent)
    batch = collector.collect(agent, 1)
    obs = {key: torch.from_numpy(value[0]) for key, value in batch.obs.items()}
    _, logits = agent.forward_distribution(obs)
    F.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in agent.player_embedding.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.policy_head.parameters())


def test_categorical_policy_and_value_do_not_read_hidden_deal() -> None:
    engine = managym.Env(seed=29, skip_trivial=True)
    engine.reset(Match().to_rust())
    actor = SemanticDecisionContract.from_env(engine).frame.actor
    agent = _agent("categorical_wdl")
    space = agent.observation_space
    original = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in space.encode(engine.observation_for_player(actor)).items()
    }
    expected = agent.forward_distribution(original)
    for seed in (17, 91):
        hidden = engine.clone_env()
        hidden.determinize(seed, perspective=actor)
        observation = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in space.encode(hidden.observation_for_player(actor)).items()
        }
        for key in original:
            torch.testing.assert_close(original[key], observation[key], rtol=0, atol=0)
        actual = agent.forward_distribution(observation)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
