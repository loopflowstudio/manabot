"""Capacity receipts bind model meaning across ordinary export and reload."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentSpec, RewardHypers
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_identity, architecture_receipt
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import SeatRoutedCollector


@pytest.mark.parametrize("width,depth", [(64, 1), (64, 2), (128, 2)])
@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_capacity_export_and_real_legal_policy(
    tmp_path: Path,
    width: int,
    depth: Literal[1, 2],
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    torch.set_num_threads(1)
    agent = Agent(
        ObservationSpace(),
        AgentSpec(
            hidden_dim=width,
            attention_layers=depth,
            value_kind=kind,
        ),
    )
    before = torch.get_rng_state().clone()
    receipt = architecture_receipt(agent)
    assert torch.equal(before, torch.get_rng_state())
    assert receipt.parameters.total == sum(p.numel() for p in agent.parameters())
    assert receipt.parameters.trainable == receipt.parameters.total
    assert sum(p.total for p in receipt.components.values()) == receipt.parameters.total
    assert ("extra_attention" in receipt.components) == (depth == 2)
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardHypers()),
        num_envs=2,
        seed=102,
    )
    batch = collector.collect(agent, 2)
    obs = {k: torch.from_numpy(v[0]) for k, v in batch.obs.items()}
    policy, value = agent(obs)
    assert policy.shape == obs["actions_valid"].shape
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    probabilities = policy.softmax(-1)
    assert (probabilities[obs["actions_valid"] == 0] == 0).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
    path = tmp_path / "capacity.pt"
    save_bc_checkpoint(
        agent, agent.observation_space, path, player_configs=Match().to_rust()
    )
    loaded, _ = load_checkpoint_agent(str(path))
    assert architecture_receipt(loaded) == receipt
    actual = loaded(obs)
    for expected, observed in zip((policy, value), actual, strict=True):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    checkpoint = torch.load(path, weights_only=False)
    # Pooling changes critic semantics while retaining every parameter shape.
    checkpoint["hypers"]["agent_hypers"]["value_aggregation"] = "masked_mean"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="architecture receipt mismatch"):
        load_checkpoint_agent(str(path))
    checkpoint["hypers"]["agent_hypers"]["value_aggregation"] = "historical_mean"
    del checkpoint["architecture"]
    if depth == 1:
        del checkpoint["hypers"]["agent_hypers"]["attention_layers"]
        del checkpoint["hypers"]["agent_hypers"]["value_aggregation"]
    torch.save(checkpoint, path)
    historical, _ = load_checkpoint_agent(str(path))
    for expected, observed in zip((policy, value), historical(obs), strict=True):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)


def test_resolved_identity_and_accounting_follow_meaning() -> None:
    space = ObservationSpace()
    implicit = AgentSpec()
    explicit = AgentSpec.model_validate(implicit.model_dump())
    assert architecture_identity(implicit, space) == architecture_identity(
        explicit, space
    )
    masked = AgentSpec(value_aggregation="masked_mean")
    agent = Agent(space, implicit)
    other = Agent(space, masked)
    baseline = architecture_receipt(agent)
    assert baseline.parameters == architecture_receipt(other).parameters
    assert baseline.identity != architecture_receipt(other).identity
    agent.requires_grad_(False)
    frozen = architecture_receipt(agent)
    assert frozen.parameters.trainable == 0
    assert frozen.parameters.total == baseline.parameters.total
    assert frozen.identity == baseline.identity
