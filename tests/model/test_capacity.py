"""Capacity receipts bind model meaning across ordinary export and reload."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import architecture_identity, architecture_receipt
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.facts import FactEncoder
from manabot.training.rollout.collection import SeatRoutedCollector


@pytest.mark.parametrize("width,depth", [(64, 1), (64, 2), (128, 2)])
@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_capacity_export_and_real_legal_policy(
    tmp_path: Path,
    width: int,
    depth: Literal[1, 2],
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    torch.set_num_threads(1)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(
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
        Reward(RewardConfig()),
        num_envs=2,
        seed=102,
        batch_closure="exact-quota",
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
    save_checkpoint(
        agent, agent.observation_space, path, player_configs=Match().to_rust()
    )
    loaded, _ = load_checkpoint(str(path))
    assert architecture_receipt(loaded) == receipt
    actual = loaded(obs)
    for expected, observed in zip((policy, value), actual, strict=True):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    checkpoint = torch.load(path, weights_only=False)
    # Pooling changes critic semantics while retaining every parameter shape.
    checkpoint["hypers"]["agent_hypers"]["value_aggregation"] = "masked_mean"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="architecture receipt mismatch"):
        load_checkpoint(str(path))
    checkpoint["hypers"]["agent_hypers"]["value_aggregation"] = "historical_mean"
    del checkpoint["architecture"]
    if depth == 1:
        del checkpoint["hypers"]["agent_hypers"]["attention_layers"]
        del checkpoint["hypers"]["agent_hypers"]["value_aggregation"]
    torch.save(checkpoint, path)
    historical, _ = load_checkpoint(str(path))
    for expected, observed in zip((policy, value), historical(obs), strict=True):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)


def test_resolved_identity_and_accounting_follow_meaning() -> None:
    space = ObservationSpace()
    implicit = ModelConfig()
    explicit = ModelConfig.model_validate(implicit.model_dump())
    assert architecture_identity(implicit, space) == architecture_identity(
        explicit, space
    )
    masked = ModelConfig(value_aggregation="masked_mean")
    agent = ActorCritic(space, implicit)
    other = ActorCritic(space, masked)
    baseline = architecture_receipt(agent)
    assert baseline.parameters == architecture_receipt(other).parameters
    assert baseline.identity != architecture_receipt(other).identity
    agent.requires_grad_(False)
    frozen = architecture_receipt(agent)
    assert frozen.parameters.trainable == 0
    assert frozen.parameters.total == baseline.parameters.total
    assert frozen.identity == baseline.identity


def test_deep_feedforward_identity_and_weights() -> None:
    space = ObservationSpace()
    default = ModelConfig(attention_layers=8, hidden_dim=384)
    explicit = ModelConfig(
        attention_layers=8, hidden_dim=384, attention_feedforward_dim=1536
    )
    assert architecture_identity(default, space) == architecture_identity(
        explicit, space
    )
    narrow = ModelConfig(
        attention_layers=8, hidden_dim=384, attention_feedforward_dim=768
    )
    assert architecture_identity(narrow, space) != architecture_identity(default, space)
    torch.set_num_threads(1)
    agent = ActorCritic(space, narrow)
    assert len(agent.extra_attention) == 7
    assert all(
        layer.mlp[0].out_features == 768
        for layer in [agent.attention, *agent.extra_attention]
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        ActorCritic(space, default).load_state_dict(agent.state_dict())


@pytest.mark.parametrize(
    "fields",
    [
        {"attention_layers": 0},
        {"attention_layers": -1},
        {"attention_feedforward_dim": 0},
        {"attention_feedforward_dim": -1},
        {"attention_feedforward_dim": 256, "attention_on": False},
    ],
)
def test_invalid_expansion(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelConfig.model_validate(fields)


def test_world_selector_reads_and_hashes_its_frozen_key() -> None:
    # The retained receipt used v5 with the historical 64-action capacity.
    space = ObservationSpace(
        ObservationConfig.from_checkpoint({"policy_input_version": 5})
    )
    current = ModelConfig(world="elemental-world-v2", policy_film=True)
    frozen = ModelConfig.model_validate(
        {"semantic_pack": "elemental-world-v2", "policy_film": True}
    )
    assert frozen == current
    # Receipts written before the selector was named `world`.
    assert architecture_identity(current, space) == (
        "8ebdcf30559998e753b3afe514b1bcf975f45b745c97bd2304c12034baafe0d6"
    )


@pytest.mark.parametrize(
    ("version", "rows"), [(1, 0), (2, 0), (3, 0), (4, 0), (5, 128), (6, 512)]
)
def test_fact_capacity_checkpoint_contract(
    tmp_path: Path, version: int, rows: int
) -> None:
    payload = {"policy_input_version": version} if version != 1 else {}
    config = ObservationConfig.from_checkpoint(payload)
    serialized = config.model_dump(mode="json")
    assert ("max_facts" in serialized) == (version == 6)
    assert ("policy_input_version" in serialized) == (version != 1)
    assert config.fact_rows == rows
    space = ObservationSpace(config)
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, space, path, player_configs=Match().to_rust())
    loaded, loaded_space = load_checkpoint(str(path))
    assert loaded_space.encoder.config.model_dump(mode="json") == serialized
    assert architecture_receipt(loaded) == architecture_receipt(agent)
    if version == 6:
        different = ActorCritic(
            ObservationSpace(ObservationConfig(max_facts=256)), agent.config
        )
        assert (
            architecture_receipt(different).identity
            != architecture_receipt(agent).identity
        )
        assert (
            architecture_receipt(different).parameters
            == architecture_receipt(agent).parameters
        )


@pytest.mark.parametrize("capacity", [256, 512])
def test_fact_capacity_preserves_timestamp_scale_and_gradients(capacity: int) -> None:
    torch.manual_seed(213)
    old, larger = FactEncoder(4, 128), FactEncoder(4, capacity)
    # Nonzero learned weights make numerical drift and dropped facts observable.
    with torch.no_grad():
        old.output.weight.fill_(0.25)
        old.output.bias.fill_(0.1)
    larger.load_state_dict(old.state_dict())
    facts = torch.zeros(1, capacity, 12)
    facts[0, :140, 0] = 1  # first object
    facts[0, :140, 1] = 6  # effect
    facts[0, :140, 9] = torch.arange(1, 141)
    facts[0, :140, 11] = 1
    objects = torch.zeros(1, 2, 4)
    padded = facts.clone()
    padded[:, 128:] = 0
    torch.testing.assert_close(old(facts[:, :128], objects), larger(padded, objects))
    facts.requires_grad_()
    output = larger(facts, objects)
    assert not torch.equal(output, larger(padded, objects))
    output.sum().backward()
    assert facts.grad is not None
    assert (facts.grad[0, 128:140, 9] != 0).all()
    assert facts.grad[0, 140:].count_nonzero() == 0
    with pytest.raises(ValueError, match="expected"):
        old(facts, objects)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_legacy_fact_capacity_cannot_be_overridden(version: int) -> None:
    with pytest.raises(ValueError, match="max_facts requires"):
        ObservationConfig.model_validate(
            {"policy_input_version": version, "max_facts": 512}
        )
