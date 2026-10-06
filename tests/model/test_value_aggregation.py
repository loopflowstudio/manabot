"""Padding, compatibility and gradient contracts of the shared value token."""

from typing import Literal

import pytest
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentSpec, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import SeatRoutedCollector

Aggregation = Literal["historical_mean", "masked_mean", "value_token"]


@pytest.mark.parametrize("aggregation", ["masked_mean", "value_token"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_padding_and_gradient_contract(
    aggregation: Aggregation,
    depth: Literal[1, 2],
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    torch.manual_seed(106)
    agent = Agent(
        ObservationSpace(),
        AgentSpec(
            value_aggregation=aggregation, attention_layers=depth, value_kind=kind
        ),
    )
    # Exercise the real object stack with a valid prefix and variable padding.
    objects = torch.randn(2, 5, 64, requires_grad=True)
    owners = torch.tensor([[True, False, True, False, True]]).expand(2, -1)

    def evaluate(padding: int, payload: float) -> torch.Tensor:
        rows = torch.cat((objects, torch.full((2, padding, 64), payload)), 1)
        ownership = torch.cat((owners, torch.zeros(2, padding, dtype=torch.bool)), 1)
        valid = torch.cat((torch.ones(2, 5), torch.zeros(2, padding)), 1)
        encoded = agent._attend_objects(rows, ownership, valid)
        return agent._value_from_objects(encoded, valid)

    expected = evaluate(0, 0)
    torch.testing.assert_close(expected, evaluate(7, 29), atol=1e-6, rtol=1e-5)
    # Unlike padding, a changed valid object must reach the critic.
    changed = objects.detach().clone()
    changed[:, 0] = torch.randn(2, 64) * 3
    valid = torch.ones(2, 5)
    changed_value = agent._value_from_objects(
        agent._attend_objects(changed, owners, valid), valid
    )
    assert not torch.allclose(expected, changed_value, atol=1e-6, rtol=1e-5)
    expected.square().sum().backward()
    assert objects.grad is not None
    assert torch.isfinite(objects.grad).all()
    assert (objects.grad.abs().sum(-1) > 0).all()
    for name, parameter in agent.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    for module in (agent.attention, agent.value_head, *agent.extra_attention):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    if agent.value_token is not None:
        assert agent.value_token.grad is not None
        assert agent.value_token.grad.abs().sum() > 0


@pytest.mark.parametrize("depth", [1, 2])
def test_token_preserves_action_focus_indexes(depth: Literal[1, 2]) -> None:
    torch.manual_seed(106)
    agent = Agent(
        ObservationSpace(),
        AgentSpec(value_aggregation="value_token", attention_layers=depth),
    )
    collector = SeatRoutedCollector(
        agent.observation_space, Match(), Reward(RewardHypers()), num_envs=1, seed=29
    )
    batch = collector.collect(agent, 1)
    obs = {key: torch.from_numpy(value[0]) for key, value in batch.obs.items()}
    objects, owners, valid, history = agent._gather_object_embeddings(obs)
    encoded = agent._attend_objects(objects, owners, valid)
    assert encoded.shape[1] == objects.shape[1] + 1
    # Probe each real-object index, including the final slot, and absent focus.
    # Attention changes representations; the index must still select that row.
    for index in range(-1, objects.shape[1]):
        obs["action_focus"].fill_(index)
        action_rows = agent.action_embedding(obs["actions"][..., :-1])
        action_rows = action_rows * obs["actions_valid"].unsqueeze(-1)
        focus = torch.zeros_like(encoded[:, 0]) if index == -1 else encoded[:, index]
        focus_rows = focus[:, None, None, :].expand(
            -1, action_rows.shape[1], agent.max_focus_objects, -1
        )
        expected = agent.policy_head(
            agent.action_layer(torch.cat((action_rows, focus_rows.flatten(2)), dim=-1))
        ).squeeze(-1)
        expected = expected.masked_fill(obs["actions_valid"] == 0, -1e8)
        actual, _ = agent.forward_distribution(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fields",
    [
        {"value_aggregation": "value_token", "attention_on": False},
        {"attention_layers": 2, "attention_on": False},
        {"value_aggregation": "masked_mean", "compound_decisions": True},
        {"attention_layers": 2, "compound_decisions": True},
        {"hidden_dim": 7, "num_attention_heads": 4},
        {"attention_layers": 3},
    ],
)
def test_invalid_architecture_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AgentSpec.model_validate(fields)


def test_missing_fields_preserve_historical_initialization() -> None:
    torch.manual_seed(19)
    implicit = Agent(ObservationSpace(), AgentSpec())
    torch.manual_seed(19)
    explicit = Agent(
        ObservationSpace(),
        AgentSpec(value_aggregation="historical_mean", attention_layers=1),
    )
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    assert not any(
        "extra_attention" in key or "value_token" in key
        for key in implicit.state_dict()
    )


@pytest.mark.parametrize(
    "aggregation", ["historical_mean", "masked_mean", "value_token"]
)
def test_real_forward_padding_and_historical_equation(aggregation: Aggregation) -> None:
    torch.manual_seed(106)
    agent = Agent(ObservationSpace(), AgentSpec(value_aggregation=aggregation))
    collector = SeatRoutedCollector(
        agent.observation_space, Match(), Reward(RewardHypers()), num_envs=1, seed=29
    )
    batch = collector.collect(agent, 1)
    obs = {key: torch.from_numpy(value[0]) for key, value in batch.obs.items()}
    expected = agent.forward_distribution(obs)
    changed = {key: value.clone() for key, value in obs.items()}
    for key in (
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
    ):
        changed[key][changed[key + "_valid"] == 0] = 123
    actual = agent.forward_distribution(changed)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    if aggregation == "historical_mean":
        objects, owners, valid, history = agent._gather_object_embeddings(obs)
        post = agent.attention(objects, owners, valid == 0)
        value = agent.value_head(post)
        logits = agent.policy_head(
            agent._gather_informed_actions(obs, post, history)
        ).squeeze(-1)
        logits = logits.masked_fill(obs["actions_valid"] == 0, -1e8)
        torch.testing.assert_close(expected[0], logits, rtol=0, atol=0)
        torch.testing.assert_close(expected[1], value, rtol=0, atol=0)
    else:
        for key in changed:
            if key.endswith("_valid") and key != "actions_valid":
                changed[key].zero_()
        with pytest.raises(ValueError, match="valid object"):
            agent(changed)
