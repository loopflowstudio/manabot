"""Padding, compatibility, gradient and count contracts of the critic readouts."""

from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import PERMANENT_DEFINITION_INDEX, ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.rollout.collection import SeatRoutedCollector

Aggregation = Literal[
    "historical_mean", "masked_mean", "value_token", "attention_readout"
]
ReadoutAggregation = Literal["masked_mean", "value_token", "attention_readout"]
READOUTS: tuple[ReadoutAggregation, ...] = (
    "masked_mean",
    "value_token",
    "attention_readout",
)


@pytest.mark.parametrize("aggregation", ["historical_mean", *READOUTS])
@pytest.mark.parametrize("depth", [1, 2])
def test_inference_attention_preserves_padded_coordinates_and_values(
    aggregation: Aggregation, depth: Literal[1, 2]
) -> None:
    torch.manual_seed(106)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(value_aggregation=aggregation, attention_layers=depth),
    ).eval()
    # Interior holes, different supports, the last slot and asymmetric owners
    # exercise restoration at original focus coordinates, not just a prefix.
    objects = torch.randn(3, 11, 64)
    owners = torch.rand(3, 11) > 0.5
    validity = torch.zeros(3, 11)
    validity[0, [0, 3, 10]] = 1
    validity[1, [1, 3]] = 1
    validity[2, [0, 10]] = 1
    with torch.enable_grad():
        dense = agent._attend_objects(objects, owners, validity)
        expected = agent._value_from_objects(dense, validity)
    with torch.inference_mode():
        compact = agent._attend_objects(objects, owners, validity)
        actual = agent._value_from_objects(compact, validity)
    torch.testing.assert_close(compact, dense, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(compact[:, :11][validity == 0]) == 0
    # The same method still supports gradients after inference. Compaction
    # must not cache detached state or alter learning's dense representation.
    with torch.enable_grad():
        trained = agent._attend_objects(objects, owners, validity)
        agent._value_from_objects(trained, validity).square().sum().backward()
    assert agent.attention.mha.in_proj_weight.grad is not None
    assert torch.isfinite(agent.attention.mha.in_proj_weight.grad).all()


@pytest.mark.parametrize("aggregation", READOUTS)
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_padding_and_gradient_contract(
    aggregation: Aggregation,
    depth: Literal[1, 2],
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    torch.manual_seed(106)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(
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
    if agent.value_readout is not None:
        for name, parameter in agent.value_readout.named_parameters():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("depth", [1, 2])
def test_token_preserves_action_focus_indexes(depth: Literal[1, 2]) -> None:
    torch.manual_seed(106)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(value_aggregation="value_token", attention_layers=depth),
    )
    collector = SeatRoutedCollector(
        agent.observation_space, Match(), Reward(RewardConfig()), num_envs=1, seed=29
    )
    batch = collector.collect(agent, 1)
    obs = {
        key: torch.from_numpy(value[0])
        for key, value in batch.observations().observation.items()
    }
    state = agent._gather_object_embeddings(obs)
    objects, owners, valid = state.objects, state.is_agent, state.validity
    encoded = agent._attend_objects(objects, owners, valid)
    assert encoded.shape[1] == objects.shape[1] + 1
    # Probe each real-object index, including the final slot, and absent focus.
    # Attention changes representations; the index must still select that row.
    for index in range(-1, objects.shape[1]):
        obs["action_focus"].fill_(index)
        action_rows = agent.action_embedding(obs["actions"][..., :-2])
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
        # Dense and legal-row matrix multiplies round differently in float32;
        # near-zero logits need an absolute floor below float32 epsilon.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "fields",
    [
        {"value_aggregation": "value_token", "attention_on": False},
        {"attention_layers": 2, "attention_on": False},
        {"compound_decisions": True},
        {"hidden_dim": 7, "num_attention_heads": 4},
        {"attention_layers": 0},
    ],
)
def test_invalid_architecture_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelConfig.model_validate(fields)


def test_missing_fields_preserve_historical_initialization() -> None:
    torch.manual_seed(19)
    implicit = ActorCritic(ObservationSpace(), ModelConfig())
    torch.manual_seed(19)
    explicit = ActorCritic(
        ObservationSpace(),
        ModelConfig(value_aggregation="historical_mean", attention_layers=1),
    )
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    assert not any(
        "extra_attention" in key or "value_token" in key
        for key in implicit.state_dict()
    )


@pytest.mark.parametrize("aggregation", ["historical_mean", *READOUTS])
def test_real_forward_padding_and_historical_equation(aggregation: Aggregation) -> None:
    torch.manual_seed(106)
    agent = ActorCritic(ObservationSpace(), ModelConfig(value_aggregation=aggregation))
    collector = SeatRoutedCollector(
        agent.observation_space, Match(), Reward(RewardConfig()), num_envs=1, seed=29
    )
    batch = collector.collect(agent, 1)
    obs = {
        key: torch.from_numpy(value[0])
        for key, value in batch.observations().observation.items()
    }
    expected = agent.forward_distribution(obs)
    changed = {key: value.clone() for key, value in obs.items()}
    for key in (
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
    ):
        padding = changed[key + "_valid"] == 0
        changed[key][padding] = 123
        if key.endswith("permanents"):
            # Categorical IDs must remain admitted even in unused slots. Vary a
            # valid identity alongside arbitrary numeric padding features.
            changed[key][..., PERMANENT_DEFINITION_INDEX][padding] = 1
    actual = agent.forward_distribution(changed)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    if aggregation == "historical_mean":
        state = agent._gather_object_embeddings(obs)
        objects, owners, valid = state.objects, state.is_agent, state.validity
        post = agent.attention(objects, owners, valid == 0)
        value = agent.value_head(post)
        batches, slots = obs["actions_valid"].bool().nonzero(as_tuple=True)
        scores = agent.policy_head(
            agent._gather_informed_actions(obs, post, state, batches, slots)
        ).squeeze(-1)
        logits = scores.new_full(obs["actions_valid"].shape, -1e8)
        logits[batches, slots] = scores
        torch.testing.assert_close(expected[0], logits, rtol=0, atol=0)
        torch.testing.assert_close(expected[1], value, rtol=0, atol=0)
    else:
        for key in changed:
            if key.endswith("_valid") and key != "actions_valid":
                changed[key].zero_()
        with pytest.raises(ValueError, match="valid object"):
            agent(changed)


# -----------------------------------------------------------------------------
# Readout witnesses: permutation, capacity, multiplicity and viewer
# -----------------------------------------------------------------------------
#
# Synthetic object rows stand in for encoded objects. Rows 0 and 1 play the
# agent and opponent player objects; later rows are permanents. Each witness
# feeds the real attention stack and critic readout.


def _readout(aggregation: ReadoutAggregation, depth: int = 2) -> ActorCritic:
    torch.manual_seed(252)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(value_aggregation=aggregation, attention_layers=depth),
    )
    return agent.eval()


def _board(
    agent: ActorCritic,
    rows: torch.Tensor,
    owners: torch.Tensor,
    *,
    padding: int = 0,
) -> torch.Tensor:
    """Critic output [1] for objects [N,D] and ownership [N], padded at the end."""
    width = rows.shape[-1]
    objects = torch.cat((rows, torch.full((padding, width), 7.0)))[None]
    ownership = torch.cat((owners, torch.zeros(padding, dtype=torch.bool)))[None]
    valid = torch.cat((torch.ones(len(rows)), torch.zeros(padding)))[None]
    with torch.no_grad():
        return agent._value_from_objects(
            agent._attend_objects(objects, ownership, valid), valid
        ).squeeze(-1)


def _rows(count: int, seed: int) -> torch.Tensor:
    return torch.randn(count, 64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("aggregation", READOUTS)
def test_value_ignores_object_order_and_empty_capacity(
    aggregation: ReadoutAggregation,
) -> None:
    agent = _readout(aggregation)
    rows = _rows(9, 1)
    owners = torch.tensor([True, False, True, True, False, True, False, False, True])
    expected = _board(agent, rows, owners)
    order = torch.tensor([4, 0, 8, 2, 6, 1, 3, 7, 5])
    torch.testing.assert_close(_board(agent, rows[order], owners[order]), expected)
    # Capacities of the frozen v10 recipe (64 permanents per seat) and beyond.
    for padding in (1, 40, 128, 250):
        torch.testing.assert_close(
            _board(agent, rows, owners, padding=padding), expected
        )


@pytest.mark.parametrize("aggregation", READOUTS)
def test_value_sees_copies_alongside_unrepeated_objects(
    aggregation: ReadoutAggregation,
) -> None:
    """One versus many copies of a permanent beside single player objects."""
    agent = _readout(aggregation)
    players = _rows(2, 2)
    copy, other = _rows(2, 3)
    owners = torch.tensor([True, False])

    def board(copies: int) -> torch.Tensor:
        rows = torch.cat((players, other[None], copy.expand(copies, -1)))
        return _board(agent, rows, torch.cat((owners, torch.ones(copies + 1) > 0)))

    values = torch.stack([board(count) for count in (1, 2, 8, 40)])
    assert torch.unique(values.round(decimals=5)).numel() == len(values)


@pytest.mark.parametrize(
    ("aggregation", "sees_uniform_replication"),
    [("masked_mean", False), ("value_token", True), ("attention_readout", False)],
)
def test_uniform_replication_reaches_only_the_token(
    aggregation: ReadoutAggregation, sees_uniform_replication: bool
) -> None:
    """Softmax attention and mean pooling read counts only relative to others.

    Duplicating every valid object leaves each attention output, their mean and
    a softmax readout unchanged. The single token is not duplicated, so its share
    of attention changes. Real boards always hold single player objects, so the
    copies witness above is the operative case; this pins the mechanism.
    """
    agent = _readout(aggregation)
    rows = _rows(5, 4)
    owners = torch.tensor([True, False, True, False, True])
    once = _board(agent, rows, owners)
    twice = _board(agent, rows.repeat(2, 1), owners.repeat(2))
    assert torch.allclose(once, twice, atol=1e-5) is not sees_uniform_replication


@pytest.mark.parametrize("aggregation", READOUTS)
def test_singleton_among_repeated_objects_reaches_the_critic(
    aggregation: ReadoutAggregation,
) -> None:
    """A unique permanent still moves the value beside forty identical copies."""
    agent = _readout(aggregation)
    players = _rows(2, 5)
    copy, singleton, replacement = _rows(3, 6)
    copies = copy.expand(40, -1)
    owners = torch.tensor([True, False, *([True] * 41)])
    with_singleton = torch.cat((players, copies, singleton[None]))
    with_replacement = torch.cat((players, copies, replacement[None]))
    assert not torch.allclose(
        _board(agent, with_singleton, owners),
        _board(agent, with_replacement, owners),
        atol=1e-5,
    )


@pytest.mark.parametrize("aggregation", READOUTS)
def test_value_is_viewer_relative(aggregation: ReadoutAggregation) -> None:
    """Swapping which seat controls each object changes the acting viewer's value."""
    agent = _readout(aggregation)
    rows = _rows(6, 7)
    owners = torch.tensor([True, False, True, True, False, False])
    assert not torch.allclose(
        _board(agent, rows, owners), _board(agent, rows, ~owners), atol=1e-5
    )


def test_attention_readout_shares_initialization_and_actor_path() -> None:
    """Under one seed, the readout arm differs from masked mean only in its critic.

    The value token is drawn before later encoders, so it shifts their initial
    draws and joins actor-facing attention; it shares neither property.
    """
    torch.manual_seed(252)
    masked = ActorCritic(
        ObservationSpace(), ModelConfig(value_aggregation="masked_mean")
    )
    torch.manual_seed(252)
    readout = ActorCritic(
        ObservationSpace(), ModelConfig(value_aggregation="attention_readout")
    )
    masked_state = masked.state_dict()
    readout_state = readout.state_dict()
    added = {key for key in readout_state if key not in masked_state}
    assert added and all(key.startswith("value_readout.") for key in added)
    assert masked_state.keys() <= readout_state.keys()
    for key, value in masked_state.items():
        torch.testing.assert_close(readout_state[key], value, rtol=0, atol=0)

    collector = SeatRoutedCollector(
        masked.observation_space, Match(), Reward(RewardConfig()), num_envs=1, seed=29
    )
    batch = collector.collect(masked, 1)
    obs = {
        key: torch.from_numpy(value)
        for key, value in batch.observations(np.array([0])).observation.items()
    }
    with torch.no_grad():
        masked_logits, masked_value = masked(obs)
        readout_logits, readout_value = readout(obs)
    torch.testing.assert_close(readout_logits, masked_logits, rtol=0, atol=0)
    assert not torch.allclose(readout_value, masked_value)
