"""The pooled token stream: exact packing, table models unchanged, batch independence."""

import pytest
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.admission import admit_observation
from manabot.model.token_stream import TOKEN_BUCKET, bucket_length, pack, unpack
from manabot.verify.action_cases import decision_case, reorder_offers

SPEC = {"hidden_dim": 32, "num_attention_heads": 2, "attention_layers": 2}


def _agent(seed: int, stream: str) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(**SPEC, value_aggregation="masked_mean", token_stream=stream),
    )


def _batch(*observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([obs[key] for obs in observations]) for key in observations[0]
    }


def test_pack_round_trip_keeps_every_valid_token_in_order() -> None:
    generator = torch.Generator().manual_seed(3)
    tokens = torch.randn(5, 90, 4, generator=generator)
    valid = torch.rand(5, 90, generator=generator) < 0.3
    valid[2] = False
    valid[2, 7] = True
    packed, packed_valid, order = pack(tokens, valid)
    assert packed.shape[1] == bucket_length(int(valid.sum(1).max()), 90)
    assert packed.shape[1] % TOKEN_BUCKET == 0 or packed.shape[1] == 90
    for row in range(5):
        count = int(valid[row].sum())
        assert packed_valid[row, :count].all() and not packed_valid[row, count:].any()
        torch.testing.assert_close(packed[row, :count], tokens[row][valid[row]])
    restored = unpack(packed * packed_valid.unsqueeze(-1), order, 90)
    torch.testing.assert_close(restored, tokens * valid.unsqueeze(-1))


def test_bucket_length_is_bounded_by_the_table() -> None:
    assert bucket_length(1, 400) == TOKEN_BUCKET
    assert bucket_length(TOKEN_BUCKET + 1, 400) == 2 * TOKEN_BUCKET
    assert bucket_length(390, 400) == 400


def test_tables_stay_the_omitted_default_and_pooled_only_adds_types() -> None:
    assert "token_stream" not in ModelConfig(**SPEC).model_dump()
    assert ModelConfig(**SPEC, token_stream="pooled").model_dump()["token_stream"] == (
        "pooled"
    )
    with pytest.raises(ValueError, match="pooled token stream"):
        ModelConfig(**SPEC, token_stream="pooled", value_aggregation="value_token")
    tables, pooled = _agent(11, "tables"), _agent(11, "pooled")
    shared = tables.state_dict()
    extra = set(pooled.state_dict()) - set(shared)
    assert extra == {"token_types.weight"}
    for name, value in shared.items():
        torch.testing.assert_close(pooled.state_dict()[name], value, rtol=0, atol=0)


def test_pooled_outputs_do_not_depend_on_batch_neighbours() -> None:
    """A row's padding length changes with its batch; its outputs must not."""
    agent = _agent(5, "pooled").eval()
    first = decision_case("attack").observation
    second = decision_case("learn").observation
    with torch.no_grad():
        alone_logits, alone_value = agent(admit_observation(agent, first))
        logits, value = agent(admit_observation(agent, _batch(first, second, second)))
    legal = first["actions_valid"].bool()
    torch.testing.assert_close(logits[:1][legal], alone_logits[legal])
    torch.testing.assert_close(value[:1], alone_value)


def test_pooled_logits_follow_their_offers() -> None:
    agent = _agent(9, "pooled").eval()
    obs = decision_case("mode").observation
    count = int(obs["actions_valid"][0].sum())
    order = torch.arange(count).flip(0)
    with torch.no_grad():
        logits, _ = agent(admit_observation(agent, obs))
        moved, _ = agent(admit_observation(agent, reorder_offers(obs, order)))
    torch.testing.assert_close(moved[0, :count], logits[0, order])


def test_pooled_learning_reaches_types_and_attention() -> None:
    agent = _agent(2, "pooled").train()
    obs = _batch(decision_case("target").observation, decision_case("land").observation)
    logits, value = agent.forward_distribution(admit_observation(agent, obs))
    legal = obs["actions_valid"].bool()
    loss = -torch.log_softmax(logits, -1)[legal].mean() + value.square().mean()
    loss.backward()
    assert agent.token_types is not None and agent.token_types.weight.grad is not None
    assert agent.token_types.weight.grad.abs().sum() > 0
    assert agent.attention.mha.in_proj_weight.grad.abs().sum() > 0
