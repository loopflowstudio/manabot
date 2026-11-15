"""Ordinary policies consume complete outside programs and public knowledge."""

import os
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch._semantic_cards import SemanticCardEncoder
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.semantic.decision_contract import SemanticDecisionContract
from tests.model.prepared_routes import device_input
from tests.semantic.test_learn_contract import learn_root  # noqa: F401


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def lookup_encoder(request: pytest.FixtureRequest) -> SemanticCardEncoder:
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("MANABOT_REQUIRE_CUDA") == "1":
            pytest.fail("Definition lookup admission requires a real CUDA device")
        pytest.skip("CUDA lookup admission runs on the bounded rental")
    return SemanticCardEncoder("ur-lessons-vs-gw-allies", 4).to(request.param)


@pytest.mark.parametrize("pattern", ["repeated", "strided", "empty", "scalar"])
def test_definition_lookup_preserves_values_and_producer_gradients(
    lookup_encoder: SemanticCardEncoder,
    pattern: Literal["repeated", "strided", "empty", "scalar"],
) -> None:
    encoder = lookup_encoder
    device = encoder.definition_rows.device
    # Resolve transport IDs through the real catalog, including its zero row.
    admitted = (encoder.definition_rows > 0).nonzero().flatten()[:2]
    choices = torch.cat((admitted.new_zeros(1), admitted))
    if pattern == "repeated":
        # More than 3072 IDs also exercises CUDA's large-index reduction.
        ids = choices[torch.arange(4098, device=device) % 3]
    elif pattern == "strided":
        ids = choices[torch.arange(24, device=device).reshape(2, 3, 4) % 3]
        ids = ids.transpose(0, 2)
        assert not ids.is_contiguous()
    elif pattern == "empty":
        ids = choices.new_empty((2, 0, 3))
    else:
        ids = choices[0]
    rows = encoder.definition_rows[ids]
    count = encoder.catalog_shape.definitions + 1
    producer = (
        torch.arange(count * 4, device=device, dtype=torch.float32)
        .reshape(count, 4)
        .div(16)
        .requires_grad_()
    )
    bias = torch.ones(4, device=device, requires_grad=True)
    definitions = producer * 2 + bias
    definitions.retain_grad()
    actual = encoder.lookup(definitions, rows)
    torch.testing.assert_close(actual, definitions[rows], atol=0, rtol=0)
    assert actual.dtype == definitions.dtype and actual.device == definitions.device

    # Dyadic weights have exact float32 sums at this bounded size. A float64
    # reference checks counts and gradients without accepting rounding drift
    # as a tolerance for a missing producer, padding suppression or averaging.
    weights = (
        (torch.arange(actual.numel(), device=device) % 7 - 2).float() / 8
    ).reshape(actual.shape)
    actual.backward(weights)
    expected = torch.zeros(count, 4, dtype=torch.float64)
    expected.index_add_(
        0, rows.cpu().reshape(-1), weights.cpu().double().reshape(-1, 4)
    )
    assert definitions.grad is not None and producer.grad is not None
    assert bias.grad is not None
    torch.testing.assert_close(
        definitions.grad.cpu().double(), expected, atol=0, rtol=0
    )
    torch.testing.assert_close(
        producer.grad.cpu().double(), expected * 2, atol=0, rtol=0
    )
    torch.testing.assert_close(
        bias.grad.cpu().double(), expected.sum(0), atol=0, rtol=0
    )
    if ids.numel():
        assert expected[0].abs().sum() > 0


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_definition_lookup_preserves_autograd_contract(
    lookup_encoder: SemanticCardEncoder, requires_grad: bool, grad_enabled: bool
) -> None:
    encoder = lookup_encoder
    definitions = torch.ones(
        encoder.catalog_shape.definitions + 1,
        4,
        device=encoder.definition_rows.device,
        requires_grad=requires_grad,
    )
    ids = (encoder.definition_rows >= 0).nonzero().flatten()[:3]
    with torch.set_grad_enabled(grad_enabled):
        actual = encoder.lookup(definitions, encoder.definition_rows[ids])
        expected = definitions[encoder.definition_rows[ids]]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.requires_grad == (requires_grad and grad_enabled)


def tensors(space, observation):
    return {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in space.encode(observation).items()
    }


def test_programs_and_known_definitions_reach_actual_policy(learn_root):  # noqa: F811
    engine = learn_root.clone_env()
    contract = SemanticDecisionContract.from_env(engine)
    actor = contract.frame.actor
    before = engine.observation_for_player(actor)
    torch.manual_seed(75)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, world="ur-lessons-vs-gw-allies"
        ),
    )
    obs = tensors(space, before)
    logits, _ = agent(device_input(agent, obs))
    take = [
        index
        for index, action in enumerate(before.action_space.actions)
        if int(action.action_type) == 14
    ]
    assert len(take) == 3
    # The outside rows otherwise have identical bounded features. Their
    # distinct executable programs must affect actual retrieval scores.
    assert len(set(logits[0, take].detach().tolist())) == 3
    # Sum the take logits: at this width a single logit can sit at a dead
    # activation and carry no gradient on its own.
    logits[0, take].sum().backward()
    assert agent.semantic_cards.token_embedding.weight.grad.abs().sum() > 0

    retrieval = contract.frame.find_verb("learn_take_lesson")
    contract.apply(engine, retrieval["id"])
    after = engine.observation_for_player(1 - actor)
    public = tensors(space, after)
    assert public["known_hand"][0, 1, :, 1].sum() == 1
    assert all(int(card.zone) != 1 for card in after.opponent_cards)
    _, value = agent(device_input(agent, public))
    forgotten = {key: value.clone() for key, value in public.items()}
    forgotten["known_hand"].zero_()
    _, forgotten_value = agent(device_input(agent, forgotten))
    assert not torch.equal(value, forgotten_value)
    # The opponent's open decklist (main, unseen and sideboard rows) also
    # reaches the value; a closed decklist would encode as these zero rows.
    assert public["decklists"][0, 1, :, 1].sum() == 41
    closed = {key: value.clone() for key, value in public.items()}
    closed["decklists"][:, 1].zero_()
    _, closed_value = agent(device_input(agent, closed))
    assert not torch.equal(value, closed_value)
    for seed in (17, 91):
        hidden = engine.clone_env()
        hidden.determinize(seed, perspective=1 - actor)
        encoded = space.encode(hidden.observation_for_player(1 - actor))
        for key in ("semantic_cards", "known_hand", "decklists"):
            np.testing.assert_array_equal(encoded[key], public[key][0].numpy())


def test_semantic_policy_rejects_unadmitted_definition(learn_root):  # noqa: F811
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, world="ur-lessons-vs-gw-allies"
        ),
    )
    actor = SemanticDecisionContract.from_env(learn_root).frame.actor
    obs = tensors(space, learn_root.observation_for_player(actor))
    obs["semantic_cards"][0, 0, 0] = 1000000
    with pytest.raises(ValueError, match="unadmitted"):
        agent(device_input(agent, obs))
