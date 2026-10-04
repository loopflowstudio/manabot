"""Ordinary policies consume complete outside programs and public knowledge."""

import numpy as np
import pytest
import torch

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from manabot.semantic.decision_contract import SemanticDecisionContract
from tests.semantic.test_learn_contract import learn_root  # noqa: F401


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
    agent = Agent(
        space,
        AgentHypers(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )
    obs = tensors(space, before)
    logits, _ = agent(obs)
    take = [
        index
        for index, action in enumerate(before.action_space.actions)
        if int(action.action_type) == 14
    ]
    assert len(take) == 3
    # The outside rows otherwise have identical bounded features. Their
    # distinct executable programs must affect actual retrieval scores.
    assert len(set(logits[0, take].detach().tolist())) == 3
    logits[0, take[0]].backward()
    assert agent.semantic_cards.token_embedding.weight.grad.abs().sum() > 0

    retrieval = contract.frame.find_verb("learn_take_lesson")
    contract.apply(engine, retrieval["id"])
    after = engine.observation_for_player(1 - actor)
    public = tensors(space, after)
    assert public["known_hand"][0, 1, :, 1].sum() == 1
    assert all(int(card.zone) != 1 for card in after.opponent_cards)
    _, value = agent(public)
    forgotten = {key: value.clone() for key, value in public.items()}
    forgotten["known_hand"].zero_()
    _, forgotten_value = agent(forgotten)
    assert not torch.equal(value, forgotten_value)
    for seed in (17, 91):
        hidden = engine.clone_env()
        hidden.determinize(seed, perspective=1 - actor)
        encoded = space.encode(hidden.observation_for_player(1 - actor))
        for key in ("semantic_cards", "known_hand"):
            np.testing.assert_array_equal(encoded[key], public[key][0].numpy())


def test_semantic_policy_rejects_unadmitted_definition(learn_root):  # noqa: F811
    space = ObservationSpace()
    agent = Agent(
        space,
        AgentHypers(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )
    actor = SemanticDecisionContract.from_env(learn_root).frame.actor
    obs = tensors(space, learn_root.observation_for_player(actor))
    obs["semantic_cards"][0, 0, 0] = 1000000
    with pytest.raises(ValueError, match="unadmitted"):
        agent(obs)
