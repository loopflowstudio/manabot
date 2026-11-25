"""Exact library facts stay viewer-scoped through transport and policy input."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import ModelConfig
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import keep_opening_hands
import managym
from managym.possible_worlds import PossibleWorldSpace
from tests.fixtures.observation_rows import observation_arrays, vector_arrays
from tests.model.prepared_routes import device_input


def _root(owner: int) -> managym.Env:
    env = managym.Env(seed=19)
    deck = {"Hakoda, Selfless Commander": 4, "Plains": 20, "Gray Ogre": 16}
    raw, _ = env.reset(
        [managym.PlayerConfig(f"seat {seat}", deck) for seat in range(2)]
    )
    keep_opening_hands(env, raw)
    env.scenario_force_battlefield(owner, "Hakoda, Selfless Commander", True)
    env.scenario_refresh()
    return env


@pytest.mark.parametrize("owner", [0, 1])
def test_library_facts_encode_for_their_owner_and_remain_private(owner: int) -> None:
    env = _root(owner)
    space = ObservationSpace()
    own = env.observation_for_player(owner)
    other = env.observation_for_player(1 - owner)
    assert set(own.agent.known_library) == {0}
    assert not other.opponent.known_library
    assert not observation_arrays(space, other)["known_library"][1].any()
    definition = own.agent.known_library[0]
    encoded = observation_arrays(space, own)["known_library"]
    np.testing.assert_array_equal(encoded[0, 0], [definition + 1, 1])
    assert not encoded[0, 1:].any()
    assert not encoded[1].any()
    assert sum(own.agent.unseen_counts.values()) == own.agent.zone_counts[0] - 1

    # Batch encoding of a live env reads its acting viewer's projection.
    batch = managym.encode_observation_batch([env], space.layout)
    rows = vector_arrays(space, [(batch, 0)])
    acting = observation_arrays(
        space, env.observation_for_player(env.current_agent_index())
    )
    for key in acting:
        np.testing.assert_array_equal(rows[key][0], acting[key], err_msg=key)


def test_library_facts_reach_text_and_the_actual_torch_policy() -> None:
    env = _root(0)
    assert env.current_agent_index() == 0
    assert "Known library position 1 from top:" in capture_text_decision(env).text
    space = ObservationSpace()
    torch.manual_seed(71)
    policy = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    prepared = policy.prepare(
        encode_policy_batch(space, [env.observation_for_player(0)])
    )
    present = device_input(policy, prepared)
    assert present.library_weights[0, 0, 0] == 1
    absent = device_input(
        policy,
        replace(
            prepared,
            library_rows=np.zeros_like(prepared.library_rows),
            library_weights=np.zeros_like(prepared.library_weights),
        ),
    )
    logits, values = policy(present)
    masked_logits, masked_values = policy(absent)
    assert not (
        torch.equal(logits, masked_logits) and torch.equal(values, masked_values)
    )
    (logits[logits > -1e7].sum() + values.sum()).backward()
    gradient = policy.semantic_cards.token_embedding.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize("owner", [0, 1])
def test_visible_library_top_keeps_its_exact_object_in_sampled_worlds(
    owner: int,
) -> None:
    env = _root(owner)
    before = env.observation_for_player(owner)
    visible = [
        (card.id, card.registry_key)
        for card in before.agent_cards
        if card.zone == managym.ZoneEnum.LIBRARY
    ]
    assert len(visible) == 1
    for seed in range(8):
        branch = env.clone_env()
        branch.determinize(seed, owner)
        after = branch.observation_for_player(owner)
        assert [
            (card.id, card.registry_key)
            for card in after.agent_cards
            if card.zone == managym.ZoneEnum.LIBRARY
        ] == visible
    space = PossibleWorldSpace.from_engine(env, viewer=owner)
    branch = space.materialize(
        0, seed=3, refresh_opponent_commitment=owner != env.current_agent_index()
    )
    after = branch.observation_for_player(owner)
    assert [
        (card.id, card.registry_key)
        for card in after.agent_cards
        if card.zone == managym.ZoneEnum.LIBRARY
    ] == visible


def test_torch_distinguishes_known_hand_from_known_library_top() -> None:
    env = _root(0)
    torch.manual_seed(71)
    policy = ActorCritic(
        ObservationSpace(),
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            world=managym.WORLD_SETUP,
        ),
    )
    base = policy.prepare(
        encode_policy_batch(policy.observation_space, [env.observation_for_player(0)])
    )
    row = base.library_rows[0, 0, 0]
    assert row > 0
    empty = replace(
        base,
        known_rows=np.zeros_like(base.known_rows),
        known_counts=np.zeros_like(base.known_counts),
        library_rows=np.zeros_like(base.library_rows),
        library_weights=np.zeros_like(base.library_weights),
    )
    hand_rows, hand_counts = empty.known_rows.copy(), empty.known_counts.copy()
    hand_rows[0, 0, 0], hand_counts[0, 0, 0] = row, 1
    library_rows = empty.library_rows.copy()
    library_weights = empty.library_weights.copy()
    library_rows[0, 0, 0], library_weights[0, 0, 0] = row, 1
    hand = replace(empty, known_rows=hand_rows, known_counts=hand_counts)
    library = replace(empty, library_rows=library_rows, library_weights=library_weights)
    encoder = policy.semantic_cards
    assert encoder is not None
    hand_input = device_input(policy, hand)
    library_input = device_input(policy, library)
    definitions = policy.name_candidates.public_definitions(
        encoder.definitions(hand_input.programs)
    )
    _, hand_knowledge = encoder(hand_input, definitions)
    _, library_knowledge = encoder(library_input, definitions)
    assert not torch.equal(hand_knowledge, library_knowledge)
