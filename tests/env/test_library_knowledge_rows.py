"""Exact library facts stay viewer-scoped through transport and policy input."""

import json

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.semantic.text_observation import capture_text_decision
import managym
from managym.possible_worlds import PossibleWorldSpace
from tests.model.prepared_routes import device_input


def _root(owner: int) -> managym.Env:
    env = managym.Env(seed=19)
    deck = {"Hakoda, Selfless Commander": 4, "Plains": 20, "Gray Ogre": 16}
    env.reset([managym.PlayerConfig(f"seat {seat}", deck) for seat in range(2)])
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
    assert not space.encode(other)["known_library"][1].any()
    definition = own.agent.known_library[0]
    encoded = space.encode(own)["known_library"]
    np.testing.assert_array_equal(encoded[0, 0], [definition + 1, 1])
    assert not encoded[0, 1:].any()
    assert not encoded[1].any()
    assert sum(own.agent.unseen_counts.values()) == own.agent.zone_counts[0] - 1

    # The shared native writer infers its layout from these supplied buffers.
    buffers = space.encoder.allocate(1)
    managym.encode_observations_into([env], buffers, 1)
    acting = space.encode(env.observation_for_player(env.current_agent_index()))
    for key in acting:
        np.testing.assert_array_equal(buffers[key][0], acting[key], err_msg=key)


def test_library_facts_reach_text_and_the_actual_torch_policy() -> None:
    env = _root(0)
    assert env.current_agent_index() == 0
    assert "Known library position 1 from top:" in capture_text_decision(env).text
    space = ObservationSpace()
    rows = space.encode(env.observation_for_player(0))
    inputs = {key: value[None] for key, value in rows.items()}
    torch.manual_seed(71)
    policy = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    present = device_input(policy, inputs)
    assert present.library_weights[0, 0, 0] == 1
    masked = {key: value.copy() for key, value in inputs.items()}
    masked["known_library"].fill(0)
    absent = device_input(policy, masked)
    logits, values = policy(present)
    masked_logits, masked_values = policy(absent)
    assert not (
        torch.equal(logits, masked_logits) and torch.equal(values, masked_values)
    )
    (logits[logits > -1e7].sum() + values.sum()).backward()
    gradient = policy.semantic_cards.token_embedding.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize("version", [9, 10])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_inputs_omit_library_positions(version: int) -> None:
    env = _root(0)
    source = json.loads(env.observation_for_player(0).toJSON())
    observation = managym.Observation.from_json(json.dumps(source))
    space = ObservationSpace(ObservationConfig(policy_input_version=version))
    expected = space.encode(observation)
    source["agent"].pop("known_library", None)
    source["opponent"].pop("known_library", None)
    observation = managym.Observation.from_json(json.dumps(source))
    assert not observation.agent.known_library
    without = space.encode(observation)
    for key in without:
        np.testing.assert_array_equal(expected[key], without[key], err_msg=key)


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


@pytest.mark.parametrize(
    "invalid", ["missing", "fractional", "absent", "nan", "infinite", "shape"]
)
def test_current_policy_refuses_missing_or_malformed_library_facts(
    invalid: str,
) -> None:
    env = _root(0)
    space = ObservationSpace()
    rows = space.encode(env.observation_for_player(0))
    inputs = {key: value[None] for key, value in rows.items()}
    policy = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    if invalid == "missing":
        del inputs["known_library"]
    elif invalid == "shape":
        inputs["known_library"] = inputs["known_library"][:, :, :-1]
    else:
        positions = {
            "fractional": 0.5,
            "absent": 0.0,
            "nan": float("nan"),
            "infinite": float("inf"),
        }
        inputs["known_library"][0, 0, 0, 1] = positions[invalid]
    with pytest.raises(ValueError, match="known_library"):
        policy.prepare(inputs)


def test_torch_distinguishes_known_hand_from_known_library_top() -> None:
    env = _root(0)
    rows = ObservationSpace().encode(env.observation_for_player(0))
    definition = rows["known_library"][0, 0, 0]
    rows["known_hand"].fill(0)
    rows["known_library"].fill(0)
    hand = {key: value[None].copy() for key, value in rows.items()}
    library = {key: value[None].copy() for key, value in rows.items()}
    hand["known_hand"][0, 0, 0] = [definition, 1]
    library["known_library"][0, 0, 0] = [definition, 1]
    torch.manual_seed(71)
    policy = ActorCritic(
        ObservationSpace(),
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            world=managym.WORLD_SETUP,
        ),
    )
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
