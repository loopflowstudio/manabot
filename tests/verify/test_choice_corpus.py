"""Choice audits resolve native aliases without labelling choices by storage id."""

import json

import numpy as np

from manabot.env.observation import STACK_SOURCE_INDEX, ObservationSpace
from manabot.managym.enums import ActionEnum
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify import choice_inputs
from manabot.verify.choice_corpus import _signatures, action_inputs
from manabot.verify.text_observation import priority_root
import managym


def choose(env: managym.Env, *choices: str) -> managym.Observation:
    for choice in choices:
        decision = capture_text_decision(env)
        action = next(
            action for action in decision.actions if action.description == choice
        )
        submit_response(env, decision, action.reference)
    actor = env.current_agent_index()
    assert actor is not None
    return env.observation_for_player(actor)


def test_payment_source_names_the_battlefield_object() -> None:
    env = priority_root((), ("Island", "Island", "Island", "Flexible Waterbender"))
    raw = choose(env, "Activate ability of P4 (Flexible Waterbender); ability 1")
    signatures = _signatures(raw)
    index, action = next(
        (index, action)
        for index, action in enumerate(raw.action_space.actions)
        if int(action.action_type) == ActionEnum.PAY_COST
    )
    permanent = next(
        item for item in raw.agent_permanents if item.object_id == action.focus[0]
    )
    assert permanent.id != permanent.object_id
    descriptor = json.loads(signatures[index])[-1][0]
    assert descriptor[:3] == ["permanent", "self", permanent.registry_key]
    inputs = action_inputs(
        choice_inputs.encode(raw, ObservationSpace()), len(signatures), "clean"
    )
    focus = inputs[index].focus[0]
    assert focus is not None and focus.kind == "permanents"


def test_stack_aliases_share_a_signature_and_reach_collision_inputs() -> None:
    env = priority_root(
        ("Waterbending Lesson",),
        ("Island", "Island", "Island", "Island", "Gray Ogre", "Hill Giant"),
    )
    # The Lesson remains on the stack while its resolution payment is pending.
    raw = choose(env, "Cast C1 (Waterbending Lesson)", "Pass priority", "Pass priority")
    spell = raw.stack_objects[0]
    assert spell.object_id is not None
    # Project the same diagnostic choice onto each native alias. Both aliases
    # must resolve to the same stack row, including in the collision checker.
    space = raw.action_space
    action = space.actions[0]
    signatures = []
    inputs = []
    observation_space = ObservationSpace()
    for identity in (spell.stack_object_id, spell.object_id):
        action.focus = [identity]
        space.actions = [action]
        raw.action_space = space
        signatures.append(_signatures(raw))
        obs = choice_inputs.encode(raw, observation_space)
        inputs.append(action_inputs(obs, 1, "clean"))
        focus = inputs[-1][0].focus[0]
        assert focus is not None and focus.kind == "stack"
        np.testing.assert_array_equal(focus.features, obs["stack"][0, 0])
    assert signatures[0] == signatures[1]
    assert inputs[0] == inputs[1]


def test_permutations_preserve_stack_sources_card_turn_state_and_names() -> None:
    raw = priority_root(("Waterbending Lesson",), ("Island",)).scenario_refresh()
    obs = choice_inputs.encode(raw, ObservationSpace())
    first_permanent = 2 + 2 * obs["agent_cards"].shape[1]
    permanent_count = obs["agent_permanents"].shape[1]
    first_stack = first_permanent + 2 * permanent_count
    obs["stack"][0, 0, STACK_SOURCE_INDEX] = first_permanent + 1
    obs["action_focus"][0, 0, 0] = first_stack
    obs["card_turn_state"][0, 1, 0] = 1
    obs["name_candidates"][0, 0, 0] = 3
    moved = choice_inputs.reorder_objects(obs)
    assert moved["stack"][0, 0, STACK_SOURCE_INDEX] == first_permanent + permanent_count
    assert moved["action_focus"][0, 0, 0] == first_stack
    assert moved["card_turn_state"][0, 1, -1] == 1
    offers = choice_inputs.reorder_offers(moved, [1, 0])
    assert offers["name_candidates"][0, 1, 0] == 3
    for key, value in choice_inputs.reorder_objects(moved).items():
        np.testing.assert_array_equal(value, obs[key], strict=True)
