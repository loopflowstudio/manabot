"""Choice audits resolve native aliases without labelling choices by storage id."""

import json

import numpy as np

from manabot.env.observation import ObservationSpace
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
    for identity in (spell.stack_object_id, spell.object_id):
        action.focus = [identity]
        space.actions = [action]
        raw.action_space = space
        signatures.append(_signatures(raw))
        obs = choice_inputs.encode(raw, ObservationSpace())
        inputs.append(action_inputs(obs, 1, "clean"))
        focus = inputs[-1][0].focus[0]
        assert focus is not None and focus.kind == "stack"
        np.testing.assert_array_equal(focus.features, obs.project("stack")[0, 0])
    assert signatures[0] == signatures[1]
    assert inputs[0] == inputs[1]


def test_permutations_rebind_focus_to_occupied_slots_and_restore_the_viewer() -> None:
    raw = priority_root(
        ("Mountain",), ("Island", "Mountain", "Gray Ogre")
    ).scenario_refresh()
    space = ObservationSpace()
    original = choice_inputs.encode(raw, space)
    count = len(raw.action_space.actions)
    assert count > 1
    with choice_inputs.reorder_objects(raw):
        moved = choice_inputs.encode(raw, space)
        expected = original.project("agent_permanents")[0, :3][::-1]
        np.testing.assert_array_equal(
            moved.project("agent_permanents")[0, :3], expected
        )
        with choice_inputs.reorder_offers(raw, list(reversed(range(count)))):
            reversed_rows = choice_inputs.encode(raw, space)
            for key in ("actions", "name_candidates", "action_focus"):
                np.testing.assert_array_equal(
                    reversed_rows.spans.tables[key].records,
                    moved.spans.tables[key].records[::-1],
                )
    assert (
        choice_inputs.encode(raw, space).batch.to_bytes() == original.batch.to_bytes()
    )
