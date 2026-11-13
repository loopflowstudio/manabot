"""Pooled mana classes retain qualifiers and shared learning-token programs."""

import json

import numpy as np

from manabot.env.observation import (
    FACT_DIM,
    FACT_EFFECT_PROGRAM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_VALUE_INDEX,
    POOLED_MANA,
    ObservationEncoder,
)
from manabot.infra.config import ObservationConfig
from manabot.managym.enums import ActionEnum
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym
from managym.experience import LegacyPlayerView
from managym.experience_projection import serialize_observation

# Integer tokens: a program of literals needs no vocabulary.
_PROGRAM = [[4, 0], [4, 1], [4, 2]]


def _observation(seat: str) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=163, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-temur-v1", key)
            for key in ("temur_left", "temur_right")
        ]
    )
    payload = json.loads(env.observation_for_player(0).toJSON())
    payload[seat]["pooled_mana_classes"] = [
        {
            "combat": False,
            "effect": "a +1/+1 counter on a non-Human creature spell it casts",
            "mana": [0, 0, 0, 0, 1, 0],
            "program": _PROGRAM,
        }
    ]
    return env, managym.Observation.from_json(json.dumps(payload))


def test_pooled_mana_classes_encode_identically_as_a_program_on_their_player() -> None:
    for seat, object_row in (("agent", 1), ("opponent", 2)):
        env, obs = _observation(seat)
        assert managym.unrepresentable_state(obs) is None
        expected = ObservationEncoder(ObservationConfig()).encode(obs)
        actual = env.encode_observation(obs)
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value, err_msg=key)

        token = managym.catalog_token_value(POOLED_MANA)
        facts = expected["facts"].reshape(-1, FACT_DIM)
        rows = facts[
            (facts[:, FACT_KIND_INDEX] == FACT_EFFECT_PROGRAM)
            & (facts[:, FACT_ID_INDEX] == token >> 16)
            & (facts[:, FACT_ID_INDEX + 1] == token & 0xFFFF)
        ]
        assert len(rows) == len(_PROGRAM)
        assert set(rows[:, FACT_OBJECT_INDEX]) == {object_row}
        np.testing.assert_array_equal(
            rows[:, FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2],
            [[4.0, ordinal] for ordinal in range(len(_PROGRAM))],
        )


def _assert_parity(env: managym.Env, obs: managym.Observation) -> None:
    for version in (9, 10):
        encoder = ObservationEncoder(ObservationConfig(policy_input_version=version))
        reference = encoder.encode(obs)
        native = {key: np.zeros_like(value) for key, value in reference.items()}
        env.encode_observation_into(obs, native)
        for key, value in reference.items():
            np.testing.assert_array_equal(native[key], value, err_msg=key)
        if obs.agent.player_index == env.current_agent_index():
            buffers = encoder.allocate(1)
            managym.encode_observations_into([env], buffers, 1)
            for key, value in reference.items():
                np.testing.assert_array_equal(buffers[key][0], value, err_msg=key)


def test_animal_attendant_mana_and_cast_receipt_native_reference_and_text_parity() -> (
    None
):
    env = priority_root(
        ("Uncle's Musings",),
        ("Animal Attendant", "Forest", "Forest", "Island", "Plains"),
    )
    owner = env.current_agent_index()
    assert owner is not None
    decision = capture_text_decision(env)
    activation = next(
        action
        for action in decision.actions
        if "Activate" in action.description and "Animal Attendant" in action.description
    )
    submit_response(env, decision, activation.reference)
    modes = capture_text_decision(env)
    green = next(action for action in modes.actions if "{G}" in action.description)
    submit_response(env, modes, green.reference)

    # The Attendant's green counts as ordinary mana and names its rider.
    obs = env.observation_for_player(owner)
    assert list(obs.agent.mana) == [0, 0, 0, 0, 1, 0]
    assert [list(pooled.mana) for pooled in obs.agent.pooled_mana_classes] == [
        [0, 0, 0, 0, 1, 0]
    ]
    _assert_parity(env, obs)
    assert (
        "Of the mana above: G 1; puts a +1/+1 counter on a non-Human creature spell it casts; expires at step end."
        in capture_text_decision(env).text
    )

    # Converge reads the mana spent: the receipt is on the stack frame.
    cast = next(
        index
        for index, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_CAST_SPELL
    )
    env.step(cast)
    for _ in range(8):
        obs = env.observation_for_player(owner)
        if obs.stack_objects:
            break
        _assert_parity(env, obs)
        pay = next(
            index
            for index, action in enumerate(obs.action_space.actions)
            if int(action.action_type) in (ActionEnum.PAY_COST, ActionEnum.CHOOSE_MODE)
        )
        env.step(pay)
    else:
        raise AssertionError("Uncle's Musings did not reach the stack")
    actor = env.current_agent_index()
    assert actor is not None
    for viewer in (owner, 1 - owner):
        obs = env.observation_for_player(viewer)
        assert obs.continuation[0].kind == 1
        assert obs.continuation[0].effect_execution
        _assert_parity(env, obs)
    assert "mana spent" in capture_text_decision(env).text


def test_spending_choices_share_native_vector_text_and_academy_observations() -> None:
    env = priority_root(("Uncle's Musings",), ("Animal Attendant",) * 5)
    for symbol in ("W", "U", "G", "G", "G"):
        decision = capture_text_decision(env)
        activate = next(
            a
            for a in decision.actions
            if "Activate" in a.description and "Animal Attendant" in a.description
        )
        submit_response(env, decision, activate.reference)
        decision = capture_text_decision(env)
        output = next(a for a in decision.actions if f"{{{symbol}}}" in a.description)
        submit_response(env, decision, output.reference)
    decision = capture_text_decision(env)
    cast = next(
        a
        for a in decision.actions
        if "Cast" in a.description and "Uncle's Musings" in a.description
    )
    submit_response(env, decision, cast.reference)
    owner = env.current_agent_index()
    assert owner is not None
    obs = env.observation_for_player(owner)
    _assert_parity(env, obs)
    assert obs.continuation[-1].decision == 31
    text = capture_text_decision(env)
    assert len(text.actions) == 3
    assert all(
        "Spend" in a.description
        and "puts" in a.description
        and "expires" in a.description
        for a in text.actions
    )
    view = serialize_observation(obs, include_details=True)
    LegacyPlayerView.model_validate(view["agent"])
    pools = view["agent"]["pooled_mana_classes"]
    assert pools[0]["effect"]
    assert "restriction" not in pools[0]
    assert pools[0]["combat"] is False
    # Every selectable unit, not just the aggregate colors, reaches the
    # policy's closed runtime vocabulary through the execution program.
    vocabulary = {
        (kind, token)
        for _, _, _, kind, token in managym.record_vocabulary()
        if kind is not None
    }
    catalog = managym.catalog_tokens()
    vocabulary.update(zip(catalog["token_kind"], catalog["token_value"], strict=True))
    for frame in obs.continuation:
        for kind, value in frame.effect_execution:
            assert kind in (4, 5, 8) or (kind, value) in vocabulary

    for symbol in ("W", "U"):
        decision = capture_text_decision(env)
        choice = next(a for a in decision.actions if f"{{{symbol}}}" in a.description)
        submit_response(env, decision, choice.reference)
        actor = env.current_agent_index()
        assert actor is not None
        _assert_parity(env, env.observation_for_player(actor))
    assert env.observation_for_player(owner).stack_objects


def test_combat_mana_without_qualifiers_round_trips_and_projects_absence() -> None:
    env, obs = _observation("agent")
    payload = json.loads(obs.toJSON())
    pooled = payload["agent"]["pooled_mana_classes"][0]
    del pooled["effect"]
    pooled["combat"] = True
    obs = managym.Observation.from_json(json.dumps(payload))
    native_class = obs.agent.pooled_mana_classes[0]
    assert isinstance(native_class, managym.PooledManaClass)
    assert native_class.effect is None
    assert native_class.restriction is None
    serialized = json.loads(obs.toJSON())["agent"]["pooled_mana_classes"][0]
    assert "effect" not in serialized and "restriction" not in serialized
    reference = ObservationEncoder(ObservationConfig()).encode(obs)
    native = env.encode_observation(obs)
    for key, value in reference.items():
        np.testing.assert_array_equal(native[key], value, err_msg=key)
    view = serialize_observation(obs, include_details=True)["agent"]
    expected = {"mana": [0, 0, 0, 0, 1, 0], "combat": True}
    assert view["pooled_mana_classes"] == [expected]
    parsed = LegacyPlayerView.model_validate(view)
    assert parsed.model_dump(mode="json")["pooled_mana_classes"] == [expected]
