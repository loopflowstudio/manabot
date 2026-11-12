"""Pooled mana with a spend effect reaches both encoders as the same program rows."""

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
    payload[seat]["spend_effect_mana"] = [
        {
            "effect": "a +1/+1 counter on a non-Human creature spell it casts",
            "mana": [0, 0, 0, 0, 1, 0],
            "program": _PROGRAM,
        }
    ]
    return env, managym.Observation.from_json(json.dumps(payload))


def test_spend_effect_mana_encodes_identically_as_a_program_on_its_player() -> None:
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
    encoder = ObservationEncoder(ObservationConfig())
    reference = encoder.encode(obs)
    native = env.encode_observation(obs, encoder.config.max_facts)
    for key, value in reference.items():
        np.testing.assert_array_equal(native[key], value, err_msg=key)


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
    assert [list(pooled.mana) for pooled in obs.agent.spend_effect_mana] == [
        [0, 0, 0, 0, 1, 0]
    ]
    _assert_parity(env, obs)
    assert (
        "Mana that puts a +1/+1 counter on a non-Human creature spell it casts: G 1."
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
        pay = next(
            index
            for index, action in enumerate(obs.action_space.actions)
            if int(action.action_type) == ActionEnum.PAY_COST
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
