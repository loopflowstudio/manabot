"""V8 retains complete mana restrictions and shared tap-cost decisions."""

import json

import numpy as np
import pytest

from manabot.env.observation import ObservationEncoder
from manabot.infra.config import ObservationConfig
from manabot.managym.enums import ActionEnum
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import priority_root
import managym


def _parity(env: managym.Env, obs: managym.Observation) -> dict[str, np.ndarray]:
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=8))
    expected = encoder.encode(obs)
    native = env.encode_observation(obs, 8)
    buffered = {key: np.zeros_like(value) for key, value in expected.items()}
    env.encode_observation_into(obs, buffered)
    for key, value in expected.items():
        np.testing.assert_array_equal(native[key], value, err_msg=key)
        np.testing.assert_array_equal(buffered[key], value, err_msg=key)
    return expected


def test_v8_pending_taps_have_scalar_parity_and_explicit_legacy_rejection() -> None:
    env = priority_root((), ("Supportive Parents", "Gray Ogre", "Gray Ogre"))
    obs = env.scenario_refresh()
    index = next(
        i
        for i, a in enumerate(obs.action_space.actions)
        if int(a.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    obs, *_ = env.step(index)
    encoded = _parity(env, obs)
    assert (encoded["relations"][:, 49] == 2).any()
    assert "tap 2 more permanents" in capture_text_decision(env).text
    for version in range(1, 8):
        with pytest.raises(ValueError, match="requires policy input v8"):
            env.encode_observation(obs, version)
        with pytest.raises(ValueError, match="requires policy input v8"):
            ObservationEncoder(ObservationConfig(policy_input_version=version)).encode(
                obs
            )


def test_v8_open_subtype_purposes_and_class_grouping_do_not_collapse() -> None:
    env = priority_root((), ("Mountain", "Gray Ogre"))
    viewer = env.current_agent_index()
    payload = json.loads(env.semantic_observation_json(viewer))["viewer_state"]
    payload["agent"]["lesson_mana"] = [0, 2, 0, 0, 0, 0]
    encodings: list[np.ndarray] = []
    classes = (
        [{"spells": ["Citizen"], "mana": [0, 2, 0, 0, 0, 0]}],
        [{"abilities": ["Citizen"], "mana": [0, 2, 0, 0, 0, 0]}],
        [{"spells": ["Citizen", "Elf"], "mana": [0, 2, 0, 0, 0, 0]}],
        [
            {"spells": ["Citizen"], "mana": [0, 1, 0, 0, 0, 0]},
            {"spells": ["Elf"], "mana": [0, 1, 0, 0, 0, 0]},
        ],
    )
    for restrictions in classes:
        payload["agent"]["restricted_mana"] = restrictions
        obs = managym.Observation.from_json(json.dumps(payload))
        encodings.append(_parity(env, obs)["facts"])
    for i, left in enumerate(encodings):
        for right in encodings[i + 1 :]:
            assert not np.array_equal(left, right)


def test_v8_chosen_output_preserves_restrictions_before_they_enter_the_pool() -> None:
    env = priority_root((), ("Hermitic Herbalist",))
    obs = env.scenario_refresh()
    index = next(
        i
        for i, a in enumerate(obs.action_space.actions)
        if int(a.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
        and a.ability_index == 1
    )
    obs, *_ = env.step(index)
    encoded = _parity(env, obs)
    assert (encoded["facts"][:, 1] == 13).any()
    assert obs.continuation[-1].mana_outputs[0].restrictions[0].spells == ["Lesson"]
    assert "only for Lesson spells" in capture_text_decision(env).text


def test_v8_vector_buffers_select_the_same_layout() -> None:
    configs = [
        managym.PlayerConfig(name, {"Mountain": 10, "Gray Ogre": 20})
        for name in ("a", "b")
    ]
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=8))
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    native = managym.VectorEnv(num_envs=1, seed=123, skip_trivial=True)
    reference = managym.VectorEnv(num_envs=1, seed=123, skip_trivial=True)
    native.set_buffers(buffers)
    [(obs, _)] = reference.reset_all(configs)
    native.reset_all_into_buffers(configs)
    for key, value in encoder.encode(obs).items():
        np.testing.assert_array_equal(buffers[key][0], value, err_msg=key)


def test_v9_projection_is_byte_identical_and_capacity_remains_explicit() -> None:
    env = priority_root((), ("Hermitic Herbalist",))
    obs = env.scenario_refresh()
    choice = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
        and action.ability_index == 1
    )
    obs, *_ = env.step(choice)
    expected = _parity(env, obs)
    count = int(expected["facts"][:, 11].sum())
    assert count > 1
    for capacity in (count, 512):
        encoder = ObservationEncoder(
            ObservationConfig(policy_input_version=9, max_facts=capacity)
        )
        encoded = encoder.encode(obs)
        native = env.encode_observation(obs, 9, capacity)
        buffered = {key: np.zeros_like(value) for key, value in encoded.items()}
        env.encode_observation_into(obs, buffered)
        for key, value in encoded.items():
            np.testing.assert_array_equal(native[key], value, err_msg=key)
            np.testing.assert_array_equal(buffered[key], value, err_msg=key)
            if capacity == 512:
                np.testing.assert_array_equal(expected[key], value, err_msg=key)
    with pytest.raises(ValueError, match="facts"):
        env.encode_observation(obs, 9, count - 1)
    with pytest.raises(ValueError, match="facts"):
        ObservationEncoder(
            ObservationConfig(policy_input_version=9, max_facts=count - 1)
        ).encode(obs)


def test_v9_vector_buffers_match_python_without_a_new_wire_width() -> None:
    configs = [
        managym.PlayerConfig(name, {"Mountain": 10, "Gray Ogre": 20})
        for name in ("a", "b")
    ]
    encoder = ObservationEncoder(
        ObservationConfig(policy_input_version=9, max_facts=128)
    )
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    native = managym.VectorEnv(num_envs=1, seed=123, skip_trivial=True)
    reference = managym.VectorEnv(num_envs=1, seed=123, skip_trivial=True)
    native.set_buffers(buffers)
    [(obs, _)] = reference.reset_all(configs)
    native.reset_all_into_buffers(configs)
    for key, value in encoder.encode(obs).items():
        np.testing.assert_array_equal(buffers[key][0], value, err_msg=key)
