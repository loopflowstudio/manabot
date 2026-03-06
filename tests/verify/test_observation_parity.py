"""Parity checks for Rust observation encoding against Python observation encoding."""

from __future__ import annotations

import numpy as np
import pytest

# Local imports
from manabot.env.observation import ObservationEncoder, ObservationSpaceHypers
import managym


def make_player_configs() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig("Red Mage", {"Mountain": 10, "Grey Ogre": 20}),
        managym.PlayerConfig("Green Mage", {"Forest": 10, "Llanowar Elves": 20}),
    ]


def random_valid_action(obs: managym.Observation, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(obs.action_space.actions)))


def allocate_vector_buffers(
    num_envs: int,
    hypers: ObservationSpaceHypers,
) -> dict[str, np.ndarray]:
    encoder = ObservationEncoder(hypers)
    return {
        "agent_player": np.zeros((num_envs, 1, encoder.player_dim), dtype=np.float32),
        "opponent_player": np.zeros(
            (num_envs, 1, encoder.player_dim),
            dtype=np.float32,
        ),
        "agent_cards": np.zeros(
            (num_envs, hypers.max_cards_per_player, encoder.card_dim),
            dtype=np.float32,
        ),
        "opponent_cards": np.zeros(
            (num_envs, hypers.max_cards_per_player, encoder.card_dim),
            dtype=np.float32,
        ),
        "agent_permanents": np.zeros(
            (num_envs, hypers.max_permanents_per_player, encoder.permanent_dim),
            dtype=np.float32,
        ),
        "opponent_permanents": np.zeros(
            (num_envs, hypers.max_permanents_per_player, encoder.permanent_dim),
            dtype=np.float32,
        ),
        "actions": np.zeros(
            (num_envs, hypers.max_actions, encoder.action_dim),
            dtype=np.float32,
        ),
        "action_focus": np.zeros(
            (num_envs, hypers.max_actions, hypers.max_focus_objects),
            dtype=np.int32,
        ),
        "agent_player_valid": np.zeros((num_envs, 1), dtype=np.float32),
        "opponent_player_valid": np.zeros((num_envs, 1), dtype=np.float32),
        "agent_cards_valid": np.zeros(
            (num_envs, hypers.max_cards_per_player),
            dtype=np.float32,
        ),
        "opponent_cards_valid": np.zeros(
            (num_envs, hypers.max_cards_per_player),
            dtype=np.float32,
        ),
        "agent_permanents_valid": np.zeros(
            (num_envs, hypers.max_permanents_per_player),
            dtype=np.float32,
        ),
        "opponent_permanents_valid": np.zeros(
            (num_envs, hypers.max_permanents_per_player),
            dtype=np.float32,
        ),
        "actions_valid": np.zeros((num_envs, hypers.max_actions), dtype=np.float32),
        "rewards": np.zeros((num_envs,), dtype=np.float64),
        "terminated": np.zeros((num_envs,), dtype=bool),
        "truncated": np.zeros((num_envs,), dtype=bool),
    }


def assert_buffer_observation_equal(
    buffers: dict[str, np.ndarray],
    env_index: int,
    encoded_obs: dict[str, np.ndarray],
) -> None:
    for key, expected in encoded_obs.items():
        actual = buffers[key][env_index]
        if key == "action_focus":
            np.testing.assert_array_equal(
                actual,
                expected,
                err_msg=f"env={env_index} key={key}",
            )
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-6,
                err_msg=f"env={env_index} key={key}",
            )


def expected_focus_indices(
    obs: managym.Observation,
    hypers: ObservationSpaceHypers,
) -> np.ndarray:
    object_to_index: dict[int, int] = {}
    current_object_index = 0

    object_to_index[obs.agent.id] = current_object_index
    current_object_index += 1
    object_to_index[obs.opponent.id] = current_object_index
    current_object_index += 1

    for cards in (obs.agent_cards, obs.opponent_cards):
        ordered = cards[: hypers.max_cards_per_player]
        for card in ordered:
            object_to_index[card.id] = current_object_index
            current_object_index += 1
        current_object_index += hypers.max_cards_per_player - len(ordered)

    for permanents in (obs.agent_permanents, obs.opponent_permanents):
        ordered = permanents[: hypers.max_permanents_per_player]
        for permanent in ordered:
            object_to_index[permanent.id] = current_object_index
            current_object_index += 1
        current_object_index += hypers.max_permanents_per_player - len(ordered)

    out = np.full(
        (hypers.max_actions, hypers.max_focus_objects),
        -1,
        dtype=np.int32,
    )
    for action_index, action in enumerate(
        obs.action_space.actions[: hypers.max_actions]
    ):
        for focus_index, object_id in enumerate(
            action.focus[: hypers.max_focus_objects]
        ):
            out[action_index, focus_index] = object_to_index.get(object_id, -1)

    return out


def test_observation_parity():
    """Run 200 games, encode every observation both ways, and assert exact parity."""
    rng = np.random.default_rng(42)
    env = managym.Env(seed=42, skip_trivial=True)
    hypers = ObservationSpaceHypers()
    encoder = ObservationEncoder(hypers)

    for game_i in range(200):
        obs, _ = env.reset(make_player_configs())

        while True:
            encoded_python = encoder.encode(obs)
            encoded_rust = env.encode_observation(obs)

            for key in encoded_python:
                np.testing.assert_allclose(
                    encoded_rust[key],
                    encoded_python[key],
                    atol=1e-6,
                    err_msg=f"Game {game_i}, key {key}",
                )

            if obs.game_over:
                break

            if len(obs.action_space.actions) == 0:
                break

            action = random_valid_action(obs, rng)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                # We still compare one final observation at the top of the loop.
                continue


def test_object_index_mapping_and_padded_focus_resolution():
    """Focus indices use deterministic object ordering and pad unused slots with -1."""
    env = managym.Env(seed=7, skip_trivial=True)
    hypers = ObservationSpaceHypers()

    obs, _ = env.reset(make_player_configs())
    encoded = env.encode_observation(obs)

    expected_focus = expected_focus_indices(obs, hypers)
    np.testing.assert_array_equal(encoded["action_focus"], expected_focus)


def test_encode_observation_into_fills_preallocated_buffers_and_validates_contract():
    env = managym.Env(seed=5, skip_trivial=True)
    hypers = ObservationSpaceHypers()
    obs, _ = env.reset(make_player_configs())

    expected = env.encode_observation(obs)

    out = {
        "agent_player": np.full((1, 26), -7.0, dtype=np.float32),
        "opponent_player": np.full((1, 26), -7.0, dtype=np.float32),
        "agent_cards": np.full(
            (hypers.max_cards_per_player, 18),
            -7.0,
            dtype=np.float32,
        ),
        "opponent_cards": np.full(
            (hypers.max_cards_per_player, 18),
            -7.0,
            dtype=np.float32,
        ),
        "agent_permanents": np.full(
            (hypers.max_permanents_per_player, 5),
            -7.0,
            dtype=np.float32,
        ),
        "opponent_permanents": np.full(
            (hypers.max_permanents_per_player, 5),
            -7.0,
            dtype=np.float32,
        ),
        "actions": np.full((hypers.max_actions, 7), -7.0, dtype=np.float32),
        "action_focus": np.full(
            (hypers.max_actions, hypers.max_focus_objects),
            -7,
            dtype=np.int32,
        ),
        "agent_player_valid": np.full((1,), -7.0, dtype=np.float32),
        "opponent_player_valid": np.full((1,), -7.0, dtype=np.float32),
        "agent_cards_valid": np.full(
            (hypers.max_cards_per_player,),
            -7.0,
            dtype=np.float32,
        ),
        "opponent_cards_valid": np.full(
            (hypers.max_cards_per_player,),
            -7.0,
            dtype=np.float32,
        ),
        "agent_permanents_valid": np.full(
            (hypers.max_permanents_per_player,),
            -7.0,
            dtype=np.float32,
        ),
        "opponent_permanents_valid": np.full(
            (hypers.max_permanents_per_player,),
            -7.0,
            dtype=np.float32,
        ),
        "actions_valid": np.full((hypers.max_actions,), -7.0, dtype=np.float32),
    }

    env.encode_observation_into(obs, out)

    for key, expected_value in expected.items():
        np.testing.assert_allclose(out[key], expected_value, atol=1e-6)

    bad = dict(out)
    bad["actions"] = np.zeros((hypers.max_actions, 7), dtype=np.float64)
    try:
        env.encode_observation_into(obs, bad)
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised, "Expected dtype mismatch to raise ValueError or TypeError"


def test_vector_env_buffer_hot_path_matches_compatibility_step_path():
    rng = np.random.default_rng(1234)
    hypers = ObservationSpaceHypers()
    encoder = ObservationEncoder(hypers)
    num_envs = 4
    player_configs = make_player_configs()

    compat = managym.VectorEnv(
        num_envs=num_envs,
        seed=123,
        skip_trivial=True,
        opponent_policy="passive",
    )
    hot = managym.VectorEnv(
        num_envs=num_envs,
        seed=123,
        skip_trivial=True,
        opponent_policy="passive",
    )
    buffers = allocate_vector_buffers(num_envs, hypers)
    hot.set_buffers(buffers)

    compat_reset = compat.reset_all(player_configs)
    hot.reset_all_into_buffers(player_configs)

    for env_index, (obs, _) in enumerate(compat_reset):
        assert_buffer_observation_equal(buffers, env_index, encoder.encode(obs))
    np.testing.assert_allclose(buffers["rewards"], np.zeros(num_envs, dtype=np.float64))
    np.testing.assert_array_equal(buffers["terminated"], np.zeros(num_envs, dtype=bool))
    np.testing.assert_array_equal(buffers["truncated"], np.zeros(num_envs, dtype=bool))

    last_compat_step = compat_reset
    for _ in range(60):
        actions = [random_valid_action(obs, rng) for obs, _ in last_compat_step]
        compat_step = compat.step(actions)
        hot.step_into_buffers(actions)

        expected_rewards = np.array([step[1] for step in compat_step], dtype=np.float64)
        expected_terminated = np.array([step[2] for step in compat_step], dtype=bool)
        expected_truncated = np.array([step[3] for step in compat_step], dtype=bool)
        np.testing.assert_allclose(buffers["rewards"], expected_rewards, atol=1e-12)
        np.testing.assert_array_equal(buffers["terminated"], expected_terminated)
        np.testing.assert_array_equal(buffers["truncated"], expected_truncated)

        for env_index, (obs, _, _, _, _) in enumerate(compat_step):
            assert_buffer_observation_equal(buffers, env_index, encoder.encode(obs))

        hot_infos = [dict(info) for info in hot.get_last_info()]
        compat_infos = [dict(info) for _, _, _, _, info in compat_step]
        assert hot_infos == compat_infos

        last_compat_step = [(obs, info) for obs, _, _, _, info in compat_step]


def test_vector_env_set_buffers_validates_contract():
    hypers = ObservationSpaceHypers()
    num_envs = 2
    player_configs = make_player_configs()
    env = managym.VectorEnv(
        num_envs=num_envs,
        seed=999,
        skip_trivial=True,
        opponent_policy="passive",
    )
    good = allocate_vector_buffers(num_envs, hypers)

    env.set_buffers(good)
    env.reset_all_into_buffers(player_configs)
    env.step_into_buffers([0, 0])

    bad_dtype = dict(good)
    bad_dtype["actions"] = np.zeros_like(good["actions"], dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        env.set_buffers(bad_dtype)

    bad_shape = dict(good)
    bad_shape["agent_player"] = np.zeros((num_envs, 2, 26), dtype=np.float32)
    with pytest.raises(ValueError):
        env.set_buffers(bad_shape)

    bad_contiguous = dict(good)
    bad_contiguous["agent_cards"] = np.asfortranarray(good["agent_cards"])
    with pytest.raises(ValueError):
        env.set_buffers(bad_contiguous)

    bad_writable = dict(good)
    read_only = good["agent_cards"].copy()
    read_only.setflags(write=False)
    bad_writable["agent_cards"] = read_only
    with pytest.raises(ValueError):
        env.set_buffers(bad_writable)

    unconfigured = managym.VectorEnv(
        num_envs=num_envs,
        seed=1000,
        skip_trivial=True,
        opponent_policy="passive",
    )
    with pytest.raises(RuntimeError):
        unconfigured.reset_all_into_buffers(player_configs)
