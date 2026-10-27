"""Parity checks for Rust observation encoding against Python observation encoding."""

from __future__ import annotations

import numpy as np
import pytest

# Local imports
<<<<<<< HEAD
from manabot.env.observation import ObservationEncoder, ObservationSpaceHypers
=======
from manabot.env.observation import ObservationConfig, ObservationEncoder
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
import managym


def make_player_configs() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig("Red Mage", {"Mountain": 10, "Gray Ogre": 20}),
        managym.PlayerConfig("Green Mage", {"Forest": 10, "Llanowar Elves": 20}),
    ]


def random_valid_action(obs: managym.Observation, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(obs.action_space.actions)))


def allocate_vector_buffers(
    num_envs: int,
<<<<<<< HEAD
    hypers: ObservationSpaceHypers,
) -> dict[str, np.ndarray]:
    encoder = ObservationEncoder(hypers)
=======
    config: ObservationConfig,
) -> dict[str, np.ndarray]:
    encoder = ObservationEncoder(config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    buffers = encoder.allocate(num_envs)
    buffers.update(
        rewards=np.zeros((num_envs,), dtype=np.float64),
        terminated=np.zeros((num_envs,), dtype=np.uint8),
        truncated=np.zeros((num_envs,), dtype=np.uint8),
    )
    return buffers


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
<<<<<<< HEAD
    hypers: ObservationSpaceHypers,
=======
    config: ObservationConfig,
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
) -> np.ndarray:
    object_to_index: dict[int, int] = {}
    current_object_index = 0

    object_to_index[obs.agent.id] = current_object_index
    current_object_index += 1
    object_to_index[obs.opponent.id] = current_object_index
    current_object_index += 1

    for cards in (obs.agent_cards, obs.opponent_cards):
<<<<<<< HEAD
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
=======
        ordered = cards[: config.max_cards_per_player]
        for card in ordered:
            object_to_index[card.id] = current_object_index
            current_object_index += 1
        current_object_index += config.max_cards_per_player - len(ordered)

    for permanents in (obs.agent_permanents, obs.opponent_permanents):
        ordered = permanents[: config.max_permanents_per_player]
        for permanent in ordered:
            object_to_index[permanent.id] = current_object_index
            current_object_index += 1
        current_object_index += config.max_permanents_per_player - len(ordered)

    out = np.full(
        (config.max_actions, config.max_focus_objects),
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        -1,
        dtype=np.int32,
    )
    for action_index, action in enumerate(
<<<<<<< HEAD
        obs.action_space.actions[: hypers.max_actions]
    ):
        for focus_index, object_id in enumerate(
            action.focus[: hypers.max_focus_objects]
=======
        obs.action_space.actions[: config.max_actions]
    ):
        for focus_index, object_id in enumerate(
            action.focus[: config.max_focus_objects]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        ):
            out[action_index, focus_index] = object_to_index.get(object_id, -1)

    return out


def test_observation_parity():
    """Run 200 games, encode every observation both ways, and assert exact parity."""
    rng = np.random.default_rng(42)
    env = managym.Env(seed=42, skip_trivial=True)
<<<<<<< HEAD
    hypers = ObservationSpaceHypers()
    encoder = ObservationEncoder(hypers)
=======
    config = ObservationConfig()
    encoder = ObservationEncoder(config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

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
<<<<<<< HEAD
    hypers = ObservationSpaceHypers()
=======
    config = ObservationConfig()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

    obs, _ = env.reset(make_player_configs())
    encoded = env.encode_observation(obs)

<<<<<<< HEAD
    expected_focus = expected_focus_indices(obs, hypers)
=======
    expected_focus = expected_focus_indices(obs, config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    np.testing.assert_array_equal(encoded["action_focus"], expected_focus)


def test_encode_observation_into_fills_preallocated_buffers_and_validates_contract() -> (
    None
):
    env = managym.Env(seed=5, skip_trivial=True)
<<<<<<< HEAD
    hypers = ObservationSpaceHypers()
    encoder = ObservationEncoder(hypers)
=======
    config = ObservationConfig()
    encoder = ObservationEncoder(config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    obs, _ = env.reset(make_player_configs())

    expected = env.encode_observation(obs)

    out = {
        key: np.full_like(value[0], -7) for key, value in encoder.allocate(1).items()
    }

    env.encode_observation_into(obs, out)

    for key, expected_value in expected.items():
        np.testing.assert_allclose(out[key], expected_value, atol=1e-6)

    bad = dict(out)
    bad["actions"] = np.zeros(
<<<<<<< HEAD
        (hypers.max_actions, encoder.action_dim), dtype=np.float64
=======
        (config.max_actions, encoder.action_dim), dtype=np.float64
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    )
    try:
        env.encode_observation_into(obs, bad)
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised, "Expected dtype mismatch to raise ValueError or TypeError"


def test_vector_env_buffer_hot_path_matches_compatibility_step_path():
    rng = np.random.default_rng(1234)
<<<<<<< HEAD
    hypers = ObservationSpaceHypers()
    encoder = ObservationEncoder(hypers)
=======
    config = ObservationConfig()
    encoder = ObservationEncoder(config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    buffers = allocate_vector_buffers(num_envs, hypers)
=======
    buffers = allocate_vector_buffers(num_envs, config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    hot.set_buffers(buffers)

    compat_reset = compat.reset_all(player_configs)
    hot.reset_all_into_buffers(player_configs)

    for env_index, (obs, _) in enumerate(compat_reset):
        assert_buffer_observation_equal(buffers, env_index, encoder.encode(obs))
    np.testing.assert_allclose(buffers["rewards"], np.zeros(num_envs, dtype=np.float64))
    np.testing.assert_array_equal(
        buffers["terminated"], np.zeros(num_envs, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        buffers["truncated"], np.zeros(num_envs, dtype=np.uint8)
    )

    last_compat_step = compat_reset
    for _ in range(60):
        actions = [random_valid_action(obs, rng) for obs, _ in last_compat_step]
        compat_step = compat.step(actions)
        hot.step_into_buffers(actions)

        expected_rewards = np.array([step[1] for step in compat_step], dtype=np.float64)
        expected_terminated = np.array(
            [step[2] for step in compat_step], dtype=np.uint8
        )
        expected_truncated = np.array([step[3] for step in compat_step], dtype=np.uint8)
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
<<<<<<< HEAD
    hypers = ObservationSpaceHypers()
=======
    config = ObservationConfig()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    num_envs = 2
    player_configs = make_player_configs()
    env = managym.VectorEnv(
        num_envs=num_envs,
        seed=999,
        skip_trivial=True,
        opponent_policy="passive",
    )
<<<<<<< HEAD
    good = allocate_vector_buffers(num_envs, hypers)
=======
    good = allocate_vector_buffers(num_envs, config)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

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


def test_elemental_indestructible_encoding_preserves_outside_and_definition_columns() -> (
    None
):
    env = managym.Env(seed=161, skip_trivial=False)
    env.reset(make_player_configs())
    env.scenario_force_battlefield(0, "Gray Ogre")
    obs = env.scenario_refresh()
    cards = obs.agent_cards
    card = cards[0]
    keywords = card.keywords
    keywords.indestructible = True
    card.keywords = keywords
    cards[0] = card
    obs.agent_cards = cards
    permanents = obs.agent_permanents
    permanent = permanents[0]
    keywords = permanent.keywords
    keywords.indestructible = True
    permanent.keywords = keywords
    permanents[0] = permanent
    obs.agent_permanents = permanents

<<<<<<< HEAD
    encoder = ObservationEncoder(ObservationSpaceHypers())
=======
    encoder = ObservationEncoder(ObservationConfig())
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    expected = encoder.encode(obs)
    actual = env.encode_observation(obs)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    assert actual["agent_cards"].shape[-1] == 41
<<<<<<< HEAD
    assert actual["agent_permanents"].shape[-1] == 27
=======
    assert actual["agent_permanents"].shape[-1] == 28
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    assert actual["agent_cards"][0, 38] == 1
    assert actual["agent_cards"][0, 37] == 0
    assert actual["agent_permanents"][0, 25] == 1
    assert actual["agent_permanents"][0, 24] == permanent.registry_key + 1
