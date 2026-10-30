"""Parity checks for Rust observation encoding against Python observation encoding."""

from __future__ import annotations

import numpy as np
import pytest

# Local imports
from manabot.env.observation import (
    CARD_DIM,
    CARD_PERMISSION_INDEX,
    FACT_DURATION_INDEX,
    FACT_EFFECT,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_ATTACHED_INDEX,
    PERMANENT_DIM,
    PERMANENT_GRANTED_LAND_INDEX,
    PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
    RELATION_KEYWORDS_INDEX,
    RELATION_X_INDEX,
    ObservationConfig,
    ObservationEncoder,
)
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
    config: ObservationConfig,
) -> dict[str, np.ndarray]:
    encoder = ObservationEncoder(config)
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
    config: ObservationConfig,
) -> np.ndarray:
    object_to_index: dict[int, int] = {}
    current_object_index = 0

    object_to_index[obs.agent.id] = current_object_index
    current_object_index += 1
    object_to_index[obs.opponent.id] = current_object_index
    current_object_index += 1

    for cards in (obs.agent_cards, obs.opponent_cards):
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
        -1,
        dtype=np.int32,
    )
    for action_index, action in enumerate(
        obs.action_space.actions[: config.max_actions]
    ):
        for focus_index, object_id in enumerate(
            action.focus[: config.max_focus_objects]
        ):
            out[action_index, focus_index] = object_to_index.get(object_id, -1)

    return out


def test_observation_parity():
    """Run 200 games, encode every observation both ways, and assert exact parity."""
    rng = np.random.default_rng(42)
    env = managym.Env(seed=42, skip_trivial=True)
    config = ObservationConfig()
    encoder = ObservationEncoder(config)

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
    config = ObservationConfig()

    obs, _ = env.reset(make_player_configs())
    encoded = env.encode_observation(obs)

    expected_focus = expected_focus_indices(obs, config)
    np.testing.assert_array_equal(encoded["action_focus"], expected_focus)


def test_encode_observation_into_fills_preallocated_buffers_and_validates_contract() -> (
    None
):
    env = managym.Env(seed=5, skip_trivial=True)
    config = ObservationConfig()
    encoder = ObservationEncoder(config)
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
        (config.max_actions, encoder.action_dim), dtype=np.float64
    )
    try:
        env.encode_observation_into(obs, bad)
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised, "Expected dtype mismatch to raise ValueError or TypeError"


def test_vector_env_buffer_hot_path_matches_compatibility_step_path():
    rng = np.random.default_rng(1234)
    config = ObservationConfig()
    encoder = ObservationEncoder(config)
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
    buffers = allocate_vector_buffers(num_envs, config)
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
    config = ObservationConfig()
    num_envs = 2
    player_configs = make_player_configs()
    env = managym.VectorEnv(
        num_envs=num_envs,
        seed=999,
        skip_trivial=True,
        opponent_policy="passive",
    )
    good = allocate_vector_buffers(num_envs, config)

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

    encoder = ObservationEncoder(ObservationConfig())
    expected = encoder.encode(obs)
    actual = env.encode_observation(obs)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    assert actual["agent_cards"].shape[-1] == CARD_DIM
    assert actual["agent_permanents"].shape[-1] == PERMANENT_DIM
    assert actual["agent_cards"][0, 38] == 1
    assert actual["agent_cards"][0, 37] == 0
    assert actual["agent_permanents"][0, 25] == 1
    assert actual["agent_permanents"][0, 24] == permanent.registry_key + 1


def _token_low(name: str) -> int:
    """The low 16 bits of a learning-token value, as fact rows store it."""
    return managym.learning_token_value(name) & 0xFFFF


# Policy input v5 encodes every Observation fact. These states reach its new
# columns through real T3 games and through each observed field directly.
_T3_DECK = {
    "Plains": 5,
    "Island": 5,
    "Swamp": 4,
    "Mountain": 5,
    "Forest": 5,
    "Trusty Boomerang": 4,
    "Glider Staff": 4,
    "Rowdy Snowballers": 4,
    "United Front": 4,
    "Lo and Li, Twin Tutors": 2,
    "Firebending Lesson": 4,
    "Toph, the First Metalbender": 2,
    "White Lotus Tile": 4,
    "Sozin's Comet": 4,
}


def test_v5_native_and_python_encoders_agree_on_t3_games() -> None:
    rng = np.random.default_rng(5)
    env = managym.Env(seed=5, skip_trivial=True)
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=5))
    reached: set[str] = set()
    for _ in range(12):
        obs, _ = env.reset(
            [managym.PlayerConfig(name, dict(_T3_DECK)) for name in ("a", "b")]
        )
        for _ in range(400):
            expected = encoder.encode(obs)
            actual = env.encode_observation(obs, 5)
            for key, value in expected.items():
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
            permanents = np.concatenate(
                (actual["agent_permanents"], actual["opponent_permanents"])
            )
            facts = actual["facts"]
            for name, seen in (
                (
                    "stun",
                    facts[:, FACT_ID_INDEX + 1] == _token_low("CounterKind::stun"),
                ),
                ("effect", facts[:, FACT_KIND_INDEX] == FACT_EFFECT),
                ("attached", permanents[:, PERMANENT_ATTACHED_INDEX]),
                ("granted land", permanents[:, PERMANENT_GRANTED_LAND_INDEX]),
                ("stack X", actual["relations"][:, RELATION_X_INDEX]),
                (
                    "spell grant",
                    actual["relations"][
                        :, RELATION_KEYWORDS_INDEX : RELATION_KEYWORDS_INDEX + 14
                    ],
                ),
                ("face down", actual["agent_cards"][:, CARD_PERMISSION_INDEX + 11]),
            ):
                if seen.any():
                    reached.add(name)
            if obs.game_over or not obs.action_space.actions:
                break
            obs, *_ = env.step(random_valid_action(obs, rng))
    # Seeded play reaches every new column, so parity covers each of them.
    assert reached == {
        "stun",
        "effect",
        "attached",
        "granted land",
        "stack X",
        "spell grant",
        "face down",
    }


def test_v5_encodes_every_observed_permanent_and_card_field_in_parity() -> None:
    env = managym.Env(seed=11, skip_trivial=False)
    env.reset(make_player_configs())
    env.scenario_force_battlefield(0, "Mountain")
    env.scenario_force_battlefield(1, "Forest")
    obs = env.scenario_refresh()
    mine, theirs = obs.agent_permanents[0], obs.opponent_permanents[0]
    mine.counters = {"stun": 1, "fire": 2, "double_strike": 3}
    mine.attached_to = theirs.id
    mine.name, mine.creature_types = "Humble Merchant", ["Citizen"]
    mine.is_granted_land, mine.granted_firebending = True, 4
    mine.temp_subtypes = ["Island", "Construct", "Citizen"]
    mine.attacking, mine.blocking, mine.abilities_lost = True, theirs.id, True
    mine.used_abilities = [1, 5]
    # Later than both permanents (ranks 1 and 2).
    base = max(mine.timestamp, theirs.timestamp)
    assert mine.timestamp < theirs.timestamp
    pump = managym.ContinuousEffect()
    pump.modification, pump.power, pump.toughness = "ModifyPt", 2, 0
    pump.duration, pump.timestamp = "EndOfTurn", base + 1
    grant = managym.ContinuousEffect()
    grant.modification, grant.keywords = "AddKeywords", ["flying", "trample"]
    grant.duration, grant.timestamp = "EndOfTurn", base + 2
    grant.source_registry_key = theirs.registry_key
    mine.effects = [pump, grant]
    obs.agent_permanents = [mine]
    agent = obs.agent
    agent.keeps_red_mana = agent.no_maximum_hand_size = True
    kept = managym.ContinuousEffect()
    kept.modification, kept.color = "KeepUnspentMana", "Red"
    kept.duration, kept.timestamp = "EndOfTurn", base + 3
    agent.effects = [kept]
    obs.agent = agent

    encoder = ObservationEncoder(ObservationConfig(policy_input_version=5))
    expected = encoder.encode(obs)
    actual = env.encode_observation(obs, 5)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    row = actual["agent_permanents"][0]
    host = 2 + 2 * encoder.cards_per_player + encoder.perms_per_player
    # Attacking (v4 column 27), then v5: exhausted, once each turn, granted
    # firebending, attachment join, granted land, blocking join, lost
    # abilities, timestamp rank, validity.
    np.testing.assert_allclose(
        row[27:PERMANENT_DIM], [1, 0, 0, 0.4, host + 1, 1, host + 1, 1, 1, 1]
    )
    # Fact rows, in order: counters, added subtypes, set name and creature
    # type, used abilities, then effect records (one row per keyword).
    facts = actual["facts"]
    kinds = facts[facts[:, FACT_VALID_INDEX] == 1, FACT_KIND_INDEX].tolist()
    assert kinds == [1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 6, 6, 6]
    effects = facts[facts[:, FACT_KIND_INDEX] == FACT_EFFECT]
    # The agent's player row comes first; its record is the latest.
    np.testing.assert_allclose(
        effects[:, [FACT_OBJECT_INDEX, FACT_VALUE_INDEX, FACT_DURATION_INDEX]],
        [[1, 0, 1], [host + 1, 0.2, 1], [host + 1, 0, 1], [host + 1, 0, 1]],
    )
    assert effects[:, FACT_ORDER_INDEX].tolist() == [5, 3, 4, 4]
    assert effects[2, FACT_SOURCE_INDEX] == theirs.registry_key + 1
    np.testing.assert_allclose(
        actual["agent_player"][0, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX:],
        [1, obs.turn.turn_number / 20, 0, 1],
    )
    assert actual["agent_cards"][0, CARD_DIM - 1] == 1
    v4 = env.encode_observation(obs, 4)
    assert v4["agent_permanents"].shape[-1] == 29
    assert v4["agent_cards"].shape[-1] == 41
