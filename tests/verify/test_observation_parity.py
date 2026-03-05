"""Parity checks for Rust observation encoding against Python observation encoding."""

from __future__ import annotations

import numpy as np

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
    for action_index, action in enumerate(obs.action_space.actions[: hypers.max_actions]):
        for focus_index, object_id in enumerate(action.focus[: hypers.max_focus_objects]):
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
        "agent_cards_valid": np.full((hypers.max_cards_per_player,), -7.0, dtype=np.float32),
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
    except ValueError:
        raised = True
    assert raised, "Expected dtype mismatch to raise ValueError"
