"""Live viewer memory agrees across independent encoders, replay and forks."""

import json
from typing import Literal

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
import managym
from managym.opponent_memory import read_records


def _root(disclosure: Literal["open", "closed"] = "open") -> managym.Env:
    env = managym.Env(seed=111)
    configs = Match().to_rust()
    for config in configs:
        config.disclosure = disclosure
    env.reset(configs)
    return env


def _observations(env: managym.Env) -> list[str]:
    return [
        env.observation_for_player(seat).opponent_memory_json() for seat in range(2)
    ]


def test_opponent_memory_metadata_and_legacy_admission() -> None:
    layout = managym.observation_layout(opponent_memory=True)
    assert len(layout["opponent_records"]["columns"]) == 18
    assert len(layout["opponent_details"]["columns"]) == 8
    assert "opponent_records" not in managym.observation_layout()
    with pytest.raises(ValueError, match="v11"):
        managym.observation_layout(opponent_memory=True, legacy=True)
    with pytest.raises(ValueError, match="v11"):
        ObservationSpace(
            ObservationConfig(
                policy_input_version=9, max_opponent_records=1, max_opponent_details=1
            )
        )


@pytest.mark.parametrize("disclosure", ["open", "closed"])
def test_opponent_memory_live_native_python_replay_fork_and_roundtrip(
    disclosure: Literal["open", "closed"],
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=32, max_opponent_details=32768)
    )
    env, replay = _root(disclosure), _root(disclosure)
    for _ in range(12):
        # Reproducible original positional choices; the engine's own validated
        # commit boundary also serves semantic Commands and built-in players.
        env.step(0)
        replay.step(0)
    assert _observations(env) == _observations(replay)
    assert any(read_records(source) for source in _observations(env))
    branch = env.clone_env()
    retained = _observations(env)
    branch.step(0)
    assert _observations(env) == retained
    for seat in range(2):
        obs = env.observation_for_player(seat)
        expected = space.encode(obs)
        actual = space.encoder.allocate()
        env.encode_observation_into(obs, actual)
        for key in ("opponent_records", "opponent_details"):
            np.testing.assert_array_equal(actual[key], expected[key])
        restored = managym.Observation.from_json(obs.toJSON())
        assert json.loads(restored.opponent_memory_json()) == json.loads(
            obs.opponent_memory_json()
        )
        for key in ("opponent_records", "opponent_details"):
            np.testing.assert_array_equal(space.encode(restored)[key], expected[key])
    env.reset(Match().to_rust())
    assert _observations(env) == ["[]", "[]"]


def test_opponent_memory_scalar_overflow_and_companion_validation() -> None:
    env = _root()
    for _ in range(12):
        env.step(0)
    obs = next(
        env.observation_for_player(seat)
        for seat in range(2)
        if read_records(env.observation_for_player(seat).opponent_memory_json())
    )
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=1, max_opponent_details=1)
    )
    with pytest.raises(managym.AgentError, match="Capacity"):
        space.encode(obs)
    with pytest.raises(ValueError, match="capacity exceeded"):
        env.encode_observation_into(obs, space.encoder.allocate())
    buffers = dict(ObservationSpace().encoder.allocate())
    buffers["opponent_records"] = np.zeros((4, 18), np.float32)
    with pytest.raises(ValueError, match="companion"):
        env.encode_observation_into(obs, buffers)


def test_opponent_memory_buffered_vector_and_builtin_choices_match_scalar() -> None:
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=32, max_opponent_details=32768)
    )
    configs = Match().to_rust()
    reference = managym.VectorEnv(2, seed=111, player_policies=("agent", "heuristic"))
    hot = managym.VectorEnv(2, seed=111, player_policies=("agent", "heuristic"))
    buffers = space.encoder.allocate(2)
    buffers.update(
        rewards=np.zeros(2, np.float64),
        terminated=np.zeros(2, np.uint8),
        truncated=np.zeros(2, np.uint8),
    )
    hot.set_buffers(buffers)
    reference.reset_all(configs)
    hot.reset_all_into_buffers(configs)
    retained = []
    for _ in range(8):
        rows = reference.step([0, 0])
        hot.step_into_buffers([0, 0])
        for index, (obs, _, _, _, _) in enumerate(rows):
            expected = space.encode(obs)
            retained.extend(read_records(obs.opponent_memory_json()))
            for key in ("opponent_records", "opponent_details"):
                np.testing.assert_array_equal(buffers[key][index], expected[key])
    assert retained, "built-in opponent choices must reach the external viewer"
