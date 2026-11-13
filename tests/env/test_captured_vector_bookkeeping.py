"""CapturedVector replay prefixes follow native games, not seeds or raw delegation."""

import numpy as np
import pytest
import torch

from manabot import managym
from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.vector_env import VectorEnv as TorchVectorEnv
from manabot.infra.config import MatchConfig, RewardConfig

_MAX_SEED = 2**64 - 1


def _players() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]


def _captured(
    seed: int, opponent_policy: str = "none"
) -> tuple[CapturedVector, dict[str, np.ndarray]]:
    vector = CapturedVector(num_envs=1, seed=seed, opponent_policy=opponent_policy)
    buffers = ObservationSpace().encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    return vector, buffers


def _seeds(vector: TorchVectorEnv) -> list[int]:
    return [snapshot.seed for snapshot in vector.hand_snapshots([0, 0])]


@pytest.mark.parametrize(
    "name",
    [
        "reset_all",
        "step",
        "take_exposure",
        "enable_exposure_capture",
        "scenario_naming_vocabulary_into_buffers",
    ],
)
def test_native_mutators_are_not_delegated(name: str) -> None:
    vector, _ = _captured(42)
    try:
        with pytest.raises(AttributeError, match="into_buffers"):
            getattr(vector, name)
        assert not hasattr(vector, "no_such_method")
    finally:
        vector.close()


def test_native_reads_are_delegated() -> None:
    vector, _ = _captured(42)
    try:
        vector.reset_all_into_buffers(_players())
        assert vector.current_game_seeds() == [42]
        assert vector.current_agent_indices() == [0]
        assert len(vector.current_game_generations()) == 1
        assert vector.semantic_decision_frames_json()[0] is not None
        assert len(vector.skip_trivial_counts()) == 1
    finally:
        vector.close()


@pytest.mark.parametrize("opponent_policy", ["none", "random", "heuristic"])
def test_terminal_replacement_at_saturated_seed_starts_a_new_prefix(
    opponent_policy: str,
) -> None:
    vector, buffers = _captured(_MAX_SEED, opponent_policy)
    try:
        vector.reset_all_into_buffers(_players())
        generation = vector.current_game_generations()[0]
        for _ in range(120):
            vector.step_into_buffers([0])
            if buffers["terminated"][0] or buffers["truncated"][0]:
                break
        else:
            pytest.fail("bounded tiny-deck fixture did not terminate")
        assert vector.current_game_seeds() == [_MAX_SEED]
        assert vector.current_game_generations()[0] != generation
        (game,) = vector.games
        assert (game.seed, game.actions, game.frame) == (_MAX_SEED, [], None)
    finally:
        vector.close()


@pytest.mark.parametrize("opponent_policy", ["none", "random", "heuristic"])
def test_same_seed_reset_starts_a_new_prefix(opponent_policy: str) -> None:
    vector, _ = _captured(42, opponent_policy)
    try:
        vector.reset_all_into_buffers(_players())
        vector.step_into_buffers([0])
        assert vector.games[0].actions == [0]
        generation = vector.current_game_generations()[0]
        vector.reset_seeds_into_buffers([42])
        assert vector.current_game_generations()[0] != generation
        assert (vector.games[0].seed, vector.games[0].actions) == (42, [])
    finally:
        vector.close()


@pytest.mark.parametrize("opponent_policy", ["none", "random", "heuristic"])
def test_torch_reset_with_a_seed_restarts_and_without_continues(
    opponent_policy: str,
) -> None:
    match = Match(MatchConfig(hero_deck={"Mountain": 12}, villain_deck={"Island": 12}))
    vector = TorchVectorEnv(
        2,
        match,
        ObservationSpace(),
        Reward(RewardConfig()),
        "cpu",
        seed=42,
        opponent_policy=opponent_policy,
    )
    try:
        first, _ = vector.reset(seed=42)
        first = {key: value.clone() for key, value in first.items()}
        assert _seeds(vector) == [42, 43]
        vector.step(torch.zeros(2, dtype=torch.int64))
        again, _ = vector.reset(seed=42)
        assert _seeds(vector) == [42, 43]
        for key, value in first.items():
            assert torch.equal(again[key], value), key
        vector.reset()
        assert _seeds(vector) == [44, 45]
    finally:
        vector.close()
