"""Fixed-seat batch reads match scalar viewer encodings without advancing games."""

import numpy as np
import pytest
import torch

from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
)
from manabot.env.vector_env import VectorEnv as TorchVectorEnv
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
import managym
from managym.disclosure import DecklistDisclosure


def _players(
    disclosure: DecklistDisclosure = "closed",
) -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig("a", {"Mountain": 8, "Gray Ogre": 4}),
            managym.PlayerConfig("b", {"Island": 8, "Wind Drake": 4}),
        ],
        decklists=disclosure,
    )


def _assert_rows(
    space: ObservationSpace,
    buffers: dict[str, np.ndarray],
    scalars: list[managym.Env],
    players: list[int],
) -> None:
    for slot, (scalar, player) in enumerate(zip(scalars, players, strict=True)):
        observation = scalar.observation_for_player(player)
        expected = encode_observation(space, observation)
        for key in expected:
            np.testing.assert_array_equal(
                buffers[key][slot], expected[key], err_msg=key
            )
        if scalar.current_agent_index() != player:
            assert not buffers["actions_valid"][slot].any()
        if observation.opponent.main_counts is None:
            assert not buffers["decklists"][slot, 1].any()


def _identity(read: managym.ObservationRead) -> tuple[int, int, int]:
    return read.game_id, read.seed, read.revision


@pytest.mark.parametrize("version", [9, 10])
@pytest.mark.parametrize("disclosure", ["open", "closed"])
@pytest.mark.parametrize("captured", [False, True])
def test_fixed_seat_scalar_parity_and_unchanged_execution(
    version: int, disclosure: DecklistDisclosure, captured: bool
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=version))
    vector = (
        CapturedVector(num_envs=2, seed=42)
        if captured
        else managym.VectorEnv(2, seed=42)
    )
    players = _players(disclosure)
    execution = allocate_vector_buffers(space, 2)
    vector.set_buffers(execution, layout=space.layout)
    vector.reset_all_into_buffers(players)
    scalars = [managym.Env(seed=42 + slot) for slot in range(2)]
    for scalar in scalars:
        observation, _ = scalar.reset(players)
        assert observation.decklists_closed is (disclosure == "closed")
    buffers = space.allocate(2)
    try:
        for _ in range(6):
            before = {key: value.copy() for key, value in execution.items()}
            frames = vector.semantic_decision_frames_json()
            info = vector.get_last_info()
            prefixes = (
                [game.model_dump() for game in vector.games] if captured else None
            )
            for seats in ([0, 1], [1, 0]):
                reads = vector.write_observations_for_players(
                    seats, buffers, layout=space.layout
                )
                _assert_rows(space, buffers, scalars, seats)
                for slot, (read, hand) in enumerate(
                    zip(reads, vector.hand_snapshots(seats), strict=True)
                ):
                    assert isinstance(read, managym.ObservationRead)
                    assert (read.slot, read.player) == (slot, seats[slot])
                    assert _identity(read) == (hand.game_id, hand.seed, hand.revision)
                    with pytest.raises(AttributeError):
                        read.player = 1 - read.player
            assert vector.semantic_decision_frames_json() == frames
            assert vector.get_last_info() == info
            if captured:
                assert [game.model_dump() for game in vector.games] == prefixes
            for key in before:
                np.testing.assert_array_equal(execution[key], before[key], err_msg=key)
            vector.step_into_buffers([0, -999], [True, False])
            scalars[0].step(0)
        vector.reset_seeds_into_buffers([42, 43])
        for scalar in scalars:
            scalar.reset(players)
        vector.write_observations_for_players([1, 0], buffers, layout=space.layout)
        _assert_rows(space, buffers, scalars, [1, 0])
    finally:
        if captured:
            vector.close()


def test_fixed_seat_invalid_requests_and_buffers_do_not_write() -> None:
    vector = managym.VectorEnv(2, seed=42)
    space = ObservationSpace()
    buffers = space.allocate(2)
    for value in buffers.values():
        value.fill(19)
    with pytest.raises(managym.AgentError, match="must be reset"):
        vector.write_observations_for_players([0, 1], buffers, layout=space.layout)
    vector.reset_all(_players())
    frames = vector.semantic_decision_frames_json()
    for seats in ([], [0], [0, 1, 0], [0, 2], [0, -1]):
        with pytest.raises((managym.AgentError, OverflowError)):
            vector.write_observations_for_players(seats, buffers, layout=space.layout)
    malformed = [dict(buffers) for _ in range(5)]
    malformed[0].pop("events_valid")
    malformed[1]["events_valid"] = np.zeros((1, 32), dtype=np.float32)
    malformed[2]["events_valid"] = buffers["events_valid"].astype(np.float64)
    malformed[3]["events_valid"] = np.zeros((2, 64), dtype=np.float32)[:, ::2]
    malformed[4]["events_valid"] = buffers["events_valid"].copy()
    malformed[4]["events_valid"].flags.writeable = False
    for bad in malformed:
        with pytest.raises((KeyError, ValueError, TypeError, BufferError)):
            vector.write_observations_for_players([0, 1], bad, layout=space.layout)
    tensors = {key: torch.from_numpy(value) for key, value in buffers.items()}
    with pytest.raises(TypeError, match="must be a NumPy array, got Tensor"):
        vector.write_observations_for_players([0, 1], tensors, layout=space.layout)
    with pytest.raises(ValueError):
        vector.write_observations_for_players(
            [0, 1], buffers, action_overflow="truncate", layout=space.layout
        )
    assert vector.semantic_decision_frames_json() == frames
    for value in buffers.values():
        assert (value == 19).all()


def test_fixed_seat_supplied_capacities_and_action_overflow() -> None:
    vector = managym.VectorEnv(1, seed=42)
    scalar = managym.Env(seed=42)
    players = _players()
    vector.reset_all(players)
    scalar.reset(players)
    tiny = ObservationSpace(ObservationConfig(max_actions=1))
    buffers = tiny.allocate(1)
    assert len(scalar.observation_for_player(0).action_space.actions) > 1
    with pytest.raises(managym.AgentError, match="[Cc]apacity|actions"):
        vector.write_observations_for_players([0], buffers, layout=tiny.layout)
    vector.write_observations_for_players(
        [0], buffers, action_overflow="first-legal", layout=tiny.layout
    )
    prefix = ObservationSpace(
        ObservationConfig(max_actions=1, action_overflow="first-legal")
    )
    _assert_rows(prefix, buffers, [scalar], [0])
    # A nonchooser has no actions, so this same capacity is sufficient.
    vector.write_observations_for_players([1], buffers, layout=tiny.layout)
    _assert_rows(tiny, buffers, [scalar], [1])
    small = ObservationSpace(ObservationConfig(max_cards_per_player=1))
    with pytest.raises(managym.AgentError, match="[Cc]apacity|cards"):
        vector.write_observations_for_players(
            [0], small.allocate(1), action_overflow="first-legal", layout=small.layout
        )


def test_fixed_seat_reset_reassign_and_replacement_identities() -> None:
    space = ObservationSpace()
    vector = CapturedVector(num_envs=2, seed=42)
    vector.set_buffers(allocate_vector_buffers(space, 2), layout=space.layout)
    players = _players()
    vector.reset_all_into_buffers(players)
    buffers = space.allocate(2)
    try:
        first = vector.write_observations_for_players(
            [1, 0], buffers, layout=space.layout
        )
        vector.reset_seeds_into_buffers([42, 43])
        reset = vector.write_observations_for_players(
            [1, 0], buffers, layout=space.layout
        )
        assert all(a.game_id != b.game_id for a, b in zip(first, reset, strict=True))
        vector.reassign_roots_into_buffers([list(reversed(players)), None])
        reassigned = vector.write_observations_for_players(
            [1, 0], buffers, layout=space.layout
        )
        assert reassigned[0].game_id != reset[0].game_id
        assert _identity(reassigned[1]) == _identity(reset[1])
        vector.step_isolated_into_buffers([-1, 0], [True, False])
        vector.replace_failed_into_buffers([True, False])
        replaced = vector.write_observations_for_players(
            [1, 0], buffers, layout=space.layout
        )
        assert replaced[0].game_id != reassigned[0].game_id
        assert replaced[0].seed == 44
        assert _identity(replaced[1]) == _identity(reassigned[1])
        scalars = [managym.Env(seed=44), managym.Env(seed=43)]
        scalars[0].reset(list(reversed(players)))
        scalars[1].reset(players)
        _assert_rows(space, buffers, scalars, [1, 0])
    finally:
        vector.close()


def test_fixed_seat_done_reads_replacement_including_maximum_seed() -> None:
    players = [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    vector = managym.VectorEnv(1, seed=2**64 - 1)
    vector.reset_all(players)
    space = ObservationSpace()
    buffers = space.allocate(1)
    before = vector.write_observations_for_players([1], buffers, layout=space.layout)[0]
    for _ in range(120):
        [(_, _, terminated, truncated, _)] = vector.step([0])
        if terminated or truncated:
            break
    else:
        pytest.fail("bounded tiny-deck fixture did not terminate")
    after = vector.write_observations_for_players([1], buffers, layout=space.layout)[0]
    assert after.game_id != before.game_id
    assert after.seed == 2**64 - 1
    scalar = managym.Env(seed=after.seed)
    scalar.reset(players)
    _assert_rows(space, buffers, [scalar], [1])


def test_fixed_seat_torch_wrapper_preserves_tensor_buffers_on_seed_rebuild() -> None:
    match = Match(MatchConfig(hero_deck={"Mountain": 12}, villain_deck={"Island": 12}))
    space = ObservationSpace()
    vector = TorchVectorEnv(2, match, space, Reward(RewardConfig()), "cpu", seed=42)
    consumer = ObservationSpace(ObservationConfig(max_events=5))
    buffers = consumer.allocate(2)
    try:
        previous = None
        for seed in (42, 99, 42):
            observations, _ = vector.reset(seed=seed)
            before = {key: value.clone() for key, value in observations.items()}
            reads = vector.write_observations_for_players(
                [1, 0], buffers, layout=consumer.layout
            )
            assert reads[0].seed == seed
            if previous is not None:
                assert reads[0].game_id != previous
            previous = reads[0].game_id
            scalars = [managym.Env(seed=seed + slot) for slot in range(2)]
            for scalar in scalars:
                scalar.reset(match.to_rust())
            _assert_rows(consumer, buffers, scalars, [1, 0])
            for key in observations:
                assert torch.equal(observations[key], before[key])
    finally:
        vector.close()
