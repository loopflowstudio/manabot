"""Fixed-seat batch reads match scalar viewer encodings without advancing games."""

import numpy as np
import pytest

from manabot.env.failures import RecordedVectorEnv
from manabot.env.observation import (
    ObservationSpace,
    step_buffers,
)
from manabot.infra.config import ObservationConfig
import managym
from managym.disclosure import DecklistDisclosure
from tests.fixtures.observation_rows import observation_arrays, vector_arrays


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


def _rows(
    space: ObservationSpace, batch: managym.ObservationBatch
) -> dict[str, np.ndarray]:
    return vector_arrays(space, [(batch, row) for row in range(batch.batch_size)])


def _assert_rows(
    space: ObservationSpace,
    batch: managym.ObservationBatch,
    scalars: list[managym.Env],
    players: list[int],
) -> None:
    rows = _rows(space, batch)
    for slot, (scalar, player) in enumerate(zip(scalars, players, strict=True)):
        observation = scalar.observation_for_player(player)
        expected = observation_arrays(space, observation)
        for key in expected:
            np.testing.assert_array_equal(rows[key][slot], expected[key], err_msg=key)
        if scalar.current_agent_index() != player:
            assert not rows["actions_valid"][slot].any()
        if observation.opponent.main_counts is None:
            assert not rows["decklists"][slot, 1].any()


def _identity(read: managym.ObservationRead) -> tuple[int, int, int]:
    return read.game_id, read.seed, read.revision


@pytest.mark.parametrize("disclosure", ["open", "closed"])
@pytest.mark.parametrize("captured", [False, True])
def test_fixed_seat_scalar_parity_and_unchanged_execution(
    disclosure: DecklistDisclosure, captured: bool
) -> None:
    space = ObservationSpace()
    vector = (
        RecordedVectorEnv(num_envs=2, seed=42)
        if captured
        else managym.VectorEnv(2, seed=42)
    )
    players = _players(disclosure)
    outcomes = step_buffers(2)
    vector.set_observation_layout(outcomes, layout=space.layout)
    vector.reset_all_into_buffers(players)
    scalars = [managym.Env(seed=42 + slot) for slot in range(2)]
    for scalar in scalars:
        observation, _ = scalar.reset(players)
        assert observation.decklists_closed is (disclosure == "closed")
    try:
        for _ in range(6):
            before = {
                **vector_arrays(space, vector.observation_rows()),
                **{key: value.copy() for key, value in outcomes.items()},
            }
            frames = vector.semantic_decision_frames_json()
            info = vector.get_last_info()
            prefixes = (
                [game.model_dump() for game in vector.games] if captured else None
            )
            for seats in ([0, 1], [1, 0]):
                reads, batch = vector.observations_for_players(seats)
                _assert_rows(space, batch, scalars, seats)
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
            execution = {
                **vector_arrays(space, vector.observation_rows()),
                **outcomes,
            }
            for key in before:
                np.testing.assert_array_equal(execution[key], before[key], err_msg=key)
            vector.step_into_buffers([0, -999], [True, False])
            scalars[0].step(0)
        vector.reset_seeds_into_buffers([42, 43])
        for scalar in scalars:
            scalar.reset(players)
        _, batch = vector.observations_for_players([1, 0])
        _assert_rows(space, batch, scalars, [1, 0])
    finally:
        if captured:
            vector.close()


def test_fixed_seat_invalid_requests_change_nothing() -> None:
    vector = managym.VectorEnv(2, seed=42)
    space = ObservationSpace()
    with pytest.raises(ValueError, match="no registered observation layout"):
        vector.observations_for_players([0, 1])
    vector.set_observation_layout(step_buffers(2), layout=space.layout)
    with pytest.raises(managym.AgentError, match="must be reset"):
        vector.observations_for_players([0, 1])
    vector.reset_all(_players())
    frames = vector.semantic_decision_frames_json()
    for seats in ([], [0], [0, 1, 0], [0, 2], [0, -1]):
        with pytest.raises((managym.AgentError, OverflowError)):
            vector.observations_for_players(seats)
    with pytest.raises(ValueError):
        vector.set_observation_layout(
            step_buffers(2), layout=space.layout, action_overflow="truncate"
        )
    assert vector.semantic_decision_frames_json() == frames


def test_fixed_seat_registered_capacities_and_action_overflow() -> None:
    vector = managym.VectorEnv(1, seed=42)
    scalar = managym.Env(seed=42)
    players = _players()
    vector.reset_all(players)
    scalar.reset(players)
    tiny = ObservationSpace(ObservationConfig(max_actions=1))
    assert len(scalar.observation_for_player(0).action_space.actions) > 1
    vector.set_observation_layout(step_buffers(1), layout=tiny.layout)
    with pytest.raises(managym.ObservationCapacity, match="actions"):
        vector.observations_for_players([0])
    # A nonchooser has no actions, so this same capacity is sufficient.
    _, batch = vector.observations_for_players([1])
    _assert_rows(tiny, batch, [scalar], [1])
    vector.set_observation_layout(
        step_buffers(1), layout=tiny.layout, action_overflow="first-legal"
    )
    _, batch = vector.observations_for_players([0])
    prefix = ObservationSpace(
        ObservationConfig(max_actions=1, action_overflow="first-legal")
    )
    _assert_rows(prefix, batch, [scalar], [0])
    small = ObservationSpace(ObservationConfig(max_cards_per_player=1))
    vector.set_observation_layout(
        step_buffers(1), layout=small.layout, action_overflow="first-legal"
    )
    with pytest.raises(managym.ObservationCapacity, match="cards"):
        vector.observations_for_players([0])


def test_fixed_seat_reset_reassign_and_replacement_identities() -> None:
    space = ObservationSpace()
    vector = RecordedVectorEnv(num_envs=2, seed=42)
    vector.set_observation_layout(step_buffers(2), layout=space.layout)
    players = _players()
    vector.reset_all_into_buffers(players)
    try:
        first, _ = vector.observations_for_players([1, 0])
        vector.reset_seeds_into_buffers([42, 43])
        reset, _ = vector.observations_for_players([1, 0])
        assert all(a.game_id != b.game_id for a, b in zip(first, reset, strict=True))
        vector.load_slots_into_buffers(
            [managym.SlotLoad.fresh(0, list(reversed(players)))]
        )
        reassigned, _ = vector.observations_for_players([1, 0])
        assert reassigned[0].game_id != reset[0].game_id
        assert _identity(reassigned[1]) == _identity(reset[1])
        vector.step_isolated_into_buffers([-1, 0], [True, False])
        vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
        replaced, batch = vector.observations_for_players([1, 0])
        assert replaced[0].game_id != reassigned[0].game_id
        assert replaced[0].seed == 46
        assert _identity(replaced[1]) == _identity(reassigned[1])
        scalars = [managym.Env(seed=46), managym.Env(seed=43)]
        scalars[0].reset(list(reversed(players)))
        scalars[1].reset(players)
        _assert_rows(space, batch, scalars, [1, 0])
    finally:
        vector.close()


def test_fixed_seat_done_reads_terminal_including_maximum_seed() -> None:
    players = [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    vector = managym.VectorEnv(1, seed=2**64 - 1)
    vector.reset_all(players)
    space = ObservationSpace()
    vector.set_observation_layout(step_buffers(1), layout=space.layout)
    before = vector.observations_for_players([1])[0][0]
    for _ in range(120):
        [(_, _, terminated, truncated, _)] = vector.step([0])
        if terminated or truncated:
            break
    else:
        pytest.fail("bounded tiny-deck fixture did not terminate")
    [after], batch = vector.observations_for_players([1])
    assert after.game_id == before.game_id
    assert after.seed == 2**64 - 1
    scalar = managym.Env(seed=after.seed)
    scalar.reset(players)
    while not scalar.is_game_over():
        scalar.step(0)
    _assert_rows(space, batch, [scalar], [1])
