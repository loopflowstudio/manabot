"""Detached native slots can switch timelines without mixing viewer memory or fresh games."""

import json

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.env.memory_store import MemoryStore, Prefix, gather_memory
from manabot.env.observation import ObservationSpace, allocate_vector_buffers
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.producers import ShardedVectorEnv
import managym
from tests.env.test_game_snapshot import setup


def test_overwriting_a_sibling_matches_dense_scalar_and_shares_immutable_history() -> (
    None
):
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=2048,
            max_decision_references=128,
            max_decision_members=256,
            decision_memory_include_own=True,
        )
    )
    vector = managym.VectorEnv(
        2,
        seed=42,
        decision_memory=True,
        decision_memory_include_own=True,
        auto_reset=False,
    )
    buffers = allocate_vector_buffers(space, 2)
    vector.set_buffers(buffers, layout=space.layout)
    vector.reset_all_into_buffers(setup("open"))
    vector.step_into_buffers([1, 0], [True, False])
    root, skipped = vector.snapshot_slots([True, False])
    assert root is not None and skipped is None and root.seed == 42
    assert root.state_digest == json.loads(root.to_replay_json())["state_digest"]
    meta, _ = vector.slot_metadata([True, False])
    assert meta is not None and meta.acting_player == root.metadata().acting_player

    store = MemoryStore((64, 2048, 128, 256))

    def capture() -> Prefix:
        return store.capture_live_rows(
            buffers,
            np.array([0]),
            vector.current_game_seeds(),
            vector.current_agent_indices(),
            vector.current_game_generations(),
            vector.decision_memory_cursors(),
        )[0]

    before = capture()
    retained = gather_memory([before])
    neighbor = {key: values[1].copy() for key, values in buffers.items()}
    options = managym.RestoreOptions.shuffle_libraries(101)
    for action in (0, 1):
        [installed] = vector.load_slots_into_buffers(
            [managym.SlotLoad.snapshot(0, root, options)]
        )
        assert installed.source_digest == root.state_digest and not installed.terminal
        assert installed.acting_player == meta.acting_player
        branch = root.restore(options)
        assert vector.state_digests([True, False])[0] == branch.state_digest()
        expected = space.allocate(1)
        managym.encode_observations_into([branch], space.layout, expected, 1)
        for key, values in expected.items():
            np.testing.assert_array_equal(buffers[key][0], values[0], err_msg=key)
        for key, values in neighbor.items():
            np.testing.assert_array_equal(buffers[key][1], values, err_msg=key)
        prefix = capture()
        assert prefix.identity.branch == str(installed.generation)
        assert prefix.identity != before.identity
        assert before.tail is not None and prefix.tail is before.tail
        assert prefix.serving.segment is not before.serving.segment
        vector.step_into_buffers([action, 0], [True, False])
        branch.step(action)
        assert vector.state_digests([True, False])[0] == branch.state_digest()
        assert capture().tail is not before.tail
    for key, rows in gather_memory([before]).items():
        np.testing.assert_array_equal(rows, retained[key])


def test_slot_admission_precedes_publication_and_seed_consumption() -> None:
    vector = managym.VectorEnv(2, seed=2**64 - 1, auto_reset=False)
    space = ObservationSpace()
    buffers = allocate_vector_buffers(space, 2)
    vector.set_buffers(buffers, layout=space.layout)
    assert vector.snapshot_slots([False, False]) == [None, None]
    with pytest.raises(managym.AgentError, match="reset"):
        vector.slot_metadata([False, True])
    vector.reset_all_into_buffers(setup())
    root, _ = vector.snapshot_slots([True, False])
    assert root is not None
    original = {key: values.copy() for key, values in buffers.items()}
    generations = vector.current_game_generations()
    other = managym.Env(seed=42, decision_memory=True)
    other.reset(setup())
    for bad, reason in [
        (managym.SlotLoad.fresh(1), "duplicate"),
        (managym.SlotLoad.fresh(2), "slot"),
        (managym.SlotLoad.snapshot(0, other.snapshot()), "memory"),
        (
            managym.SlotLoad.snapshot(
                0, root, managym.RestoreOptions.exact(forced_first_action=999)
            ),
            "forced_first_action",
        ),
    ]:
        with pytest.raises(managym.AgentError, match=reason):
            vector.load_slots_into_buffers([managym.SlotLoad.fresh(1), bad])
        assert vector.current_game_generations() == generations
        for key, values in original.items():
            np.testing.assert_array_equal(buffers[key], values)
    results = vector.load_slots_into_buffers(
        [managym.SlotLoad.fresh(1), managym.SlotLoad.snapshot(0, root)]
    )
    assert [r.slot for r in results] == [1, 0]
    assert results[0].source_digest is None
    assert vector.next_fresh_seeds() == [2**64 - 1] * 2


@pytest.mark.parametrize("isolated", [False, True])
def test_deferred_terminal_reports_once_and_preserves_fresh_evidence(
    isolated: bool,
) -> None:
    vector = managym.VectorEnv(2, seed=42, auto_reset=False)
    vector.enable_exposure_capture()
    vector.enable_statistics()
    space = ObservationSpace()
    buffers = allocate_vector_buffers(space, 2)
    vector.set_buffers(buffers, layout=space.layout)
    vector.reset_all_into_buffers(setup())
    fresh_generation = vector.current_game_generations()[0]
    for _ in range(64):
        prior, _ = vector.snapshot_slots([True, False])
        if isolated:
            vector.step_isolated_into_buffers([0, 0], [True, False])
        else:
            vector.step_into_buffers([0, 0], [True, False])
        if buffers["terminated"][0]:
            break
    else:
        pytest.fail("fixture did not reach a terminal decision")
    assert prior is not None
    winner = vector.get_last_info()[0]["winner_index"]
    assert vector.current_game_generations()[0] == fresh_generation
    assert vector.next_fresh_seeds() == [44, 45]
    assert vector.current_agent_indices()[0] == -1
    with pytest.raises(managym.AgentError, match="Terminal"):
        vector.step_into_buffers([0, 0], [True, False])
    [ended] = vector.load_slots_into_buffers(
        [
            managym.SlotLoad.snapshot(
                1, prior, managym.RestoreOptions.exact(forced_first_action=0)
            )
        ]
    )
    assert ended.terminal and ended.acting_player is None and buffers["terminated"][1]
    assert vector.get_last_info()[1]["winner_index"] == winner
    assert vector.next_fresh_seeds() == [44, 43]
    assert vector.statistics()[1] is None
    assert vector.take_exposure() == [[], []]
    fresh = vector.load_slots_into_buffers([managym.SlotLoad.fresh(i) for i in (0, 1)])
    assert [r.seed for r in fresh] == [44, 43]
    assert not buffers["terminated"].any()
    [summary] = vector.take_statistics()
    assert summary.generation == fresh_generation and summary.terminal
    first, second = vector.take_exposure()
    assert len(first) == 1 and json.loads(first[0])["terminal"] and second == []
    assert all(row is not None for row in vector.statistics())


def test_sparse_reads_beside_quarantined_slot_and_direct_replacement() -> None:
    space = ObservationSpace()
    vector = managym.VectorEnv(2, seed=42, auto_reset=False)
    vector.set_buffers(allocate_vector_buffers(space, 2), layout=space.layout)
    vector.reset_all_into_buffers(setup())
    vector.step_isolated_into_buffers([999, 0], [True, False])
    assert vector.get_last_info()[0]["game_error_class"] == "AgentError"
    with pytest.raises(managym.AgentError, match="poisoned"):
        vector.snapshot_slots([True, False])
    out = space.allocate(2)
    for value in out.values():
        value.fill(23)
    [read] = vector.write_observations_for_players([None, 0], out, layout=space.layout)
    assert read.slot == 1 and all((value[0] == 23).all() for value in out.values())
    _, root = vector.snapshot_slots([False, True])
    assert root is not None
    [branch] = vector.load_slots_into_buffers([managym.SlotLoad.snapshot(0, root)])
    assert branch.seed == 43 and vector.next_fresh_seeds() == [44, 45]
    vector.step_isolated_into_buffers([0, 0], [True, False])
    assert "game_error" not in vector.get_last_info()[0]


def test_last_snapshot_capacity_error_leaves_the_batch_untouched() -> None:
    config = ObservationConfig(
        max_decision_records=1,
        max_decision_references=128,
        max_decision_reference_details=2048,
        max_decision_members=256,
    )
    space = ObservationSpace(config)
    vector = managym.VectorEnv(
        2, seed=42, decision_memory_include_own=True, auto_reset=False
    )
    buffers = allocate_vector_buffers(space, 2)
    vector.set_buffers(buffers, layout=space.layout)
    vector.reset_all_into_buffers(setup())
    source = managym.Env(seed=42, decision_memory_include_own=True)
    source.reset(setup())
    source.step(1)
    source.step(0)
    before = {name: value.copy() for name, value in buffers.items()}
    generations = vector.current_game_generations()
    with pytest.raises(managym.ObservationCapacity, match="decision_records"):
        vector.load_slots_into_buffers(
            [managym.SlotLoad.fresh(1), managym.SlotLoad.snapshot(0, source.snapshot())]
        )
    assert vector.current_game_generations() == generations
    assert vector.next_fresh_seeds() == [44, 45]
    for name, value in before.items():
        np.testing.assert_array_equal(buffers[name], value, err_msg=name)


def test_detached_masked_capture_from_sharded_recording_vector() -> None:
    space = ObservationSpace()
    producers = ShardedVectorEnv(4, 42, workers=2)
    try:
        producers.set_buffers(allocate_vector_buffers(space, 4), layout=space.layout)
        producers.reset_all_into_buffers(setup())
        assert producers.next_fresh_seeds() == [46, 47, 48, 49]
        mask = [False, False, True, False]
        snapshots = producers.snapshot_slots(mask)
        metadata = producers.slot_metadata(mask)
        assert [s is not None for s in snapshots] == mask
        assert [m is not None for m in metadata] == mask
        root, meta = snapshots[2], metadata[2]
        assert root is not None and meta is not None and root.seed == 44
        assert meta.acting_player == root.metadata().acting_player
        reads = producers.write_observations_for_players(
            [None, None, 1, None], space.allocate(4), layout=space.layout
        )
        assert [(r.slot, r.seed, r.player) for r in reads] == [(2, 44, 1)]
    finally:
        producers.close()
    assert root.restore().state_digest() == root.state_digest


def test_return_to_fresh_restores_the_sampled_roster_and_seed() -> None:
    match = Match(
        MatchConfig.authored("elemental-world-v2", "elemental_ug", "elemental_bg")
    )
    roster = elemental_roster(match)
    vector = managym.VectorEnv(1, seed=42, auto_reset=False)
    vector.set_uniform_roster(roster)
    space = ObservationSpace()
    vector.set_buffers(allocate_vector_buffers(space, 1), layout=space.layout)
    vector.reset_all_into_buffers(match.to_rust())
    [(a, b)] = vector.current_matchup_indices()
    changed = (a + 1) % len(roster)
    source = managym.Env(seed=42)
    source.reset([roster[changed], roster[b]])
    vector.load_slots_into_buffers([managym.SlotLoad.snapshot(0, source.snapshot())])
    assert vector.current_matchup_indices() == [(changed, b)]
    [fresh] = vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
    assert fresh.seed == 42
    assert vector.current_matchup_indices() == [(a, b)]
    [seats] = vector.current_player_configs()
    assert [seat.decklist for seat in seats] == [roster[i].decklist for i in (a, b)]
