"""Detached native slots can switch timelines without mixing viewer memory or fresh games."""

from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.env.memory_store import MemoryStore, Prefix, gather_memory
from manabot.env.observation import ObservationSpace, allocate_vector_buffers
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.producers import ShardedVectorEnv
from manabot.training.rollout.snapshot_vector import SlotInstall, replay_snapshot
from manabot.training.rollout.snapshots import root_id
import managym
from managym.decision import Command
from managym.recording import (
    RecordedVectorEnv,
    ScalarCapture,
    read_record,
    read_records,
)
from managym.replay import replay_failure_inputs
from managym.statistics import read_statistics
from tests.env.test_game_snapshot import setup
from tests.env.test_vector_reducers import History
from tests.fixtures.recorded_commands import record_until_object_candidate


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


def test_recorded_slot_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    space = ObservationSpace()
    arrays = allocate_vector_buffers(space, 2)
    with closing(RecordedVectorEnv(num_envs=2, seed=42, games_path=path)) as vector:
        vector.set_buffers(arrays, layout=space.layout)
        vector.reset_all_into_buffers(setup())
        neighbor = vector.current_game_generations()[1]
        for _ in range(64):
            vector.step_into_buffers([0, 0], [True, False])
            if arrays["terminated"][0]:
                break
        else:
            pytest.fail("tiny deck did not reach terminal")
        assert vector.current_agent_indices()[0] == -1
        assert vector.current_game_seeds() == [42, 43]
        [record] = read_records(path)
        game = record.games[0]
        assert game.end is not None and game.end.status == "complete"
        assert game.end.winner == vector.get_last_info()[0]["winner_index"]

        seats = managym.GameSetup(list(reversed(setup())), decklists="mixed")
        request = managym.SlotLoad.fresh(1, player_configs=seats).with_slot(0)
        assert request.slot == 0
        [loaded] = vector.load_slots_into_buffers([request])
        assert (loaded.slot, loaded.seed, loaded.terminal) == (0, 44, False)
        moved = loaded.with_slot(1)
        assert (moved.slot, moved.seed, moved.generation) == (1, 44, loaded.generation)
        assert loaded.slot == 0
        assert vector.current_game_generations()[1] == neighbor
        assert not arrays["terminated"][0] and arrays["actions_valid"][0].any()
        assert vector.game_inputs(0).players[0].name == seats[0].name
        assert vector.game_inputs(0).players[0].disclosure == managym.resolve_decklists(
            seats, 44
        )
        [later] = vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
        assert (
            later.seed == 46 and vector.game_inputs(0).players[0].name == seats[0].name
        )
        finished, untouched = read_records(path)
        assert finished == record
        [opening] = untouched.games
        assert opening.seed == 44 and opening.actions == []
        assert opening.end is not None and opening.end.status == "interrupted"


def test_slot_admission_precedes_publication_and_seed_consumption() -> None:
    vector = managym.VectorEnv(2, seed=2**64 - 1)
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
        (managym.SlotLoad.fresh(0, []), "2 player configs"),
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
    vector = managym.VectorEnv(2, seed=42)
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
    vector = managym.VectorEnv(2, seed=42)
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


@pytest.mark.parametrize("sharded", [False, True])
def test_last_snapshot_capacity_error_leaves_the_batch_untouched(sharded: bool) -> None:
    config = ObservationConfig(
        max_decision_records=1,
        max_decision_references=128,
        max_decision_reference_details=2048,
        max_decision_members=256,
    )
    space = ObservationSpace(config)
    vector = (
        ShardedVectorEnv(2, seed=42, workers=2, decision_memory_include_own=True)
        if sharded
        else managym.VectorEnv(2, seed=42, decision_memory_include_own=True)
    )
    try:
        buffers = allocate_vector_buffers(space, 2)
        vector.set_buffers(buffers, layout=space.layout)
        vector.reset_all_into_buffers(setup())
        source = managym.Env(seed=42, decision_memory_include_own=True)
        source.reset(setup())
        source.step(1)
        source.step(0)
        before = {name: value.copy() for name, value in buffers.items()}
        generations = vector.current_game_generations()
        inputs = (
            [vector.game_inputs(i).model_dump() for i in (0, 1)]
            if isinstance(vector, ShardedVectorEnv)
            else []
        )
        with pytest.raises(managym.ObservationCapacity, match="decision_records"):
            vector.load_slots_into_buffers(
                [
                    managym.SlotLoad.fresh(1, list(reversed(setup()))),
                    managym.SlotLoad.snapshot(0, source.snapshot()),
                ]
            )
        assert vector.current_game_generations() == generations
        assert vector.next_fresh_seeds() == [44, 45]
        for name, value in before.items():
            np.testing.assert_array_equal(buffers[name], value, err_msg=name)
        if isinstance(vector, ShardedVectorEnv):
            assert [vector.game_inputs(i).model_dump() for i in (0, 1)] == inputs
        loaded = vector.load_slots_into_buffers(
            [
                managym.SlotLoad.fresh(1, list(reversed(setup()))),
                managym.SlotLoad.fresh(0),
            ]
        )
        assert [(entry.slot, entry.seed) for entry in loaded] == [(1, 45), (0, 44)]
        assert vector.current_player_configs()[1][0].name == "b"
    finally:
        if isinstance(vector, ShardedVectorEnv):
            vector.close()


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


def test_four_workers_share_cold_root_history_and_recapture_shuffled_branches() -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=2048,
            max_decision_references=128,
            max_decision_members=256,
            decision_memory_include_own=True,
        )
    )
    with closing(
        ShardedVectorEnv(
            4,
            42,
            workers=4,
            decision_memory=True,
            decision_memory_include_own=True,
        )
    ) as vector:
        buffers = allocate_vector_buffers(space, 4)
        vector.set_buffers(buffers, layout=space.layout)
        vector.reset_all_into_buffers(setup("open"))
        vector.step_into_buffers([1, 0, 0, 0], [True, False, False, False])
        neighbor, root = vector.capture_slots([3, 0])
        assert (neighbor.slot, root.slot) == (3, 0)
        order = [3, 0, 2, 1]
        options = managym.RestoreOptions.shuffle_libraries(101)
        siblings = [
            SlotInstall(
                slot, root_id(root.recipe), root.recipe, restore_options=options
            )
            for slot in order
        ]
        generations = vector.current_game_generations()
        for invalid, reason in [
            (replace(siblings[0], hot_handle=neighbor.hot_handle), "identity mismatch"),
            (replace(siblings[0], slot=4), "invalid"),
            (replace(siblings[0], slot=siblings[1].slot), "duplicate"),
        ]:
            with pytest.raises(ValueError, match=reason):
                vector.install_slots([siblings[1], invalid])
            assert vector.current_game_generations() == generations
        corrupt = replace(root.recipe, replay_json="{}")
        with pytest.raises(managym.AgentError, match="invalid snapshot replay"):
            vector.install_slots([SlotInstall(0, -1, None), SlotInstall(3, 7, corrupt)])
        assert vector.current_game_generations() == generations
        with patch(
            "manabot.training.rollout.snapshot_vector.replay_snapshot",
            wraps=replay_snapshot,
        ) as replayed:
            assert [r.slot for r in vector.install_slots(siblings)] == order
            replayed.assert_called_once_with(root.recipe)
        prefixes = MemoryStore((64, 2048, 128, 256)).capture_live_rows(
            buffers,
            np.arange(4),
            vector.current_game_seeds(),
            vector.current_agent_indices(),
            vector.current_game_generations(),
            vector.decision_memory_cursors(),
        )
        shared = prefixes[0].tail
        assert shared is not None and all(p.tail is shared for p in prefixes)
        assert len({id(p.serving.segment) for p in prefixes}) == 4
        assert len({p.identity.branch for p in prefixes}) == 4
        assert isinstance(root.hot_handle, managym.GameSnapshot)
        scalar = root.hot_handle.restore(options)
        assert [s.root_digest for s in vector.capture_slots(order)] == [
            scalar.state_digest()
        ] * 4
        vector.step_into_buffers([0, 1, 0, 1])
        first, second = vector.capture_slots([1, 0])
        assert first.root_digest != second.root_digest
        scalar.step(0)
        assert second.root_digest == scalar.state_digest()
        assert second.recipe.replay_json is not None
        assert (
            json.loads(second.recipe.replay_json)["interventions"][-1]["op"]
            == "shuffle_libraries"
        )
        assert replay_snapshot(second.recipe).state_digest == second.root_digest


@pytest.mark.parametrize("captured", [False, True])
def test_return_to_fresh_restores_the_sampled_roster_and_seed(captured: bool) -> None:
    match = Match(
        MatchConfig.authored("elemental-world-v2", "elemental_ug", "elemental_bg")
    )
    roster = elemental_roster(match)
    vector = (RecordedVectorEnv if captured else managym.VectorEnv)(num_envs=1, seed=42)
    vector.set_uniform_roster(roster)
    space = ObservationSpace()
    vector.set_buffers(allocate_vector_buffers(space, 1), layout=space.layout)
    vector.reset_all_into_buffers(match.to_rust())
    [(a, b)] = vector.current_matchup_indices()
    changed = (a + 1) % len(roster)
    source = managym.Env(seed=42)
    source.reset([roster[changed], roster[b]])
    with pytest.raises(managym.AgentError, match="roster"):
        vector.load_slots_into_buffers([managym.SlotLoad.fresh(0, setup())])
    vector.load_slots_into_buffers([managym.SlotLoad.snapshot(0, source.snapshot())])
    assert vector.current_matchup_indices() == [(changed, b)]
    [fresh] = vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
    assert fresh.seed == 42
    assert vector.current_matchup_indices() == [(a, b)]
    [seats] = vector.current_player_configs()
    decks = [roster[i].decklist for i in (a, b)]
    assert [seat.decklist for seat in seats] == decks
    if isinstance(vector, RecordedVectorEnv):
        assert [seat.decklist for seat in vector.game_inputs(0).players] == decks
        vector.close()


@pytest.mark.parametrize("isolated", [False, True])
def test_captured_deferred_terminals_write_fresh_games_once(
    tmp_path: Path, isolated: bool
) -> None:
    games = tmp_path / "games.jsonl"
    stats = tmp_path / "stats.jsonl"
    vector = RecordedVectorEnv(
        num_envs=3,
        seed=2**64 - 1,
        games_path=games,
        statistics_path=stats,
        reducers=[History()],
    )
    space = ObservationSpace()
    arrays = allocate_vector_buffers(space, 3)
    vector.set_buffers(arrays, layout=space.layout)
    vector.reset_all_into_buffers(setup())
    root, _, _ = vector.snapshot_slots([True, False, False])
    assert root is not None
    step = vector.step_isolated_into_buffers if isolated else vector.step_into_buffers
    with patch.object(
        managym.GameSnapshot,
        "to_replay_json",
        side_effect=AssertionError("serialized hot branch"),
    ):
        vector.load_slots_into_buffers([managym.SlotLoad.snapshot(1, root)])
        for count in range(1, 65):
            step([0, 0, 0], [True, False, False])
            if arrays["terminated"][0]:
                break
        else:
            pytest.fail("fixture did not reach terminal")
        [record] = read_records(games)
        assert record.games[0].end.status == "complete"
        assert len(record.games[0].actions) == count
        assert vector.rows()[0][0][:3] == (count, 0, "complete")
        vector.load_slots_into_buffers([managym.SlotLoad.snapshot(0, root)])
        step([0, 0, 0], [True, True, False])
        [fresh] = vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
        assert fresh.seed == 2**64 - 1
        assert vector.next_seeds == [fresh.seed] * 3
        step([0, 0, 0], [True, False, False])
        vector.close()
        vector.close()
    ended = [g for r in read_records(games) for g in r.games]
    assert [(g.stream, len(g.actions), g.end.status) for g in ended] == [
        (0, count, "complete"),
        (0, 1, "interrupted"),
        (2, 0, "interrupted"),
    ]
    assert [(s.stream, s.terminal) for s in read_statistics(stats)] == [
        (0, True),
        (0, False),
        (2, False),
    ]
    assert sorted((n, status) for n, _, status, _ in vector.rows()[0]) == sorted(
        (len(g.actions), g.end.status) for g in ended
    )


@pytest.mark.parametrize("journal", [False, True])
def test_branch_record_reproduces_commands_interventions_and_failed_suffix(
    tmp_path: Path, journal: bool
) -> None:
    source = managym.Env(seed=23)
    scalar = ScalarCapture(seed=23)
    frame, _ = record_until_object_candidate(source, scalar)
    command = Command(
        "branch",
        frame.revision,
        frame.offers[0]["id"],
        object_preconditions=(frame.object_candidates[0],),
    )
    source.execute_semantic_command_json(command.to_json())
    root = source.snapshot()
    scalar.close()
    cursor = tmp_path / "journal.jsonl"
    vector = RecordedVectorEnv(
        num_envs=1,
        journal_path=cursor if journal else None,
        failure_directory=tmp_path / "failures",
    )
    space = ObservationSpace()
    vector.set_buffers(allocate_vector_buffers(space, 1), layout=space.layout)
    vector.reset_all_into_buffers(setup())
    vector.load_slots_into_buffers(
        [
            managym.SlotLoad.snapshot(
                0, root, managym.RestoreOptions.shuffle_libraries(123)
            )
        ]
    )
    vector.step_into_buffers([0])
    if journal:
        saved = read_record(cursor).games[0]
        assert saved.actions[-1] == 0 and any(
            isinstance(a, str) and json.loads(a) == json.loads(command.to_json())
            for a in saved.actions
        )
        assert saved.interventions[-1].op == "shuffle_libraries"
    with pytest.raises(managym.AgentError):
        vector.step_into_buffers([999])
    vector.close()
    del source, root, vector
    [path] = (tmp_path / "failures").glob("*.jsonl")
    evidence = read_record(path)
    game = evidence.games[0]
    assert game.seed == 23 and game.actions[-2:] == [0, 999]
    assert game.committed == len(game.actions) - 1
    assert (
        game.object_candidate_frames
        and game.interventions[-1].at_action == len(game.actions) - 2
    )
    [result] = replay_failure_inputs(evidence)
    assert (
        result["at_retained_failure"]
        and result["matches_binding"]
        and result["matches_error"]
    )


def test_sharded_producers_support_arbitrary_worker_counts() -> None:
    space = ObservationSpace()
    with pytest.raises(ValueError, match="at least 2 workers"):
        ShardedVectorEnv(8, 42, workers=1)
    with pytest.raises(ValueError, match="at least one stream each"):
        ShardedVectorEnv(6, 42, workers=7)

    with closing(ShardedVectorEnv(8, 42, workers=6)) as vector:
        buffers = allocate_vector_buffers(space, 8)
        vector.set_buffers(buffers, layout=space.layout)
        vector.reset_all_into_buffers(setup())
        assert vector.next_fresh_seeds() == [50, 51, 52, 53, 54, 55, 56, 57]
        captured = vector.capture_slots([5, 0])
        assert [c.slot for c in captured] == [5, 0]
        installed = vector.install_slots(
            [
                SlotInstall(
                    7,
                    root_id(captured[1].recipe),
                    captured[1].recipe,
                    captured[1].hot_handle,
                ),
                SlotInstall(2, -1, None),
            ]
        )
        assert [r.slot for r in installed] == [7, 2]
        assert installed[0].root_digest == captured[1].root_digest
        assert (
            vector.current_game_seeds()[7] == 42
            and vector.current_game_seeds()[2] == 52
        )
