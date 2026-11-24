"""Shared raw prefixes preserve sampled decisions through learning and incident capture."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
import random
from typing import Literal
from unittest.mock import patch
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.rl import NumericalError
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.backends.torch.state import capture_incident
from manabot.env.match import Match, Reward
import manabot.env.memory_store as memory_store
from manabot.env.memory_store import (
    MEMORY_FIELDS,
    CompactObservation,
    MemoryStore,
    Prefix,
    PrefixIdentity,
    PrefixObservation,
    archive_memory,
    gather_memory,
    restore_memory,
    select_rows,
)
from manabot.env.observation import ObservationSpace, allocate_vector_buffers
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
from manabot.training.learning import BatchClosure
from manabot.training.rollout.collection import (
    Collector,
    RecordedObservations,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.rollout.producers import ShardedVectorEnv
import managym
from tests.model.test_decision_memory import memory_input, policy


def test_memory_prefix_tail_fork_reset_and_duplicate_gather() -> None:
    space, dense = memory_input()
    store = MemoryStore((4, 1024, 1024, 16384))
    identity = PrefixIdentity("game-1", 0, "root")
    before = store.capture(dense, 0, identity)
    # Coalescing changes no row counts; old samples keep their multiplicity.
    dense["decision_records"][0, 0, 3] = 2
    dense["decision_records"][0, 0, 9] = 1
    after = store.capture(dense, 0, identity)
    assert before.tail is not None and after.tail is not None
    assert before.tail.previous is after.tail.previous
    assert store.capture(dense, 0, identity).tail is after.tail
    assert before.tail.chunk is not after.tail.chunk
    dense["decision_records"][0, 0, 21] = 9
    fork = store.capture(dense, 0, PrefixIdentity("game-1", 0, "fork"))
    assert fork.tail is not None
    assert fork.tail.chunk is not after.tail.chunk
    for key in MEMORY_FIELDS:
        dense[key].fill(0)  # Later games overwrite native buffers.
    empty = store.capture(dense, 0, PrefixIdentity("game-2", 1, "root"))
    recorded = RecordedObservations(
        {},
        np.array([0, 1, 2, 3]),
        memory=(before, after, fork, empty),
        memory_capacities=(4, 1024, 1024, 16384),
    )
    selected = recorded.gather(np.array([2, 0, 2, 3, 1]))
    assert isinstance(selected, PrefixObservation)
    selected = selected.materialize()
    np.testing.assert_array_equal(
        selected["decision_records"][:, 0, 3], [2, 1, 2, 0, 2]
    )
    np.testing.assert_array_equal(
        selected["decision_records"][:, 0, 21], [9, 20, 9, 0, 20]
    )
    assert recorded.ordered((4, 1)) == {}
    assert "decision_records" not in recorded.rows
    archive = archive_memory(selected.prefixes, (4, 1024, 1024, 16384))
    restored = gather_memory(restore_memory(archive))
    for key in MEMORY_FIELDS:
        np.testing.assert_array_equal(restored[key], selected[key])
    bad = tuple(a.copy() for a in archive.chunks[0])
    bad[0].flat[0] += 1
    with pytest.raises(ValueError, match="corrupt"):
        restore_memory(replace(archive, chunks=(bad, *archive.chunks[1:])))
    with pytest.raises(ValueError, match="missing"):
        restore_memory(replace(archive, chunks=archive.chunks[:-1]))
    with pytest.raises(ValueError):
        before.tail.chunk.rows[0].setflags(write=True)
    assert space.shapes["decision_records"] == (4, 74)


def test_slicing_a_compact_export_preserves_edits_and_prefix_order() -> None:
    _, dense = memory_input()
    store = MemoryStore((4, 1024, 1024, 16384))
    full = store.capture(dense, 0, PrefixIdentity("game", 0, "root"))
    empty = Prefix(PrefixIdentity("empty", 0, "root"), None)
    obs = CompactObservation({}, (full, empty, full), (4, 1024, 1024, 16384))
    obs["decision_records"][2, 0, 3] = 21
    selected = select_rows(obs, slice(1, 3))
    assert isinstance(selected, CompactObservation)
    assert selected.prefixes == (empty, full)
    assert selected.memory_capacities == obs.memory_capacities
    assert selected["decision_records"][1, 0, 3] == 21
    assert np.shares_memory(selected["decision_records"], obs["decision_records"])


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_compact_memory_prepares_exactly_as_dense(
    encoder: Literal["transformer", "gru"],
    summaries: int,
) -> None:
    space, dense = memory_input()
    store = MemoryStore((4, 1024, 1024, 16384))
    prefix = store.capture(dense, 0, PrefixIdentity("game", 1, "root"))
    compact = CompactObservation(
        {k: v for k, v in dense.items() if k not in MEMORY_FIELDS},
        [prefix],
        (4, 1024, 1024, 16384),
    )
    model = policy(space, encoder, summaries)
    expected = array_fields(model.prepare(dense), np.ndarray)
    actual = array_fields(model.prepare(compact), np.ndarray)
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with pytest.raises(ValueError, match="capacity mismatch"):
        model.prepare(CompactObservation(compact, [prefix], (8, 1024, 1024, 16384)))
    # Cropped arrays without their explicit raw contract are rejected.
    with pytest.raises(ValueError, match="capacity mismatch"):
        model.prepare(dict(compact))

    for key in MEMORY_FIELDS:
        dense[key].fill(0)
    empty = store.capture(dense, 0, PrefixIdentity("empty", 1, "root"))
    empty_compact = CompactObservation(
        {k: v for k, v in dense.items() if k not in MEMORY_FIELDS},
        [empty],
        (4, 1024, 1024, 16384),
    )
    expected_empty = array_fields(model.prepare(dense), np.ndarray)
    actual_empty = array_fields(model.prepare(empty_compact), np.ndarray)
    for key in expected_empty:
        np.testing.assert_array_equal(
            actual_empty[key], expected_empty[key], err_msg=key
        )


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_memory_collector_retains_old_batches_and_incident_budget(
    encoder: Literal["transformer", "gru"],
    summaries: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=65536,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    model = policy(space, encoder, summaries)

    def collector() -> Collector:
        return open_collector(
            space,
            Match(),
            Reward(RewardConfig()),
            num_envs=1,
            learning_seats="one",
            batch_closure="exact-quota",
            opponent_mode="self",
            numerics=CollectionSession(device="cpu", seed=111, precision="float32"),
        )

    original = collector()
    try:
        first = collect(original, model, 16)
        before = first.observations(np.array([15, 0, 15])).observation.materialize()
        assert isinstance(first.next_obs, PrefixObservation)
        assert not set(MEMORY_FIELDS).intersection(first.next_obs)
        bootstrap_before = first.next_obs.materialize()
        with torch.no_grad():
            expected_value = model.get_value(
                on_device(model.prepare(bootstrap_before), "cpu")
            )
            gathered: list[weakref.ReferenceType[np.ndarray]] = []

            def observe_gather(prefixes: tuple[Prefix, ...]) -> dict[str, np.ndarray]:
                arrays = gather_memory(prefixes)
                gathered.extend(weakref.ref(a) for a in arrays.values())
                return arrays

            with monkeypatch.context() as patch:
                patch.setattr(memory_store, "gather_memory", observe_gather)
                admitted = AdmittedRollout(model, first)
                prepared_bootstrap = admitted.next_observation(torch.device("cpu"))
                actual_value = model.get_value(prepared_bootstrap)
                admitted.admit_rows(np.array([15, 0, 15]), 2)
                learner_input, behavior = admitted.rows(
                    np.array([15, 0, 15]), torch.device("cpu")
                )
                assert not set(MEMORY_FIELDS).intersection(behavior.observation)
                assert torch.isfinite(model.get_value(learner_input)).all()
            # Prepared input cannot keep temporary raw padded arrays alive.
            assert not gathered  # retained preparation never gathers dense history
            prefix = first.next_obs.prefixes[0]
            selected = PrefixObservation(
                {
                    key: np.repeat(value, 3, axis=0)
                    for key, value in first.next_obs.items()
                },
                [prefix, Prefix(PrefixIdentity("empty", 1, "root"), None), prefix],
                first.next_obs.memory_capacities,
            )
            selected_batch = replace(
                first, next_obs=selected, next_done=np.zeros(3, bool)
            )
            selected_input = AdmittedRollout(model, selected_batch).next_observation(
                torch.device("cpu")
            )
            expected_input = on_device(
                model.prepare(selected.materialize(), inference=True), "cpu"
            )
            for key, value in array_fields(expected_input, torch.Tensor).items():
                if key.startswith("/memory/"):
                    continue  # retained GRU bootstrap shares causal nodes
                torch.testing.assert_close(
                    array_fields(selected_input, torch.Tensor)[key],
                    value,
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                model.get_value(selected_input), model.get_value(expected_input)
            )
        torch.testing.assert_close(actual_value, expected_value)
        assert not set(MEMORY_FIELDS).intersection(first.next_obs)
        second = collect(original, model, 8)
        assert isinstance(second.next_obs, PrefixObservation)
        # The retained bootstrap survives native buffer reuse by later collection.
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                first.next_obs.materialize()[key], bootstrap_before[key]
            )
        with torch.no_grad():
            torch.testing.assert_close(
                model.get_value(admitted.next_observation(torch.device("cpu"))),
                expected_value,
            )
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                first.observations(np.array([15, 0, 15])).observation.materialize()[
                    key
                ],
                before[key],
            )
        learner = Learner(model, "cpu", minibatch_rows=4)
        error = NumericalError(
            "fixture", {"failed": torch.tensor([[False]] * 7 + [[True]])}
        )
        captured = capture_incident(
            error,
            learner,
            original,
            second,
            None,
            np.random.default_rng(1),
            evidence_bytes=1,
            max_rows=1,
        )
        assert not captured.shapes["decision_memory"]["complete"]
        assert captured.shapes["decision_memory"]["required_bytes"] > 1
        assert not any(key.startswith("memory/") for key in captured.evidence)
        assert not captured.full_batch_captured
        exact = capture_incident(
            error,
            learner,
            original,
            second,
            None,
            np.random.default_rng(1),
            evidence_bytes=32 * 1024**2,
            max_rows=1,
        )
        assert exact.shapes["decision_memory"]["complete"]
        assert int(exact.evidence["batch_row_indexes"][0]) == 7
        assert "memory/offsets" in exact.evidence
        shuffled = NumericalError(
            "fixture",
            {
                "indexes": torch.tensor([6, 1, 7]),
                "failed": torch.tensor([False, True, False]),
            },
        )
        mapped = capture_incident(
            shuffled,
            learner,
            original,
            second,
            None,
            np.random.default_rng(1),
            evidence_bytes=32 * 1024**2,
            max_rows=1,
        )
        assert int(mapped.evidence["batch_row_indexes"][0]) == 1
    finally:
        close_collector(original)


def test_memory_frozen_opponent_uses_the_compact_raw_contract() -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=65536,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    model = policy(space, "transformer", 4)
    collector = open_collector(
        space,
        Match(),
        Reward(RewardConfig()),
        num_envs=1,
        opponent_mode="frozen",
        opponent_agent=model,
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=CollectionSession(device="cpu", seed=111, precision="float32"),
    )
    try:
        batch = collect(collector, model, 4)
        assert batch.actions.size == 4
        assert isinstance(batch.next_obs, PrefixObservation)
        assert not set(MEMORY_FIELDS).intersection(batch.next_obs)
        assert len(batch.recorded.memory) == 4
        assert not set(MEMORY_FIELDS).intersection(batch.recorded.rows)
    finally:
        close_collector(collector)


def test_live_suffix_capture_preserves_chunk_boundaries_coalescing_and_epochs() -> None:
    _, sample = memory_input()
    capacities = (128, 1024, 1024, 16384)
    buffers = {
        key: np.zeros((1, cap, sample[key].shape[-1]), np.float32)
        for key, cap in zip(MEMORY_FIELDS, capacities, strict=True)
    }
    store = MemoryStore(capacities)
    dense_store = MemoryStore(capacities)
    identity = PrefixIdentity("repeated-seed", 0, "root")
    retained: list[Prefix] = []
    for count in (0, 1, 31, 32, 33, 64, 65, 97):
        for index in range(count):
            header = buffers["decision_records"][0, index]
            header[:] = sample["decision_records"][0, 0]
            header[1] = header[7] = header[9] = index
        prefix = store.capture_live(
            buffers, 0, identity, epoch=1, counts=(count, 0, 0, 0)
        )
        expected = dense_store.capture(buffers, 0, identity)
        assert prefix.tail is not None or count == 0
        for key, value in gather_memory([expected]).items():
            np.testing.assert_array_equal(gather_memory([prefix])[key], value)
        retained.append(prefix)
    before = retained[-1]
    assert before.tail is not None and before.tail.previous is not None
    # Same counts do not mean unchanged content: coalescing rewrites the tail.
    buffers["decision_records"][0, 96, 3] = 2
    buffers["decision_records"][0, 96, 9] = 97
    after = store.capture_live(buffers, 0, identity, epoch=1, counts=(97, 0, 0, 0))
    assert after.tail is not None
    assert after.tail.previous is before.tail.previous
    assert after.tail.chunk.rows[0][0, 3] == 2
    assert before.tail.chunk.rows[0][0, 3] == 1
    # Reset/replay with the same seed and row count must not reuse closed chunks.
    buffers["decision_records"][0, 0, 21] = 19
    reset = store.capture_live(buffers, 0, identity, epoch=2, counts=(97, 0, 0, 0))
    expected = dense_store.capture(buffers, 0, identity)
    assert reset.tail is not None and reset.tail.key == expected.tail.key
    assert reset.tail.key != after.tail.key
    with pytest.raises(ValueError, match="regressed"):
        store.capture_live(buffers, 0, identity, epoch=2, counts=(96, 0, 0, 0))


def test_native_live_suffix_counts_and_viewers_match_dense_capture() -> None:
    capacities = (512, 16384, 256, 8192)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=capacities[0],
            max_decision_reference_details=capacities[1],
            max_decision_references=capacities[2],
            max_decision_members=capacities[3],
        )
    )
    buffers = allocate_vector_buffers(space, 1)
    env = managym.VectorEnv(
        1, seed=11, decision_memory=True, player_policies=("agent", "agent")
    )
    env.set_buffers(buffers, layout=space.layout)
    env.reset_all_into_buffers(Match().to_rust())
    dense_buffers = space.allocate(1)
    store, dense_store = MemoryStore(capacities), MemoryStore(capacities)
    rng = random.Random(19)
    observed: set[int] = set()
    for _ in range(140):
        viewer = env.current_agent_indices()[0]
        observed.add(viewer)
        epoch = env.current_game_generations()[0]
        cursor = env.decision_memory_cursors()[0][viewer]
        counts = cursor[:4]
        env.write_observations_for_players([viewer], dense_buffers, layout=space.layout)
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(buffers[key], dense_buffers[key])
        identity = PrefixIdentity(str(env.current_game_seeds()[0]), viewer, "root")
        prefix = store.capture_live(
            buffers, 0, identity, epoch=epoch, counts=counts, tail_version=cursor[4]
        )
        expected = dense_store.capture(buffers, 0, identity)
        assert prefix.counts == tuple(counts)
        assert (prefix.tail.key if prefix.tail else None) == (
            expected.tail.key if expected.tail else None
        )
        env.step_into_buffers([rng.randrange(int(buffers["actions_valid"][0].sum()))])
    assert observed == {0, 1}


def test_sharded_native_suffix_metadata_tracks_each_stream() -> None:
    capacities = (512, 16384, 256, 8192)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=512,
            max_decision_reference_details=16384,
            max_decision_references=256,
            max_decision_members=8192,
        )
    )
    buffers = allocate_vector_buffers(space, 2)
    env = ShardedVectorEnv(2, 11, workers=2, decision_memory=True)
    try:
        env.set_buffers(buffers, layout=space.layout)
        env.reset_all_into_buffers(Match().to_rust())
        generations = env.current_game_generations()
        assert len(set(generations)) == 2
        store = MemoryStore(capacities)
        for _ in range(12):
            for row, viewer in enumerate(env.current_agent_indices()):
                counts = env.decision_memory_counts()[row][viewer]
                prefix = store.capture_live(
                    buffers,
                    row,
                    PrefixIdentity(str(env.current_game_seeds()[row]), viewer, "root"),
                    epoch=env.current_game_generations()[row],
                    counts=counts,
                )
                dense = store.capture(buffers, row, prefix.identity)
                assert prefix.counts == dense.counts
                assert (prefix.tail.key if prefix.tail else None) == (
                    dense.tail.key if dense.tail else None
                )
            env.step_into_buffers([0, 0])
        env.reset_all_into_buffers(Match().to_rust())
        assert not set(generations) & set(env.current_game_generations())
    finally:
        env.close()


def test_native_cursor_reuses_unchanged_prefix_and_versions_coalescing() -> None:
    _, sample = memory_input()
    capacities = (128, 1024, 1024, 16384)
    buffers = {
        key: np.zeros((1, cap, sample[key].shape[-1]), np.float32)
        for key, cap in zip(MEMORY_FIELDS, capacities, strict=True)
    }
    buffers["decision_records"][0, 0] = sample["decision_records"][0, 0]
    store = MemoryStore(capacities)
    identity = PrefixIdentity("seed", 0, "root")
    first = store.capture_live(
        buffers, 0, identity, epoch=1, counts=(1, 0, 0, 0), tail_version=1
    )
    unchanged = store.capture_live(
        buffers, 0, identity, epoch=1, counts=(1, 0, 0, 0), tail_version=1
    )
    assert unchanged is first
    buffers["decision_records"][0, 0, 3] = 2
    buffers["decision_records"][0, 0, 9] = 1
    changed = store.capture_live(
        buffers, 0, identity, epoch=1, counts=(1, 0, 0, 0), tail_version=2
    )
    assert changed.tail is not first.tail
    assert first.tail is not None and changed.tail is not None
    assert first.tail.chunk.rows[0][0, 3] == 1
    assert changed.tail.chunk.rows[0][0, 3] == 2


def test_bulk_live_capture_skips_unchanged_rows_and_preserves_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, sample = memory_input()
    buffers = {key: np.repeat(value, 3, axis=0) for key, value in sample.items()}
    # Use the supplied transport's actual capacities, including ordinary fields.
    capacities = tuple(buffers[key].shape[1] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    seeds, seats, epochs = [11, 12, 13], [0, 1, 0], [1, 1, 1]
    counts = [
        int(buffers["decision_records"][0, :, 19].sum()),
        int(buffers["decision_reference_details"][0, :, 5].sum()),
        int(buffers["decision_references"][0, :, 12].sum()),
        int(buffers["decision_records"][0, :, 15:18].sum()),
    ]
    cursors = [[[*counts, 1], [*counts, 1]] for _ in seeds]
    first = store.capture_live_rows(
        buffers, np.arange(3), seeds, seats, epochs, cursors
    )
    for row, prefix in enumerate(first):
        expected = MemoryStore(capacities).capture(buffers, row, prefix.identity)
        assert prefix.tail is not None and expected.tail is not None
        assert prefix.tail.key == expected.tail.key

    def compare_prefix(self: Prefix, other: object) -> bool:
        raise AssertionError("bulk cursor masking must not compare Python prefixes")

    monkeypatch.setattr(Prefix, "__eq__", compare_prefix)
    calls: list[int] = []
    original = store._capture_live_admitted

    def capture(
        buffers: Mapping[str, np.ndarray],
        row: int,
        identity: PrefixIdentity,
        *,
        epoch: int,
        counts: Sequence[int],
        tail_version: int | None,
    ) -> Prefix:
        calls.append(row)
        return original(
            buffers,
            row,
            identity,
            epoch=epoch,
            counts=counts,
            tail_version=tail_version,
        )

    monkeypatch.setattr(store, "_capture_live_admitted", capture)
    repeated = store.capture_live_rows(
        buffers, np.array([2, 0]), seeds, seats, epochs, cursors
    )
    assert repeated[0] is first[2] and repeated[1] is first[0]
    assert not calls
    # Only the coalesced tail changes. Its identity stays the same object.
    cursors[1][1][4] += 1
    buffers["decision_records"][1, 0, 3] += 1
    buffers["decision_records"][1, 0, 9] += 1
    changed = store.capture_live_rows(
        buffers, np.arange(3), seeds, seats, epochs, cursors
    )
    assert calls == [1]
    assert changed[1].identity is first[1].identity
    assert changed[1].tail.key != first[1].tail.key
    assert changed[0] is first[0] and changed[2] is first[2]
    # Switching viewers must capture, and changing generation must re-admit.
    seats[0] = 1
    epochs[2] += 1
    restored = store.capture_live_rows(
        buffers, np.arange(3), seeds, seats, epochs, cursors
    )
    assert restored[2].identity.branch == str(epochs[2])
    assert restored[2].identity != first[2].identity
    assert restored[2].tail is first[2].tail
    assert restored[2].serving.segment is not first[2].serving.segment
    retained = gather_memory([first[1], changed[1]])
    assert retained["decision_records"][:, 0, 3].tolist() == [
        first[1].tail.chunk.rows[0][0, 3],
        first[1].tail.chunk.rows[0][0, 3] + 1,
    ]
    assert calls == [1, 0, 2]
    with pytest.raises(ValueError, match="live seat"):
        store.capture_live_rows(
            buffers, np.array([0]), seeds, [-1, 1, 0], epochs, cursors
        )
    # A seed change inside the same generation is a regression, not a hit.
    seeds[0] += 1
    with pytest.raises(ValueError, match="regressed"):
        store.capture_live_rows(buffers, np.array([0]), seeds, seats, epochs, cursors)
    seeds[0] -= 1
    # An actual changed row keeps integer and range admission.
    cursors[0][1][4] += 1
    buffers["decision_records"][0, 0, 10] = np.nan
    with pytest.raises(ValueError):
        store.capture_live_rows(buffers, np.array([0]), seeds, seats, epochs, cursors)


@pytest.mark.parametrize("closure", ["complete-games", "exact-quota"])
def test_terminal_bootstrap_retains_empty_memory(closure: BatchClosure) -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=16,
            max_decision_references=256,
            max_decision_reference_details=512,
            max_decision_members=256,
            decision_memory_include_own=True,
        )
    )
    model = policy(space, "gru", 1)
    deck = {"Mountain": 7}
    collector = open_collector(
        space,
        Match(MatchConfig(hero_deck=deck, villain_deck=deck)),
        Reward(RewardConfig()),
        num_envs=2,
        seed=197,
        learning_seats="one",
        batch_closure=closure,
        opponent_mode="self",
        numerics=CollectionSession(device="cpu", seed=197),
    )
    try:
        with patch.object(
            torch,
            "multinomial",
            side_effect=lambda probs, *args, **kwargs: torch.zeros(
                (len(probs), 1), dtype=torch.long
            ),
        ):
            # With seven lands and the first offer, seat zero decides twice per game.
            batch = collect(collector, model, 1)
            assert collector._env.current_game_seeds() == [197, 198]
            assert isinstance(batch.next_obs, PrefixObservation)
            if closure == "exact-quota":
                assert batch.next_done.tolist() == [False, True]
                live, parked = batch.next_obs.prefixes
                assert live.identity.game == "197" and live.counts[0] > 0
                assert parked.counts == (0, 0, 0, 0)
                batch = collect(collector, model, 1)
                assert collector._env.current_game_seeds() == [197, 200]
            assert collector._env.current_agent_indices() == [-1, -1]
            assert batch.next_done.all() and isinstance(
                batch.next_obs, PrefixObservation
            )
            assert all(
                prefix.counts == (0, 0, 0, 0) for prefix in batch.next_obs.prefixes
            )
            assert any(prefix.counts[0] > 0 for prefix in batch.recorded.memory)
    finally:
        close_collector(collector)
