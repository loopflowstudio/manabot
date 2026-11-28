"""Shared raw prefixes preserve sampled decisions through learning and incident capture."""

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
    MemoryStore,
    Prefix,
    PrefixIdentity,
    RetainedMemory,
    archive_memory,
    gather_memory,
    restore_memory,
)
from manabot.env.observation import ObservationSpace, step_buffers
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
from manabot.policy.decision_memory import retained_memory
from manabot.policy.spans import ObservationSelection
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
from tests.model.test_decision_memory import memory_root, policy
from tests.training.memory_chunks import (
    capacities,
    capture_stream,
    chunk_batch,
    record,
)

CAPACITIES = (4, 1024, 1024, 16384)


def small_space() -> ObservationSpace:
    space, _ = memory_root()
    return space


def single(
    space: ObservationSpace, batch: managym.ObservationBatch
) -> ObservationSelection:
    return ObservationSelection(space, ((batch, 0),))


def test_memory_prefix_tail_fork_reset_and_duplicate_gather() -> None:
    space = small_space()
    store = MemoryStore(CAPACITIES)
    identity = PrefixIdentity("game-1", 0, "root")
    batches = [
        chunk_batch(space, [record(0)]),
        # Coalescing changes no row counts; old samples keep their multiplicity.
        chunk_batch(space, [record(0, count=2)]),
        chunk_batch(space, [record(0, count=2, life=9)]),
        chunk_batch(space, []),
    ]
    before = store.capture(single(space, batches[0]), 0, identity)
    after = store.capture(single(space, batches[1]), 0, identity)
    assert before.tail is not None and after.tail is not None
    assert before.tail.previous is after.tail.previous
    assert store.capture(single(space, batches[1]), 0, identity).tail is after.tail
    assert before.tail.chunk is not after.tail.chunk
    fork = store.capture(
        single(space, batches[2]), 0, PrefixIdentity("game-1", 0, "fork")
    )
    assert fork.tail is not None
    assert fork.tail.chunk is not after.tail.chunk
    empty = store.capture(
        single(space, batches[3]), 0, PrefixIdentity("game-2", 1, "root")
    )
    recorded = RecordedObservations(
        ObservationSelection(
            space,
            tuple((batch, 0) for batch in batches),
            RetainedMemory((before, after, fork, empty), CAPACITIES),
        ),
        np.array([0, 1, 2, 3]),
    )
    selected = recorded.gather(np.array([2, 0, 2, 3, 1]))
    assert selected.memory is not None
    rows = gather_memory(selected.memory.prefixes)
    np.testing.assert_array_equal(rows["decision_records"][:, 0, 3], [2, 1, 2, 0, 2])
    np.testing.assert_array_equal(rows["decision_records"][:, 0, 21], [9, 20, 9, 0, 20])
    assert "decision_records" not in recorded.rows.fields
    archive = archive_memory(selected.memory.prefixes, CAPACITIES)
    restored = gather_memory(restore_memory(archive))
    for key in MEMORY_FIELDS:
        np.testing.assert_array_equal(restored[key], rows[key])
    bad = tuple(a.copy() for a in archive.chunks[0])
    bad[0].flat[0] += 1
    with pytest.raises(ValueError, match="corrupt"):
        restore_memory(replace(archive, chunks=(bad, *archive.chunks[1:])))
    with pytest.raises(ValueError, match="missing"):
        restore_memory(replace(archive, chunks=archive.chunks[:-1]))
    with pytest.raises(ValueError):
        before.tail.chunk.rows[0].setflags(write=True)
    assert space.shapes["decision_records"] == (4, 74)


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_retained_memory_prepares_exactly_as_native_tables(
    encoder: Literal["transformer", "gru"],
    summaries: int,
) -> None:
    space = small_space()
    native = single(space, chunk_batch(space, [record(0)]))
    store = MemoryStore(CAPACITIES)
    prefix = store.capture(native, 0, PrefixIdentity("game", 1, "root"))
    kept = ObservationSelection(
        space, native.coordinates, RetainedMemory((prefix,), CAPACITIES)
    )
    model = policy(space, encoder, summaries)
    native_input = model.prepare(native)
    kept_input = model.prepare(kept)
    expected = array_fields(native_input, np.ndarray)
    actual = array_fields(kept_input, np.ndarray)
    for key in expected:
        if key.startswith("/memory/"):
            continue  # a retained GRU prefix folds shared causal nodes
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with torch.no_grad():
        torch.testing.assert_close(
            model.encode_decision_memory(on_device(kept_input, "cpu")),
            model.encode_decision_memory(on_device(native_input, "cpu")),
            atol=2e-6,
            rtol=2e-5,
        )
    with pytest.raises(ValueError, match="capacity mismatch"):
        model.prepare(
            ObservationSelection(
                space,
                native.coordinates,
                RetainedMemory((prefix,), (8, 1024, 1024, 16384)),
            )
        )

    opening = single(space, chunk_batch(space, []))
    empty = store.capture(opening, 0, PrefixIdentity("empty", 1, "root"))
    expected_empty = array_fields(model.prepare(opening), np.ndarray)
    actual_empty = array_fields(
        model.prepare(
            ObservationSelection(
                space, opening.coordinates, RetainedMemory((empty,), CAPACITIES)
            )
        ),
        np.ndarray,
    )
    for key in expected_empty:
        if key.startswith("/memory/"):
            continue
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
        before = gather_memory(
            first.observations(np.array([15, 0, 15])).observation.memory.prefixes
        )
        assert retained_memory(first.next_obs) is not None
        assert not set(MEMORY_FIELDS).intersection(first.next_obs.fields)
        bootstrap_before = gather_memory(first.next_obs.memory.prefixes)
        # Collected rows hold no memory tables: their history is the prefix.
        with torch.no_grad():
            expected_value = model.get_value(
                on_device(model.prepare(first.next_obs, inference=True), "cpu")
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
                assert not set(MEMORY_FIELDS).intersection(behavior.observation.fields)
                assert torch.isfinite(model.get_value(learner_input)).all()
            # Prepared input cannot keep temporary raw padded arrays alive.
            assert not gathered  # retained preparation never gathers dense history
            memory = retained_memory(first.next_obs)
            assert memory is not None
            prefix = memory.prefixes[0]
            [coordinate] = first.next_obs.coordinates
            selected = ObservationSelection(
                space,
                (coordinate,) * 3,
                RetainedMemory(
                    (prefix, Prefix(PrefixIdentity("empty", 1, "root"), None), prefix),
                    memory.capacities,
                ),
            )
            selected_batch = replace(
                first, next_obs=selected, next_done=np.zeros(3, bool)
            )
            selected_input = AdmittedRollout(model, selected_batch).next_observation(
                torch.device("cpu")
            )
            expected_input = on_device(model.prepare(selected, inference=True), "cpu")
            for key, value in array_fields(expected_input, torch.Tensor).items():
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
        assert not set(MEMORY_FIELDS).intersection(first.next_obs.fields)
        second = collect(original, model, 8)
        assert retained_memory(second.next_obs) is not None
        # The retained bootstrap survives native buffer reuse by later collection.
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                gather_memory(first.next_obs.memory.prefixes)[key],
                bootstrap_before[key],
            )
        with torch.no_grad():
            torch.testing.assert_close(
                model.get_value(admitted.next_observation(torch.device("cpu"))),
                expected_value,
            )
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                gather_memory(
                    first.observations(
                        np.array([15, 0, 15])
                    ).observation.memory.prefixes
                )[key],
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
        assert retained_memory(batch.next_obs) is not None
        assert not set(MEMORY_FIELDS).intersection(batch.next_obs.fields)
        assert len(batch.recorded.rows.memory.prefixes) == 4
        assert not set(MEMORY_FIELDS).intersection(batch.recorded.rows.fields)
    finally:
        close_collector(collector)


def test_live_suffix_capture_preserves_chunk_boundaries_coalescing_and_epochs() -> None:
    space = ObservationSpace(
        small_space().config.model_copy(update={"max_decision_records": 128})
    )
    caps = capacities(space)
    store = MemoryStore(caps)
    full_store = MemoryStore(caps)
    identity = PrefixIdentity("1", 0, "1")
    kept: list[Prefix] = []
    for count in (0, 1, 31, 32, 33, 64, 65, 97):
        native = single(space, chunk_batch(space, [record(i) for i in range(count)]))
        prefix = capture_stream(store, native, count)
        expected = full_store.capture(native, 0, identity)
        assert prefix.tail is not None or count == 0
        for key, value in gather_memory([expected]).items():
            np.testing.assert_array_equal(gather_memory([prefix])[key], value)
        kept.append(prefix)
    before = kept[-1]
    assert before.tail is not None and before.tail.previous is not None
    # Same counts do not mean unchanged content: coalescing rewrites the tail.
    coalesced = single(
        space,
        chunk_batch(space, [*(record(i) for i in range(96)), record(96, count=2)]),
    )
    after = capture_stream(store, coalesced, 98)
    assert after.tail is not None
    assert after.tail.previous is before.tail.previous
    assert after.tail.chunk.rows[0][0, 3] == 2
    assert before.tail.chunk.rows[0][0, 3] == 1
    # Reset/replay with the same seed and row count must not reuse closed chunks.
    replayed = single(
        space,
        chunk_batch(
            space,
            [
                record(0, life=19),
                *(record(i) for i in range(1, 96)),
                record(96, count=2),
            ],
        ),
    )
    reset = capture_stream(store, replayed, 98, epoch=2)
    expected = full_store.capture(replayed, 0, identity)
    assert reset.tail is not None and reset.tail.key == expected.tail.key
    assert reset.tail.key != after.tail.key
    shorter = single(space, chunk_batch(space, [record(i) for i in range(96)]))
    with pytest.raises(ValueError, match="regressed"):
        capture_stream(store, shorter, 99, epoch=2)


def test_native_live_suffix_counts_and_viewers_match_full_capture() -> None:
    capacities_ = (512, 16384, 256, 8192)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=capacities_[0],
            max_decision_reference_details=capacities_[1],
            max_decision_references=capacities_[2],
            max_decision_members=capacities_[3],
        )
    )
    buffers = step_buffers(1)
    env = managym.VectorEnv(
        1, seed=11, decision_memory=True, player_policies=("agent", "agent")
    )
    env.set_observation_layout(buffers, space.layout)
    env.reset_all_into_buffers(Match().to_rust())
    store, full_store = MemoryStore(capacities_), MemoryStore(capacities_)
    rng = random.Random(19)
    observed: set[int] = set()
    for _ in range(140):
        viewer = env.current_agent_indices()[0]
        observed.add(viewer)
        selection = ObservationSelection(space, tuple(env.observation_rows()))
        [prefix] = store.capture_live_rows(
            selection,
            np.array([0]),
            env.current_game_seeds(),
            [viewer],
            env.current_game_generations(),
            env.decision_memory_cursors(),
        )
        expected = full_store.capture(selection, 0, prefix.identity)
        cursor = env.decision_memory_cursors()[0][viewer]
        assert prefix.counts == tuple(cursor[:4])
        assert (prefix.tail.key if prefix.tail else None) == (
            expected.tail.key if expected.tail else None
        )
        env.step_into_buffers([rng.randrange(int(selection.action_counts[0]))])
    assert observed == {0, 1}


def test_sharded_native_suffix_metadata_tracks_each_stream() -> None:
    capacities_ = (512, 16384, 256, 8192)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=512,
            max_decision_reference_details=16384,
            max_decision_references=256,
            max_decision_members=8192,
        )
    )
    env = ShardedVectorEnv(2, 11, workers=2, decision_memory=True)
    try:
        env.set_observation_layout(step_buffers(2), "error", layout=space.layout)
        env.reset_all_into_buffers(Match().to_rust())
        generations = env.current_game_generations()
        assert len(set(generations)) == 2
        store = MemoryStore(capacities_)
        for _ in range(12):
            selection = ObservationSelection(space, tuple(env.observation_rows()))
            viewers = env.current_agent_indices()
            prefixes = store.capture_live_rows(
                selection,
                np.arange(2),
                env.current_game_seeds(),
                viewers,
                env.current_game_generations(),
                env.decision_memory_cursors(),
            )
            for row, (viewer, prefix) in enumerate(zip(viewers, prefixes, strict=True)):
                counts = env.decision_memory_counts()[row][viewer]
                full = MemoryStore(capacities_).capture(selection, row, prefix.identity)
                assert prefix.counts == tuple(counts) == full.counts
                assert (prefix.tail.key if prefix.tail else None) == (
                    full.tail.key if full.tail else None
                )
            env.step_into_buffers([0, 0])
        env.reset_all_into_buffers(Match().to_rust())
        assert not set(generations) & set(env.current_game_generations())
    finally:
        env.close()


def test_native_cursor_reuses_unchanged_prefix_and_versions_coalescing() -> None:
    space = ObservationSpace(
        small_space().config.model_copy(update={"max_decision_records": 128})
    )
    store = MemoryStore(capacities(space))
    native = single(space, chunk_batch(space, [record(0)]))
    first = capture_stream(store, native, 1)
    unchanged = capture_stream(store, native, 1)
    assert unchanged is first
    changed = capture_stream(
        store, single(space, chunk_batch(space, [record(0, count=2)])), 2
    )
    assert changed.tail is not first.tail
    assert first.tail is not None and changed.tail is not None
    assert first.tail.chunk.rows[0][0, 3] == 1
    assert changed.tail.chunk.rows[0][0, 3] == 2


def test_bulk_live_capture_skips_unchanged_rows_and_preserves_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = small_space()
    batch = chunk_batch(space, [record(0)])
    selection = ObservationSelection(space, ((batch, 0),) * 3)
    store = MemoryStore(CAPACITIES)
    seeds, seats, epochs = [11, 12, 13], [0, 1, 0], [1, 1, 1]
    counts = [
        int(selection.spans.tables[key].counts.ravel()[0]) for key in MEMORY_FIELDS
    ]
    cursors = [[[*counts, 1], [*counts, 1]] for _ in seeds]
    first = store.capture_live_rows(
        selection, np.arange(3), seeds, seats, epochs, cursors
    )
    for row, prefix in enumerate(first):
        expected = MemoryStore(CAPACITIES).capture(selection, row, prefix.identity)
        assert prefix.tail is not None and expected.tail is not None
        assert prefix.tail.key == expected.tail.key

    def compare_prefix(self: Prefix, other: object) -> bool:
        raise AssertionError("bulk cursor masking must not compare Python prefixes")

    monkeypatch.setattr(Prefix, "__eq__", compare_prefix)
    calls: list[int] = []
    original = store._capture_live_admitted

    def capture(
        buffers: ObservationSelection,
        row: int,
        identity: PrefixIdentity,
        **kwargs: object,
    ) -> Prefix:
        calls.append(row)
        return original(buffers, row, identity, **kwargs)

    monkeypatch.setattr(store, "_capture_live_admitted", capture)
    repeated = store.capture_live_rows(
        selection.select([2, 0]), np.array([2, 0]), seeds, seats, epochs, cursors
    )
    assert repeated[0] is first[2] and repeated[1] is first[0]
    assert not calls
    # Only the coalesced tail changes. Its identity stays the same object.
    coalesced = chunk_batch(space, [record(0, count=2)])
    selection = ObservationSelection(space, ((batch, 0), (coalesced, 0), (batch, 0)))
    cursors[1][1][4] += 1
    changed = store.capture_live_rows(
        selection, np.arange(3), seeds, seats, epochs, cursors
    )
    assert calls == [1]
    assert changed[1].identity is first[1].identity
    assert changed[1].tail.key != first[1].tail.key
    assert changed[0] is first[0] and changed[2] is first[2]
    # Switching viewers must capture, and changing generation must re-admit.
    seats[0] = 1
    epochs[2] += 1
    restored = store.capture_live_rows(
        selection, np.arange(3), seeds, seats, epochs, cursors
    )
    assert restored[2].identity.branch == str(epochs[2])
    assert restored[2].identity != first[2].identity
    assert restored[2].tail is first[2].tail
    assert restored[2].serving.segment is not first[2].serving.segment
    kept = gather_memory([first[1], changed[1]])
    assert kept["decision_records"][:, 0, 3].tolist() == [
        first[1].tail.chunk.rows[0][0, 3],
        first[1].tail.chunk.rows[0][0, 3] + 1,
    ]
    assert calls == [1, 0, 2]
    with pytest.raises(ValueError, match="live seat"):
        store.capture_live_rows(
            selection.select([0]), np.array([0]), seeds, [-1, 1, 0], epochs, cursors
        )
    # A seed change inside the same generation is a regression, not a hit.
    seeds[0] += 1
    with pytest.raises(ValueError, match="regressed"):
        store.capture_live_rows(
            selection.select([0]), np.array([0]), seeds, seats, epochs, cursors
        )


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
            memory = retained_memory(batch.next_obs)
            assert memory is not None
            if closure == "exact-quota":
                assert batch.next_done.tolist() == [False, True]
                live, parked = memory.prefixes
                assert live.identity.game == "197" and live.counts[0] > 0
                assert parked.counts == (0, 0, 0, 0)
                batch = collect(collector, model, 1)
                assert collector._env.current_game_seeds() == [197, 200]
            assert collector._env.current_agent_indices() == [-1, -1]
            memory = retained_memory(batch.next_obs)
            assert batch.next_done.all() and memory is not None
            assert all(prefix.counts == (0, 0, 0, 0) for prefix in memory.prefixes)
            assert any(
                prefix.counts[0] > 0 for prefix in batch.recorded.rows.memory.prefixes
            )
    finally:
        close_collector(collector)
