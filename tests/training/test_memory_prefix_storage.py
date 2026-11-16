"""Shared raw prefixes preserve sampled decisions through learning and recovery."""

from dataclasses import replace
from io import BytesIO
from typing import Literal
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.rl import NumericalError
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.backends.torch.state import capture_incident, load, save
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
)
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig, RewardConfig
from manabot.training.rollout.collection import (
    CollectorSnapshot,
    RecordedObservations,
    SeatRoutedCollector,
)
from tests.model.test_opponent_memory import memory_input, policy


def test_memory_prefix_tail_fork_reset_and_duplicate_gather() -> None:
    space, dense = memory_input()
    store = MemoryStore((4, 1024))
    identity = PrefixIdentity("game-1", 0, "root")
    before = store.capture(dense, 0, identity)
    # Coalescing changes no row counts; old samples keep their multiplicity.
    dense["opponent_records"][0, 0, 3] = 2
    dense["opponent_records"][0, 0, 9] = 1
    after = store.capture(dense, 0, identity)
    assert before.details[0] is after.details[0]
    assert store.capture(dense, 0, identity).tail is after.tail
    assert before.records[0] is not after.records[0]
    dense["opponent_details"][0, 0, 6] = 9
    fork = store.capture(dense, 0, PrefixIdentity("game-1", 0, "fork"))
    assert fork.details[0] is not after.details[0]
    for key in MEMORY_FIELDS:
        dense[key].fill(0)  # Later games overwrite native buffers.
    empty = store.capture(dense, 0, PrefixIdentity("game-2", 1, "root"))
    recorded = RecordedObservations(
        {},
        np.array([0, 1, 2, 3]),
        memory=(before, after, fork, empty),
        memory_capacities=(4, 1024),
    )
    selected = recorded.gather(np.array([2, 0, 2, 3, 1]))
    assert isinstance(selected, CompactObservation)
    np.testing.assert_array_equal(
        selected["opponent_records"][:, 0, 3], [2, 1, 2, 0, 2]
    )
    np.testing.assert_array_equal(
        selected["opponent_details"][:, 0, 6], [9, 20, 9, 0, 20]
    )
    assert recorded.ordered((4, 1)) == {}
    assert "opponent_records" not in recorded.rows
    archive = archive_memory(selected.prefixes, (4, 1024))
    restored = gather_memory(restore_memory(archive))
    for key in MEMORY_FIELDS:
        np.testing.assert_array_equal(restored[key], selected[key])
    bad = archive.chunks[0].copy()
    bad.flat[0] += 1
    with pytest.raises(ValueError, match="corrupt"):
        restore_memory(replace(archive, chunks=(bad, *archive.chunks[1:])))
    with pytest.raises(ValueError, match="missing"):
        restore_memory(replace(archive, chunks=archive.chunks[:-1]))
    with pytest.raises(ValueError):
        before.details[0].rows.setflags(write=True)
    assert space.encoder.shapes["opponent_records"] == (4, 18)


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_compact_memory_prepares_exactly_as_dense(
    encoder: Literal["transformer", "gru"],
    summaries: int,
) -> None:
    space, dense = memory_input()
    store = MemoryStore((4, 1024))
    prefix = store.capture(dense, 0, PrefixIdentity("game", 1, "root"))
    compact = CompactObservation(
        {k: v for k, v in dense.items() if k not in MEMORY_FIELDS}, [prefix], (4, 1024)
    )
    model = policy(space, encoder, summaries)
    expected = array_fields(model.prepare(dense), np.ndarray)
    actual = array_fields(model.prepare(compact), np.ndarray)
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with pytest.raises(ValueError, match="capacity mismatch"):
        model.prepare(CompactObservation(compact, [prefix], (8, 1024)))
    # Cropped arrays without their explicit raw contract are rejected.
    with pytest.raises(ValueError, match="capacity mismatch"):
        model.prepare(dict(compact))

    for key in MEMORY_FIELDS:
        dense[key].fill(0)
    empty = store.capture(dense, 0, PrefixIdentity("empty", 1, "root"))
    empty_compact = CompactObservation(
        {k: v for k, v in dense.items() if k not in MEMORY_FIELDS}, [empty], (4, 1024)
    )
    expected_empty = array_fields(model.prepare(dense), np.ndarray)
    actual_empty = array_fields(model.prepare(empty_compact), np.ndarray)
    for key in expected_empty:
        np.testing.assert_array_equal(
            actual_empty[key], expected_empty[key], err_msg=key
        )


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
def test_memory_collector_recovery_retains_old_batches_and_incident_budget(
    encoder: Literal["transformer", "gru"],
    summaries: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=64, max_opponent_details=65536)
    )
    model = policy(space, encoder, summaries)

    def collector() -> SeatRoutedCollector:
        return SeatRoutedCollector(
            space,
            Match(),
            Reward(RewardConfig()),
            num_envs=1,
            learning_seats="one",
            batch_closure="exact-quota",
            opponent_mode="self",
            recovery_max_game_microsteps=1000,
            numerics=CollectionSession(device="cpu", seed=111, precision="float32"),
        )

    original, resumed = collector(), collector()
    try:
        first = original.collect(model, 16)
        before = first.observations(np.array([15, 0, 15])).observation
        assert isinstance(first.next_obs, PrefixObservation)
        assert not MEMORY_FIELDS.intersection(first.next_obs)
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
                assert not MEMORY_FIELDS.intersection(behavior.observation)
                assert torch.isfinite(model.get_value(learner_input)).all()
            # Prepared input cannot keep temporary raw padded arrays alive.
            assert gathered and all(ref() is None for ref in gathered)
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
        assert not MEMORY_FIELDS.intersection(first.next_obs)
        snapshot = original.snapshot()
        assert not MEMORY_FIELDS.intersection(snapshot.buffers)
        encoded = BytesIO()
        save(snapshot, encoded)
        encoded.seek(0)
        decoded = load(encoded)
        assert isinstance(decoded, CollectorSnapshot)
        resumed.restore(decoded, lambda: None)
        second = original.collect(model, 8)
        replayed = resumed.collect(model, 8)
        assert isinstance(second.next_obs, PrefixObservation)
        assert isinstance(replayed.next_obs, PrefixObservation)
        # Both retained bootstrap and collector replay survive native buffer reuse.
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                first.next_obs.materialize()[key], bootstrap_before[key]
            )
            np.testing.assert_array_equal(
                second.next_obs.materialize()[key], replayed.next_obs.materialize()[key]
            )
        with torch.no_grad():
            torch.testing.assert_close(
                model.get_value(admitted.next_observation(torch.device("cpu"))),
                expected_value,
            )
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                first.observations(np.array([15, 0, 15])).observation[key], before[key]
            )
            np.testing.assert_array_equal(
                second.observations(np.arange(8)).observation[key],
                replayed.observations(np.arange(8)).observation[key],
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
        assert not captured.shapes["opponent_memory"]["complete"]
        assert captured.shapes["opponent_memory"]["required_bytes"] > 1
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
        assert exact.shapes["opponent_memory"]["complete"]
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
        original.close()
        resumed.close()


def test_memory_frozen_opponent_uses_the_compact_raw_contract() -> None:
    space = ObservationSpace(
        ObservationConfig(max_opponent_records=64, max_opponent_details=65536)
    )
    model = policy(space, "transformer", 4)
    collector = SeatRoutedCollector(
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
        batch = collector.collect(model, 4)
        assert batch.actions.size == 4
        assert isinstance(batch.next_obs, PrefixObservation)
        assert not MEMORY_FIELDS.intersection(batch.next_obs)
        assert len(batch.recorded.memory) == 4
        assert not MEMORY_FIELDS.intersection(batch.recorded.rows)
    finally:
        collector.close()
