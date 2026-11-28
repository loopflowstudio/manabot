"""Frozen collection reuses stable records without changing the acting policy."""

from dataclasses import replace
import gc
import weakref

import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.memory_store import (
    MemoryStore,
    Prefix,
    PrefixIdentity,
    RetainedMemory,
    archive_memory,
    gather_memory,
    record_tail,
    restore_memory,
)
from manabot.env.observation import ObservationSpace
from manabot.policy.prepare import Prepared
from manabot.policy.spans import ObservationSelection
import managym
from tests.model.test_decision_memory import policy
from tests.training.memory_chunks import (
    capacities,
    capture_stream,
    capture_streams,
    chunk_batch,
    chunk_input,
    chunk_space,
    record,
)


def single(
    space: ObservationSpace, batch: managym.ObservationBatch
) -> ObservationSelection:
    return ObservationSelection(space, ((batch, 0),))


def retained(
    native: ObservationSelection, prefixes: list[Prefix]
) -> ObservationSelection:
    """`native`'s rows with `prefixes` as their interned decision memory."""
    caps = capacities(native.space)
    return ObservationSelection(
        native.space, native.coordinates, RetainedMemory(tuple(prefixes), caps)
    )


def stateless(
    model: ActorCritic, native: ObservationSelection
) -> Prepared[torch.Tensor]:
    """The reference: each row's own native memory tables, without a fold."""
    return on_device(model.prepare(native, inference=True), "cpu")


@pytest.mark.parametrize("summaries", [1, 4])
@pytest.mark.parametrize("live", [False, True])
def test_collection_fold_matches_native_cold_warm_coalesced_fork_reset_and_update(
    summaries: int, live: bool
) -> None:
    space = chunk_space()
    # Introduce the second reference only after the first complete chunk.
    raw = single(
        space, chunk_batch(space, [record(i, second=i >= 32) for i in range(65)])
    )
    store = MemoryStore(capacities(space))
    identity = PrefixIdentity("game", 0, "root")
    prefix = store.capture(raw, 0, identity)
    selected = retained(raw, [prefix])
    model = policy(space, "gru", summaries, live=live, bucketed=True)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        for _ in range(2):
            prepared = scope.prepare(selected)
            expected = stateless(model, raw)
            for actual, dense in zip(
                model.forward_distribution(prepared),
                model.forward_distribution(expected),
                strict=True,
            ):
                torch.testing.assert_close(actual, dense, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(
                model.encode_decision_memory(prepared),
                model.encode_decision_memory(expected),
                atol=2e-6,
                rtol=2e-5,
            )
        assert prepared.memory is not None and prepared.memory.fold is not None
        assert len(prepared.memory.numbers) == 1
        assert len(scope.memory.nodes) == 64
        changed = single(
            space,
            chunk_batch(
                space,
                [*(record(i, second=i >= 32) for i in range(64)), record(64, count=2)],
            ),
        )
        coalesced = store.capture(changed, 0, identity)
        for candidate, native in (
            (coalesced, changed),
            (replace(prefix, identity=PrefixIdentity("fork", 1, "branch")), raw),
            (replace(prefix, native_epoch=2), raw),
        ):
            actual = scope.prepare(retained(native, [candidate]))
            expected = stateless(model, native)
            for a, b in zip(
                model.forward_distribution(actual),
                model.forward_distribution(expected),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        with pytest.raises(RuntimeError, match="autograd"):
            with torch.enable_grad():
                model(prepared)
        next(model.parameters()).add_(0.01)
        with pytest.raises(RuntimeError, match="changed"):
            scope.prepare(selected)
        assert not scope.memory.nodes
    assert not scope.memory.nodes
    with PolicyInference(model, cache_catalog=False) as fresh, torch.no_grad():
        actual = fresh.prepare(selected)
        expected = stateless(model, raw)
        for a, b in zip(
            model.forward_distribution(actual),
            model.forward_distribution(expected),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_collection_cache_has_weak_node_ownership_and_empty_history_is_zero() -> None:
    space, raw = chunk_input()
    store = MemoryStore(capacities(space))
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    selected = retained(raw, [prefix])
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        prepared = scope.prepare(selected)
        model(prepared)
        prepared = scope.prepare(selected)  # warm Prepared also owns no raw nodes
        model(prepared)
        assert len(scope.memory.nodes) == 64
        weak = [weakref.ref(node) for node in prefix.tail.chunks()]
        del selected, prefix
        gc.collect()
        assert all(node() is None for node in weak)
        assert not scope.memory.nodes
        _, opening = chunk_input(0)
        empty = store.capture(opening, 0, PrefixIdentity("empty", 0, "root"))
        batch = scope.prepare(retained(opening, [empty]))
        assert torch.count_nonzero(model.encode_decision_memory(batch)) == 0


def test_collection_fold_preserves_shuffled_repeated_and_empty_policy_rows() -> None:
    space = chunk_space()
    store = MemoryStore(capacities(space))
    identity = PrefixIdentity("game", 0, "root")
    batches = [
        chunk_batch(space, [record(i) for i in range(length)])
        for length in (0, 32, 33, 65)
    ]
    prefixes = [store.capture(single(space, b), 0, identity) for b in batches]
    model = policy(space, "gru", 1, bucketed=True)
    with PolicyInference(model) as scope, torch.no_grad():
        model(scope.prepare(retained(single(space, batches[-1]), [prefixes[-1]])))
        order = [3, 0, 1, 3, 2]
        native = ObservationSelection(space, tuple((batches[i], 0) for i in order))
        actual = scope.prepare(retained(native, [prefixes[i] for i in order]))
        for a, b in zip(
            model.forward_distribution(actual),
            model.forward_distribution(stateless(model, native)),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_warm_collection_prepares_only_the_open_tail() -> None:
    """A cached root carries the full reference join without its ancestry."""
    space, raw = chunk_input()
    store = MemoryStore(capacities(space))
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    selected = retained(raw, [prefix])
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        model(scope.prepare(selected))
        warm = scope.prepare(selected)
        assert warm.memory is not None and warm.memory.fold is not None
        assert warm.memory.fold.node_count == 2
        assert len(warm.memory.numbers) == 1
        assert len(warm.memory.reference_numbers) == 0
        assert len(warm.memory.detail_numbers) == 0
        torch.testing.assert_close(
            model.encode_decision_memory(warm),
            model.encode_decision_memory(stateless(model, raw)),
            atol=2e-6,
            rtol=2e-5,
        )
        # Shared content under a different game lifetime cannot use this root.
        doubled = raw.select([0, 0])
        mixed = retained(doubled, [prefix, replace(prefix, native_epoch=2)])
        # A mixed cache can hold the newer root for one lifetime and its
        # predecessor for another. Rejecting the former must not expose the
        # latter to the wrong lifetime through the global root map.
        assert prefix.tail is not None and prefix.tail.previous is not None
        older = record_tail(prefix).previous.previous
        assert older is not None
        scope.memory.nodes[older] = replace(
            scope.memory.nodes[older], lifetime=(prefix.identity, 2)
        )
        assert not scope.memory.roots(mixed)
        actual = scope.prepare(mixed)
        torch.testing.assert_close(
            model.encode_decision_memory(actual),
            model.encode_decision_memory(stateless(model, doubled)),
            atol=2e-6,
            rtol=2e-5,
        )


def test_native_open_segment_reuses_records_and_preserves_issued_versions() -> None:
    """Append/coalesce work is bounded by new records, across snapshot order."""
    space = chunk_space()
    store = MemoryStore(capacities(space))
    model = policy(space, "gru", 1, live=True)
    issued = []
    sources = []
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        for length in (5, 6, 7, 31, 32, 33):
            source = single(
                space, chunk_batch(space, [record(i) for i in range(length)])
            )
            prefix = capture_stream(store, source, length)
            actual = scope.prepare(retained(source, [prefix]))
            torch.testing.assert_close(
                model.encode_decision_memory(actual),
                model.encode_decision_memory(stateless(model, source)),
                atol=2e-6,
                rtol=2e-5,
            )
            if length in (6, 7, 32, 33):
                assert len(actual.memory.numbers) == 2
            issued.append(prefix)
            sources.append(source)
        original = gather_memory(issued)
        # Revisit old snapshots after the shared segment grew/sealed.
        for prefix, source in reversed(list(zip(issued, sources, strict=True))):
            actual = scope.prepare(retained(source, [prefix]))
            torch.testing.assert_close(
                model.encode_decision_memory(actual),
                model.encode_decision_memory(stateless(model, source)),
                atol=2e-6,
                rtol=2e-5,
            )
        for key, value in gather_memory(
            restore_memory(archive_memory(issued, capacities(space)))
        ).items():
            assert (value == original[key]).all()


def test_coalesced_open_record_keeps_both_versions_and_reuses_earlier_fragments() -> (
    None
):
    space = chunk_space()
    store = MemoryStore(capacities(space))
    raw = single(space, chunk_batch(space, [record(i) for i in range(7)]))
    before = capture_stream(store, raw, 7)
    model = policy(space, "gru", 4)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        model(scope.prepare(retained(raw, [before])))
        previous = record_tail(before)
        changed = single(
            space,
            chunk_batch(space, [*(record(i) for i in range(6)), record(6, count=2)]),
        )
        after = capture_stream(store, changed, 8)
        current = record_tail(after)
        assert current is not previous and current.previous is previous.previous
        for prefix, native in ((after, changed), (before, raw), (after, changed)):
            actual = scope.prepare(retained(native, [prefix]))
            assert len(actual.memory.numbers) == 1
            torch.testing.assert_close(
                model.encode_decision_memory(actual),
                model.encode_decision_memory(stateless(model, native)),
                atol=2e-6,
                rtol=2e-5,
            )


def test_identical_native_prefixes_do_not_share_a_growing_segment() -> None:
    space = chunk_space()
    store = MemoryStore(capacities(space))
    short = chunk_batch(space, [record(i) for i in range(5)])
    first = capture_streams(
        store, ObservationSelection(space, ((short, 0), (short, 0))), [5, 5]
    )
    assert first[0].tail is first[1].tail  # immutable content is interned
    assert record_tail(first[0]) is not record_tail(first[1])
    # Both streams grow to seven records; stream one saw a different life total
    # at its sixth decision.
    grown = [
        chunk_batch(space, [record(i) for i in range(7)]),
        chunk_batch(space, [record(i, life=21 if i == 5 else 20) for i in range(7)]),
    ]
    native = ObservationSelection(space, ((grown[0], 0), (grown[1], 0)))
    prefixes = capture_streams(store, native, [7, 7])
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        for row, prefix in enumerate(prefixes):
            rows = native.select([row])
            actual = scope.prepare(retained(rows, [prefix]))
            torch.testing.assert_close(
                model.encode_decision_memory(actual),
                model.encode_decision_memory(stateless(model, rows)),
                atol=2e-6,
                rtol=2e-5,
            )
