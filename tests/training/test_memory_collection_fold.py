"""Frozen collection reuses complete nodes without changing the acting policy."""

from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import PolicyInference
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import MemoryStore, PrefixIdentity, PrefixObservation
from tests.model.test_decision_memory import policy
from tests.training.test_memory_chunk_cache import chunk_input


@pytest.mark.parametrize("summaries", [1, 4])
@pytest.mark.parametrize("live", [False, True])
def test_collection_fold_matches_dense_cold_warm_coalesced_fork_reset_and_update(
    summaries: int, live: bool
) -> None:
    space, raw = chunk_input()
    # Introduce the second reference only after the first complete chunk.
    raw["decision_members"][0, :128, 0] = 1
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    prefix = store.capture(raw, 0, identity)
    selected = PrefixObservation(raw, [prefix], capacities)
    model = policy(space, "gru", summaries, live=live, bucketed=True)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        for _ in range(2):
            prepared = scope.prepare(selected)
            expected = on_device(
                model.prepare(selected.materialize(), inference=True), "cpu"
            )
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
        assert len(scope.memory.nodes) == 2
        changed = {key: value.copy() for key, value in raw.items()}
        changed["decision_records"][0, 64, 3] = 2
        changed["decision_records"][0, 64, 9] = 65
        coalesced = store.capture(changed, 0, identity)
        for candidate in (
            coalesced,
            replace(prefix, identity=PrefixIdentity("fork", 1, "branch")),
            replace(prefix, native_epoch=2),
        ):
            obs = PrefixObservation(changed, [candidate], capacities)
            actual = scope.prepare(obs)
            expected = on_device(
                model.prepare(obs.materialize(), inference=True), "cpu"
            )
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
        expected = on_device(
            model.prepare(selected.materialize(), inference=True), "cpu"
        )
        for a, b in zip(
            model.forward_distribution(actual),
            model.forward_distribution(expected),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_collection_cache_has_weak_node_ownership_and_empty_history_is_zero() -> None:
    space, raw = chunk_input()
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    selected = PrefixObservation(raw, [prefix], capacities)
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        prepared = scope.prepare(selected)
        model(prepared)
        prepared = scope.prepare(selected)  # warm Prepared also owns no raw nodes
        model(prepared)
        assert len(scope.memory.nodes) == 2
        weak = [weakref.ref(node) for node in prefix.tail.chunks()]
        del selected, prefix
        gc.collect()
        assert all(node() is None for node in weak)
        assert not scope.memory.nodes
        for key in MEMORY_FIELDS:
            raw[key][:] = 0
        empty = store.capture(raw, 0, PrefixIdentity("empty", 0, "root"))
        batch = scope.prepare(PrefixObservation(raw, [empty], capacities))
        assert torch.count_nonzero(model.encode_decision_memory(batch)) == 0


def test_collection_fold_preserves_shuffled_repeated_and_empty_policy_rows() -> None:
    space, raw = chunk_input()
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    prefixes = []
    for length in (0, 32, 33, 65):
        selected = {key: value.copy() for key, value in raw.items()}
        selected["decision_records"][:, length:] = 0
        selected["decision_members"][:, length * 4 :] = 0
        if not length:
            for key in MEMORY_FIELDS:
                selected[key][:] = 0
        prefixes.append(store.capture(selected, 0, identity))
    model = policy(space, "gru", 1, bucketed=True)
    with PolicyInference(model) as scope, torch.no_grad():
        warm = PrefixObservation(raw, [prefixes[-1]], capacities)
        model(scope.prepare(warm))
        order = [3, 0, 1, 3, 2]
        ordinary = {
            key: np.repeat(value, len(order), axis=0) for key, value in raw.items()
        }
        batch = PrefixObservation(ordinary, [prefixes[i] for i in order], capacities)
        actual = scope.prepare(batch)
        dense = on_device(model.prepare(batch.materialize(), inference=True), "cpu")
        for a, b in zip(
            model.forward_distribution(actual),
            model.forward_distribution(dense),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_warm_collection_prepares_only_the_open_tail() -> None:
    """A cached root carries the full reference join without its ancestry."""
    space, raw = chunk_input()
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    selected = PrefixObservation(raw, [prefix], capacities)
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        model(scope.prepare(selected))
        warm = scope.prepare(selected)
        assert warm.memory is not None and warm.memory.fold is not None
        assert warm.memory.fold.node_count == 2
        assert len(warm.memory.numbers) == 1
        assert len(warm.memory.reference_numbers) == 0
        assert len(warm.memory.detail_numbers) == 0
        dense = on_device(model.prepare(selected.materialize(), inference=True), "cpu")
        torch.testing.assert_close(
            model.encode_decision_memory(warm),
            model.encode_decision_memory(dense),
            atol=2e-6,
            rtol=2e-5,
        )
        # Shared content under a different game lifetime cannot use this root.
        mixed = PrefixObservation(
            {key: np.repeat(value, 2, axis=0) for key, value in raw.items()},
            [prefix, replace(prefix, native_epoch=2)],
            capacities,
        )
        # A mixed cache can hold the newer root for one lifetime and its
        # predecessor for another. Rejecting the former must not expose the
        # latter to the wrong lifetime through the global root map.
        assert prefix.tail is not None and prefix.tail.previous is not None
        older = prefix.tail.previous.previous
        assert older is not None
        scope.memory.nodes[older] = replace(
            scope.memory.nodes[older], lifetime=(prefix.identity, 2)
        )
        assert not scope.memory.roots(mixed)
        actual = scope.prepare(mixed)
        expected = on_device(model.prepare(mixed.materialize(), inference=True), "cpu")
        torch.testing.assert_close(
            model.encode_decision_memory(actual),
            model.encode_decision_memory(expected),
            atol=2e-6,
            rtol=2e-5,
        )
