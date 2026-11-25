"""Record chunks preserve global joins, exact learning and bounded ownership."""

from dataclasses import fields, replace
import gc
from typing import Literal
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.policy import PolicyInference
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import (
    MemoryStore,
    PrefixIdentity,
    RetainedMemory,
    archive_memory,
    gather_memory,
    restore_memory,
)
import manabot.policy.decision_memory as memory
from manabot.policy.decision_memory import MemoryFeatures, validate_retained_memory
from manabot.policy.prepare import Tables
from manabot.policy.spans import ObservationSelection
from tests.fixtures.memory_rows import memory_rows, retained
from tests.model.test_decision_memory import _two_reference_input, policy
from tests.training.memory_chunks import (
    capacities,
    capture_stream,
    chunk_batch,
    chunk_input,
    record,
)


def with_prefixes(
    selection: ObservationSelection, prefixes: list, space_capacities: tuple[int, ...]
) -> ObservationSelection:
    return ObservationSelection(
        selection.space,
        selection.coordinates,
        RetainedMemory(tuple(prefixes), space_capacities),
    )


def test_chunk_cache_preserves_native_features_outputs_and_gradients() -> None:
    space, raw = chunk_input()
    caps = capacities(space)
    store = MemoryStore(caps)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    native = raw.select(np.array([0, 0]))
    selected = with_prefixes(native, [prefix, prefix], caps)
    model = policy(space, "gru", 1, bucketed=True)
    expected = model.prepare(native, inference=True)
    actual = model.prepare(selected, inference=True)
    for name, values in array_fields(expected, np.ndarray).items():
        if name.startswith("/memory/"):
            continue  # causal nodes are shared, while native rows repeat them
        np.testing.assert_array_equal(
            array_fields(actual, np.ndarray)[name], values, err_msg=name
        )
    outputs = []
    gradients = []
    for prepared in (expected, actual):
        model.zero_grad(set_to_none=True)
        logits, value = model.forward_distribution(on_device(prepared, "cpu"))
        (logits.sum() + value.sum()).backward()
        outputs.append((logits.detach(), value.detach()))
        gradients.append(
            {
                name: p.grad.clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
        )
    for a, b in zip(*outputs, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(
            gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-4
        )
    # Cache warmup preserves results; a different vocabulary must be admitted.
    warm = model.prepare(selected, inference=True)
    for name, values in array_fields(actual, np.ndarray).items():
        np.testing.assert_array_equal(array_fields(warm, np.ndarray)[name], values)
    tables = model._admission_spec().tables
    changed = replace(tables, name_definitions=np.empty(0, np.int64))
    with pytest.raises(ValueError, match="name row"):
        validate_retained_memory(selected.memory, changed, 0, False)
    assert store.measure([prefix])["feature_cache_bytes"] > 0
    nodes = [weakref.ref(node) for node in prefix.tail.chunks()]
    del prefix, selected
    gc.collect()
    assert all(node() is None for node in nodes)
    assert not store._tails and not store._chunks


def test_chunk_boundaries_coalescing_companions_and_archive() -> None:
    space, raw = chunk_input()
    caps = capacities(space)
    store = MemoryStore(caps)
    identity = PrefixIdentity("game", 0, "root")
    before = store.capture(raw, 0, identity)
    # The open record coalesces a repeated decision.
    coalesced = chunk_batch(
        space, [*(record(i) for i in range(64)), record(64, count=2)]
    )
    after = store.capture(ObservationSelection(space, ((coalesced, 0),)), 0, identity)
    assert before.tail.previous is after.tail.previous
    assert before.tail is not after.tail
    rows = memory_rows(space, raw.batch)
    # A changed first chunk cannot reuse the successor's boundary certificate.
    rows["decision_records"][31, 3] = 2
    rows["decision_records"][31, 9] = 32
    with pytest.raises(ValueError, match="span"):
        retained(space, raw.coordinates, [rows])
    rows["decision_records"][31, 3] = 1
    rows["decision_records"][31, 9] = 31
    # Members in later chunks may refer backwards, never to unowned snapshots.
    rows["decision_members"][128, 0] = 3
    with pytest.raises(ValueError, match="join"):
        retained(space, raw.coordinates, [rows])
    archive = archive_memory([after, before, after], caps)
    restored = gather_memory(restore_memory(archive))
    expected = gather_memory([after, before, after])
    for key in MEMORY_FIELDS:
        np.testing.assert_array_equal(restored[key], expected[key])
    # Same references with a different detail program get a different key.
    rows["decision_members"][128, 0] = 1
    [unchanged] = retained(space, raw.coordinates, [rows]).memory.prefixes
    rows["decision_reference_details"][0, 3] += 1
    [changed] = retained(space, raw.coordinates, [rows]).memory.prefixes
    assert unchanged.tail.previous.previous.key == after.tail.previous.previous.key
    assert changed.tail.previous.previous.key != after.tail.previous.previous.key


@pytest.mark.parametrize("encoder", ["gru", "transformer"])
def test_record_features_do_not_depend_on_selected_prefix_length(
    encoder: Literal["gru", "transformer"],
) -> None:
    space, raw = chunk_input()
    _, short = chunk_input(32)
    model = policy(space, encoder, 1, bucketed=True)
    before = model.prepare(short).memory
    after = model.prepare(raw).memory
    assert before is not None and after is not None
    # Appending history changes sequence coordinates, never an earlier record's
    # embedding input. The GRU's ordering and the transformer's position signal
    # are sequence-level operations.
    np.testing.assert_array_equal(before.numbers, after.numbers[:32])
    assert before.numbers.shape[1] == 38
    assert before.lengths == (32,) and after.lengths == (65,)
    caps = capacities(space)
    store = MemoryStore(caps)
    identity = PrefixIdentity("game", 0, "root")
    earlier = store.capture(short, 0, identity)
    later = store.capture(raw, 0, identity)
    assert earlier.tail is later.tail.previous.previous
    for prefix, dense in ((earlier, before), (later, after)):
        kept = model.prepare(with_prefixes(raw, [prefix], caps), inference=True).memory
        assert kept is not None
        for field in fields(MemoryFeatures):
            np.testing.assert_array_equal(
                getattr(kept, field.name),
                getattr(dense, field.name),
                err_msg=field.name,
            )


@pytest.mark.parametrize(
    "field,column,low,high",
    [
        ("decision_records", 10, 0, 4),
        ("decision_records", 11, 0, 11),
        ("decision_records", 18, 1, 2),
        *(("decision_records", c, 0, 1) for c in (60, 63, 66, 69, 72, 73)),
        ("decision_references", 0, 0, 2),
        *(("decision_references", c, 0, 1) for c in (6, 7, 8, 9, 13)),
    ],
)
def test_changed_chunk_still_rejects_invalid_integer_categories(
    field: str, column: int, low: int, high: int
) -> None:
    space, batch, rows = _two_reference_input()
    coordinates = ((batch, 0),)
    original = float(rows[field][0, column])
    retained(space, coordinates, [rows])
    for invalid in (low - 1, high + 1, low + 0.5, float("nan"), float("inf")):
        rows[field][0, column] = invalid
        with pytest.raises(ValueError):
            retained(space, coordinates, [rows])
        rows[field][0, column] = original


def test_learner_assembles_collected_record_features_without_featurizing_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space, raw = chunk_input()
    caps = capacities(space)
    store = MemoryStore(caps)
    prefix = capture_stream(store, raw, 65)
    obs = with_prefixes(raw, [prefix], caps)
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        collected = model.encode_decision_memory(scope.prepare(obs))
    original = memory._prepare_memory_rows

    def require_cached_records(
        buffers: dict[str, np.ndarray] | None, tables: Tables, **kwargs: object
    ) -> memory.MemoryRows[np.ndarray] | None:
        assert buffers is not None and buffers["decision_records"].shape[1] == 0
        return original(buffers, tables, **kwargs)

    monkeypatch.setattr(memory, "_prepare_memory_rows", require_cached_records)
    learned = model.encode_decision_memory(on_device(model.prepare(obs), "cpu"))
    torch.testing.assert_close(learned, collected, atol=2e-6, rtol=2e-5)
    assert learned.requires_grad

    # A bootstrap prefix can grow without being passed through collection.
    # Learning must featurize only that new record, not its partial raw chunk.
    _, grown = chunk_input(66)
    appended = capture_stream(store, grown, 66)
    seen: list[int] = []

    def count_new_records(
        buffers: dict[str, np.ndarray] | None, tables: Tables, **kwargs: object
    ) -> memory.MemoryRows[np.ndarray] | None:
        assert buffers is not None
        seen.append(buffers["decision_records"].shape[1])
        return original(buffers, tables, **kwargs)

    monkeypatch.setattr(memory, "_prepare_memory_rows", count_new_records)
    model.prepare(with_prefixes(grown, [appended], caps))
    assert sum(seen) == 1
