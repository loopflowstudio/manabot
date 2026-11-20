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
from manabot.env.decision_memory import MEMORY_FIELDS, integer_words
from manabot.env.memory_store import (
    MemoryStore,
    PrefixIdentity,
    PrefixObservation,
    archive_memory,
    gather_memory,
    restore_memory,
    select_rows,
)
from manabot.env.observation import ObservationSpace
import manabot.policy.decision_memory as memory
from manabot.policy.decision_memory import MemoryFeatures, validate_retained_memory
from manabot.policy.prepare import Tables
from tests.model.test_decision_memory import _two_reference_input, policy


def chunk_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space, raw = _two_reference_input()
    space = ObservationSpace(
        space.config.model_copy(update={"max_decision_records": 96})
    )
    header = raw["decision_records"][0, 0].copy()
    members = raw["decision_members"][0, :4].copy()
    raw["decision_records"] = np.zeros((1, 96, 74), np.float32)
    for index in range(65):
        row = header.copy()
        for column in (0, 6, 8):
            row[column : column + 2] = integer_words(index)
        row[14] = index * 4
        raw["decision_records"][0, index] = row
        raw["decision_members"][0, index * 4 : index * 4 + 4] = members
    return space, raw


def test_chunk_cache_preserves_dense_features_outputs_and_gradients() -> None:
    space, raw = chunk_input()
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    dense = {key: np.repeat(value, 2, axis=0) for key, value in raw.items()}
    selected = PrefixObservation(dense, [prefix, prefix], capacities)
    model = policy(space, "gru", 1, bucketed=True)
    expected = model.prepare(dense, inference=True)
    actual = model.prepare(selected, inference=True)
    for name, values in array_fields(expected, np.ndarray).items():
        if name.startswith("/memory/"):
            continue  # causal nodes are shared, while dense prefixes repeat them
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
        validate_retained_memory(selected, changed, 0, False)
    assert store.measure([prefix])["feature_cache_bytes"] > 0
    nodes = [weakref.ref(node) for node in prefix.tail.chunks()]
    del prefix, selected
    gc.collect()
    assert all(node() is None for node in nodes)
    assert not store._tails and not store._chunks


def test_chunk_boundaries_coalescing_companions_and_archive() -> None:
    space, raw = chunk_input()
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    before = store.capture(raw, 0, identity)
    raw["decision_records"][0, 64, 3] = 2
    raw["decision_records"][0, 64, 9] = 65
    after = store.capture(raw, 0, identity)
    assert before.tail.previous is after.tail.previous
    assert before.tail is not after.tail
    # A changed first chunk cannot reuse the successor's boundary certificate.
    raw["decision_records"][0, 31, 3] = 2
    raw["decision_records"][0, 31, 9] = 32
    with pytest.raises(ValueError, match="span"):
        store.capture(raw, 0, identity)
    raw["decision_records"][0, 31, 3] = 1
    raw["decision_records"][0, 31, 9] = 31
    # Members in later chunks may refer backwards, never to unowned snapshots.
    raw["decision_members"][0, 128, 0] = 3
    with pytest.raises(ValueError, match="join"):
        store.capture(raw, 0, identity)
    archive = archive_memory([after, before, after], capacities)
    restored = gather_memory(restore_memory(archive))
    expected = gather_memory([after, before, after])
    for key in MEMORY_FIELDS:
        np.testing.assert_array_equal(restored[key], expected[key])
    # Same references with a different detail program get a different key.
    raw["decision_members"][0, 128, 0] = 1
    raw["decision_reference_details"][0, 0, 3] += 1
    changed = store.capture(raw, 0, identity)
    assert changed.tail.previous.previous is not after.tail.previous.previous


def test_selected_dense_export_retains_edits_and_rejects_corruption() -> None:
    space, raw = chunk_input()
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    exported = PrefixObservation(raw, [prefix], capacities).materialize()
    model = policy(space, "gru", 1, bucketed=True)
    # Warm the immutable path before changing its explicit dense export.
    model.prepare(PrefixObservation(raw, [prefix], capacities))
    exported["decision_records"][0, 0, 21] += 1
    selected = select_rows(exported, np.array([0, 0]))
    assert (
        selected["decision_records"][0, 0, 21] == exported["decision_records"][0, 0, 21]
    )
    expected = model.prepare(exported)
    actual = model.prepare(select_rows(exported, np.array([0])))
    for name, value in array_fields(expected, np.ndarray).items():
        np.testing.assert_array_equal(array_fields(actual, np.ndarray)[name], value)
    exported["decision_records"][0, 32, 1] = 0
    with pytest.raises(ValueError, match="ordinal"):
        model.prepare(select_rows(exported, np.array([0])))


@pytest.mark.parametrize("encoder", ["gru", "transformer"])
def test_record_features_do_not_depend_on_selected_prefix_length(
    encoder: Literal["gru", "transformer"],
) -> None:
    space, raw = chunk_input()
    short = {key: value.copy() for key, value in raw.items()}
    short["decision_records"][:, 32:] = 0
    short["decision_members"][:, 32 * 4 :] = 0
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
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    earlier = store.capture(short, 0, identity)
    later = store.capture(raw, 0, identity)
    assert earlier.tail is later.tail.previous.previous
    for prefix, dense in ((earlier, before), (later, after)):
        retained = model.prepare(
            PrefixObservation(raw, [prefix], capacities), inference=True
        ).memory
        assert retained is not None
        for field in fields(MemoryFeatures):
            np.testing.assert_array_equal(
                getattr(retained, field.name),
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
def test_changed_native_chunk_still_rejects_invalid_integer_categories(
    field: str, column: int, low: int, high: int
) -> None:
    space, raw = _two_reference_input()
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    # Native counts describe the live slices, independently of category values.
    counts = (
        int(raw["decision_records"][0, :, 19].sum()),
        int(raw["decision_reference_details"][0, :, 5].sum()),
        int(raw["decision_references"][0, :, 12].sum()),
        int(raw["decision_records"][0, :, 15:18].sum()),
    )
    original = float(raw[field][0, 0, column])
    for invalid in (low - 1, high + 1, low + 0.5, float("nan"), float("inf")):
        raw[field][0, 0, column] = original
        store = MemoryStore(capacities)
        identity = PrefixIdentity("game", 0, "root")
        store.capture_live(raw, 0, identity, epoch=1, counts=counts, tail_version=1)
        raw[field][0, 0, column] = invalid
        with pytest.raises(ValueError):
            store.capture_live(raw, 0, identity, epoch=1, counts=counts, tail_version=2)


def test_learner_assembles_collected_record_features_without_featurizing_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space, raw = chunk_input()
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    details = int((raw[MEMORY_FIELDS[1]][0, :, 5] == 1).sum())
    prefix = store.capture_live(
        raw, 0, identity, epoch=1, counts=(65, details, 2, 260), tail_version=65
    )
    obs = PrefixObservation(raw, [prefix], capacities)
    model = policy(space, "gru", 1)
    with PolicyInference(model, cache_catalog=False) as scope, torch.no_grad():
        collected = model.encode_decision_memory(scope.prepare(obs))
    original = memory._prepare_dense_memory

    def require_cached_records(
        buffers: dict[str, np.ndarray], tables: Tables
    ) -> memory.MemoryRows[np.ndarray] | None:
        assert buffers["decision_records"].shape[1] == 0
        return original(buffers, tables)

    monkeypatch.setattr(memory, "_prepare_dense_memory", require_cached_records)
    learned = model.encode_decision_memory(on_device(model.prepare(obs), "cpu"))
    torch.testing.assert_close(learned, collected, atol=2e-6, rtol=2e-5)
    assert learned.requires_grad

    # A bootstrap prefix can grow without being passed through collection.
    # Learning must featurize only that new record, not its partial raw chunk.
    raw["decision_records"][0, 65] = raw["decision_records"][0, 64]
    for column in (0, 6, 8):
        raw["decision_records"][0, 65, column : column + 2] = integer_words(65)
    raw["decision_records"][0, 65, 14] = 260
    raw["decision_members"][0, 260:264] = raw["decision_members"][0, 256:260]
    appended = store.capture_live(
        raw, 0, identity, epoch=1, counts=(66, details, 2, 264), tail_version=66
    )
    seen: list[int] = []

    def count_new_records(
        buffers: dict[str, np.ndarray], tables: Tables
    ) -> memory.MemoryRows[np.ndarray] | None:
        seen.append(buffers["decision_records"].shape[1])
        return original(buffers, tables)

    monkeypatch.setattr(memory, "_prepare_dense_memory", count_new_records)
    model.prepare(PrefixObservation(raw, [appended], capacities))
    assert sum(seen) == 1
