"""Record chunks preserve global joins, exact learning and bounded ownership."""

from dataclasses import replace
import gc
from typing import Literal
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields, on_device
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
from manabot.policy.decision_memory import validate_retained_memory
from tests.model.test_decision_memory import _two_reference_input, policy


def chunk_input() -> tuple[ObservationSpace, dict[str, np.ndarray]]:
    space, raw = _two_reference_input()
    space = ObservationSpace(
        space.encoder.config.model_copy(update={"max_decision_records": 96})
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
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("game", 0, "root"))
    dense = {key: np.repeat(value, 2, axis=0) for key, value in raw.items()}
    selected = PrefixObservation(dense, [prefix, prefix], capacities)
    model = policy(space, "gru", 1, bucketed=True)
    expected = model.prepare(dense, inference=True)
    actual = model.prepare(selected, inference=True)
    for name, values in array_fields(expected, np.ndarray).items():
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
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(
            gradients[0][name], gradients[1][name], atol=0, rtol=0
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
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
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
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
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
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
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
        for name, expected in array_fields(dense, np.ndarray).items():
            np.testing.assert_array_equal(
                array_fields(retained, np.ndarray)[name], expected, err_msg=name
            )
