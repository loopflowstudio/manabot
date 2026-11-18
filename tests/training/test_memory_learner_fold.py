"""Execution-chunk sharing preserves the full-prefix policy and its gradients."""

import gc
import weakref

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import (
    MemoryStore,
    PrefixIdentity,
    PrefixObservation,
    gather_memory,
)
from tests.model.test_decision_memory import policy
from tests.training.test_memory_chunk_cache import chunk_input


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("summaries", [1, 4])
def test_learner_fold_matches_stateless_policy_and_gradients(
    live: bool, summaries: int
) -> None:
    space, raw = chunk_input()
    # Reference two first appears in the second chunk. Later records mix it
    # with reference one, exercising both local and predecessor-owned joins.
    raw["decision_members"][0, : 32 * 4, 0] = 1
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    identity = PrefixIdentity("game", 0, "root")
    prefixes = []
    for length in (0, 31, 32, 33, 65):
        selected = {key: value.copy() for key, value in raw.items()}
        selected["decision_records"][:, length:] = 0
        selected["decision_members"][:, length * 4 :] = 0
        if length <= 32:
            selected["decision_references"][:, 1:] = 0
            selected["decision_reference_details"][:, 8:] = 0
        if not length:
            for key in MEMORY_FIELDS:
                selected[key][:] = 0
        prefixes.append(store.capture(selected, 0, identity))
    # A coalesced tail and another root exercise branches, backwards reference
    # joins and nodes whose exact predecessor differs despite matching row bytes.
    changed = {key: value.copy() for key, value in raw.items()}
    changed["decision_records"][0, 64, 3] = 2
    changed["decision_records"][0, 64, 9] = 65
    prefixes.append(store.capture(changed, 0, identity))
    changed["decision_reference_details"][0, 0, 3] += 1
    prefixes.append(store.capture(changed, 0, PrefixIdentity("fork", 1, "fork")))
    prefixes = [prefixes[i] for i in (4, 0, 2, 5, 4, 1, 6, 3)]
    dense = {key: np.repeat(value, len(prefixes), axis=0) for key, value in raw.items()}
    dense.update(gather_memory(prefixes))
    retained = PrefixObservation(dense, prefixes, capacities)
    model = policy(space, "gru", summaries, live=live, bucketed=True)
    expected = model.prepare(retained.materialize())
    actual = model.prepare(retained)
    assert actual.memory is not None and actual.memory.fold is not None
    assert expected.memory is not None and expected.memory.fold is None
    assert len(actual.memory.numbers) < len(expected.memory.numbers)
    assert sum(sum(level.schedule.sizes) for level in actual.memory.fold.levels) == len(
        actual.memory.numbers
    )
    for _ in range(2):
        outputs = []
        gradients = []
        for prepared in (expected, actual):
            model.zero_grad(set_to_none=True)
            logits, values = model.forward_distribution(on_device(prepared, "cpu"))
            # Unequal row contributions detect accidental tail sorting, duplicate
            # suppression or averaging instead of summed shared gradients.
            weights = torch.arange(1, len(prefixes) + 1)[:, None]
            (logits.mul(weights).sum() + values.mul(weights).sum()).backward()
            outputs.append((logits.detach(), values.detach()))
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
                gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-4, msg=name
            )
        assert any(
            g.abs().sum() > 0
            for name, g in gradients[1].items()
            if "decision_memory.gru" in name
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-5)
    # Prepared owns array routes, never native prefixes or a persistent graph.
    nodes = [weakref.ref(node) for p in prefixes if p.tail for node in p.tail.chunks()]
    del prefixes, retained
    gc.collect()
    assert all(node() is None for node in nodes)


def test_fold_empty_history_is_zero_and_transformer_stays_stateless() -> None:
    space, raw = chunk_input()
    for key in MEMORY_FIELDS:
        raw[key][:] = 0
    capacities = tuple(space.encoder.shapes[key][0] for key in MEMORY_FIELDS)
    store = MemoryStore(capacities)
    prefix = store.capture(raw, 0, PrefixIdentity("empty", 0, "root"))
    retained = PrefixObservation(raw, [prefix], capacities)
    for encoder in ("gru", "transformer"):
        model = policy(space, encoder, 1)
        prepared = on_device(model.prepare(retained), "cpu")
        assert torch.count_nonzero(model.encode_decision_memory(prepared)) == 0
        assert prepared.memory is not None
        assert (prepared.memory.fold is not None) == (encoder == "gru")
