"""Execution-chunk sharing preserves the full-prefix policy and its gradients."""

import gc
import weakref

import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.env.memory_store import MemoryStore, PrefixIdentity, RetainedMemory
from manabot.policy.spans import ObservationSelection
from tests.model.test_decision_memory import policy
from tests.training.memory_chunks import (
    capacities,
    chunk_batch,
    chunk_input,
    chunk_space,
    record,
)


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("summaries", [1, 4])
def test_learner_fold_matches_stateless_policy_and_gradients(
    live: bool, summaries: int
) -> None:
    space = chunk_space()
    caps = capacities(space)
    store = MemoryStore(caps)
    identity = PrefixIdentity("game", 0, "root")

    def log(length: int, **changes: int) -> list:
        # Reference two first appears in the second chunk. Later records mix it
        # with reference one, exercising both local and predecessor-owned joins.
        records = [record(i, second=i >= 32, **changes) for i in range(length)]
        return records

    variants = [log(length) for length in (0, 31, 32, 33, 65)]
    # A coalesced tail and another root exercise branches, backwards reference
    # joins and nodes whose exact predecessor differs despite matching row bytes.
    coalesced = [*log(64), record(64, count=2)]
    variants.append(coalesced)
    forked = [*log(64, program=2), record(64, count=2, program=2)]
    variants.append(forked)
    batches = [chunk_batch(space, records) for records in variants]
    prefixes = [
        store.capture(
            ObservationSelection(space, ((batch, 0),)),
            0,
            PrefixIdentity("fork", 1, "fork") if index == 6 else identity,
        )
        for index, batch in enumerate(batches)
    ]
    order = (4, 0, 2, 5, 4, 1, 6, 3)
    prefixes = [prefixes[i] for i in order]
    native = ObservationSelection(space, tuple((batches[i], 0) for i in order))
    retained = ObservationSelection(
        space, native.coordinates, RetainedMemory(tuple(prefixes), caps)
    )
    model = policy(space, "gru", summaries, live=live, bucketed=True)
    expected = model.prepare(native)
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
    space, raw = chunk_input(0)
    caps = capacities(space)
    store = MemoryStore(caps)
    prefix = store.capture(raw, 0, PrefixIdentity("empty", 0, "root"))
    retained = ObservationSelection(
        space, raw.coordinates, RetainedMemory((prefix,), caps)
    )
    for encoder in ("gru", "transformer"):
        model = policy(space, encoder, 1)
        prepared = on_device(model.prepare(retained), "cpu")
        assert torch.count_nonzero(model.encode_decision_memory(prepared)) == 0
        assert prepared.memory is not None
        assert (prepared.memory.fold is not None) == (encoder == "gru")
