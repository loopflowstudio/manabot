"""Shared multi-branch decision-memory prefix fixture for fold tests."""

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import (
    MemoryStore,
    PrefixIdentity,
    PrefixObservation,
    gather_memory,
)
from manabot.env.observation import ObservationSpace
from tests.training.test_memory_chunk_cache import chunk_input


def memory_fold_observation() -> tuple[ObservationSpace, PrefixObservation]:
    """Build a PrefixObservation with shared prefixes, empty rows, a branch and a fork."""
    space, raw = chunk_input()
    # Reference two first appears in the second chunk. Later records mix it
    # with reference one, exercising both local and predecessor-owned joins.
    raw["decision_members"][0, : 32 * 4, 0] = 1
    capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
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
    return space, PrefixObservation(dense, prefixes, capacities)
