"""Edited decision-memory rows, admitted the way an archived prefix is restored.

Tests construct an unusual or malformed log by editing a projected memory table,
then retain it through `restore_memory`, the entry point for memory rows that
did not come from the live native encoder. Its chunk admission is the check
under test; nothing here relaxes it.
"""

from collections.abc import Mapping, Sequence
from hashlib import sha256

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import (
    MemoryArchive,
    PrefixIdentity,
    RetainedMemory,
    restore_memory,
)
from manabot.env.observation import ObservationSpace
from manabot.policy.decision_memory import live_memory, record_chunks
from manabot.policy.spans import ObservationSelection
import managym

type MemoryRows = dict[str, np.ndarray]


def memory_rows(space: ObservationSpace, batch: managym.ObservationBatch) -> MemoryRows:
    """Editable padded memory tables of the batch's first row, without a batch axis."""
    selection = ObservationSelection(space, ((batch, 0),))
    return {key: selection.project(key)[0].copy() for key in MEMORY_FIELDS}


def _chunk_key(rows: tuple[np.ndarray, ...]) -> str:
    digest = sha256()
    for block in rows:
        digest.update(str(block.shape).encode())
        digest.update(block.tobytes())
    return digest.hexdigest()


def retained(
    space: ObservationSpace,
    rows: Sequence[tuple[managym.ObservationBatch, int]],
    memories: Sequence[Mapping[str, np.ndarray]],
    *,
    capacities: tuple[int, ...] | None = None,
) -> ObservationSelection:
    """`rows` with `memories` (padded tables, one per row) as restored prefixes."""
    keys: list[str] = []
    chunks: list[tuple[np.ndarray, ...]] = []
    offsets = [0]
    indexes: list[int] = []
    for memory in memories:
        [live] = live_memory({key: memory[key][None] for key in MEMORY_FIELDS})
        for blocks in record_chunks(live):
            chunk = tuple(np.ascontiguousarray(b, np.float32) for b in blocks)
            keys.append(_chunk_key(chunk))
            chunks.append(chunk)
            indexes.append(len(chunks) - 1)
        offsets.append(len(indexes))
    if capacities is None:
        capacities = tuple(space.shapes[key][0] for key in MEMORY_FIELDS)
    identities = tuple(
        PrefixIdentity(f"fixture-{row}", 0, "root") for row in range(len(memories))
    )
    prefixes = restore_memory(
        MemoryArchive(
            capacities,
            identities,
            tuple(keys),
            tuple(chunks),
            np.array(offsets, np.int64),
            np.array(indexes, np.int64),
        )
    )
    return ObservationSelection(
        space, tuple(rows), RetainedMemory(prefixes, capacities)
    )
