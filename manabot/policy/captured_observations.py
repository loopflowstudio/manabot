"""Persist policy calls in backend captures, including their interned viewer histories."""

from collections.abc import Mapping
from dataclasses import asdict
import json

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import (
    MemoryArchive,
    PrefixIdentity,
    RetainedMemory,
    archive_memory,
    restore_memory,
)
from manabot.env.observation import ObservationSpace
from manabot.policy.spans import ObservationSelection
import managym


def pack_observation(
    observation: ObservationSelection | managym.ObservationBatch,
) -> dict[str, np.ndarray]:
    memory = None
    if isinstance(observation, ObservationSelection):
        batch = observation.batch
        memory = observation.memory
    else:
        batch = observation
    arrays = {"batch": np.frombuffer(batch.to_bytes(), dtype=np.uint8)}
    if memory is not None:
        archive = archive_memory(memory.prefixes, memory.capacities)
        arrays["memory"] = np.array(
            json.dumps(
                {
                    "keys": archive.keys,
                    "identities": [asdict(identity) for identity in archive.identities],
                }
            )
        )
        arrays["memory/offsets"] = archive.offsets
        arrays["memory/indexes"] = archive.indexes
        for key, chunk in zip(archive.keys, archive.chunks, strict=True):
            for field, rows in zip(MEMORY_FIELDS, chunk, strict=True):
                arrays[f"memory/{key}/{field}"] = rows
    return arrays


def unpack_observation(
    arrays: Mapping[str, np.ndarray], space: ObservationSpace
) -> ObservationSelection:
    if "batch" not in arrays:
        raise ValueError(
            "dense observation capture requires the revision that wrote it"
        )
    data = arrays["batch"]
    if data.dtype != np.uint8 or data.ndim != 1:
        raise ValueError("expected observation batch bytes")
    batch = managym.ObservationBatch.from_bytes(data.tobytes())
    if batch.layout.shapes != space.shapes:
        raise ValueError("captured batch layout differs from policy")
    memory = None
    if "memory" in arrays:
        metadata = json.loads(str(arrays["memory"].item()))
        identities = tuple(PrefixIdentity(**row) for row in metadata["identities"])
        if len(identities) != batch.batch_size or not all(
            k in space.shapes for k in MEMORY_FIELDS
        ):
            raise ValueError("captured decision memory differs from batch")
        keys = tuple(metadata["keys"])
        capacities = (
            space.config.max_decision_records,
            space.config.max_decision_reference_details,
            space.config.max_decision_references,
            space.config.max_decision_members,
        )
        archive = MemoryArchive(
            capacities,
            identities,
            keys,
            tuple(
                tuple(arrays[f"memory/{key}/{field}"] for field in MEMORY_FIELDS)
                for key in keys
            ),
            arrays["memory/offsets"],
            arrays["memory/indexes"],
        )
        memory = RetainedMemory(restore_memory(archive), capacities)
    return ObservationSelection(
        space, tuple((batch, i) for i in range(batch.batch_size)), memory
    )
