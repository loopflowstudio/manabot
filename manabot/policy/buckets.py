"""Bucket selection for bucketed padding: the sizes a batch's variable axes trim to.

Every observation buffer arrives padded to its declared capacity, yet real
batches use little of it. With `ActorCritic(..., bucketed_shapes=True)`, set
by a training stage's `Execution.bucketed_shapes`, the policy trims each
variable axis to a bucket: the next power of two at or above the batch's need,
at least `MIN_BUCKET` and at most the declared capacity. A forward then takes
one of a few shapes per axis. The setting changes no `ModelConfig` field,
parameter, buffer, checkpoint or policy input.

`row_needs` measures what each row needs on every axis; `sizes_of` buckets a
group of rows from those needs (`bucket_sizes` does both for one batch).
A learner measures each rollout row once and regroups the needs for every
epoch's minibatches. Preparation reads original-coordinate bucket views and
uses `object_map` to translate the integer joins it emits.

An axis's need covers every row the default path could read or reject: the
last row with any nonzero value (so poisoned padding is kept and still
rejected) and every row a reference can name (effect CSR spans, frame ids,
action continuation joins, object references). Trimming therefore removes only
all-zero rows nothing refers to, so admission outcomes and messages are
unchanged and outputs differ only by floating-point summation order.

Object axes (cards, permanents and the input v11 stack) move object
coordinates. They are trimmed only when padding objects are inert: not under
`historical_mean`, whose average includes padding, and not with belief rows,
which follow the objects. The deprecated v9/v10 layouts have no stack: its
capacity is zero and its keys are absent.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from manabot.env.memory_store import PrefixObservation
from manabot.env.observation import ObservationBuffers, ObservationSpace, width
from manabot.infra.config import ModelConfig
from manabot.policy.decision_memory import retained_memory

MIN_BUCKET = 8

# Fixed column order of retained rollout extents, including absent legacy axes.
AXES = (
    "facts",
    "effects",
    "details",
    "members",
    "relations",
    "actions",
    "events",
    "cards",
    "known",
    "decklists",
    "permanents",
    "stack",
)
_EVENT_WIDTH = width("events")


def bucket(need: int, capacity: int) -> int:
    """The next power of two at or above `need`, within [MIN_BUCKET, capacity]."""
    return min(capacity, max(MIN_BUCKET, 1 << max(need - 1, 0).bit_length()))


@dataclass(frozen=True)
class BucketSizes:
    """Row counts per variable axis for one batch, and the declared capacities."""

    sizes: dict[str, int]
    capacities: dict[str, int]

    @property
    def trims_objects(self) -> bool:
        return any(self.sizes[axis] < self.capacities[axis] for axis in OBJECT_AXES)


@dataclass(frozen=True)
class BucketedObservation:
    """Views in original object coordinates, with the extents used to select them."""

    rows: dict[str, np.ndarray]
    buckets: BucketSizes
    needs: np.ndarray


def has_history_events(obs: Mapping[str, object]) -> bool:
    """Whether `events` holds policy-history rows, whose contexts name objects."""
    events = obs.get("events")
    shape = getattr(events, "shape", None)
    return shape is not None and len(shape) == 3 and shape[-1] == _EVENT_WIDTH


def object_trimming(config: ModelConfig) -> bool:
    """Object padding is inert for masked aggregation without belief rows."""
    return (
        config.value_aggregation != "historical_mean"
        and config.belief_count_buckets == 0
    )


def object_segments(capacities: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """(axis, start, rows) of each object segment after the two player rows.

    The object table is players, agent cards, opponent cards, agent
    permanents, opponent permanents, then (input v11) the stack, each segment
    at its declared capacity; a v9/v10 stack has none.
    """
    segments: list[tuple[str, int, int]] = []
    start = 2
    for axis in ("cards", "cards", "permanents", "permanents", "stack"):
        rows = capacities.get(axis, 0)
        segments.append((axis, start, rows))
        start += rows
    return segments


# Axes whose rows are object-table segments.
OBJECT_AXES: tuple[str, ...] = ("cards", "permanents", "stack")


def row_needs(obs: ObservationBuffers, space: ObservationSpace) -> np.ndarray | None:
    """Rows each batch row needs on every variable axis, [B, len(AXES)] int64.

    A batch's need on an axis is the maximum over its rows, so needs measured
    once per row serve any later grouping of those rows (`sizes_of`). Object
    axes always count the rows references name, so the same needs serve a
    policy with or without object trimming. None for a batch whose keys or
    shapes differ from the declared layout: it is not trimmed, so the model
    rejects it exactly as it would without buckets.
    """
    extents = space.layout.axis_extents(
        obs, omit_decision_memory=isinstance(obs, PrefixObservation)
    )
    if extents is None:
        return None
    zeros = np.zeros_like(extents["actions"])
    return np.stack([extents.get(axis, zeros) for axis in AXES], axis=1)


def sizes_of(
    needs: np.ndarray, space: ObservationSpace, *, objects: bool
) -> BucketSizes:
    """The buckets of a batch whose rows need `needs` [B, len(AXES)]."""
    capacities = dict.fromkeys(AXES, 0)
    for axis, buffers in space.axis_buffers.items():
        key, dimension = buffers[0]
        capacities[axis] = space.shapes[key][dimension]
    most = needs.max(0, initial=0)
    sizes = {
        axis: bucket(int(most[column]), capacities[axis])
        for column, axis in enumerate(AXES)
    }
    if not objects:
        for axis in OBJECT_AXES:
            sizes[axis] = capacities[axis]
    return BucketSizes(sizes, capacities)


def bucket_sizes(
    obs: ObservationBuffers, space: ObservationSpace, *, objects: bool
) -> BucketSizes | None:
    """Measure one dense batch's buckets; None leaves it untouched (`row_needs`).

    Dense extent helper (`axis_extents`); removed in 0.3.0.
    """
    needs = row_needs(obs, space)
    return None if needs is None else sizes_of(needs, space, objects=objects)


def bucket_observation(
    obs: ObservationBuffers,
    space: ObservationSpace,
    needs: np.ndarray | None = None,
    *,
    objects: bool,
) -> BucketedObservation | None:
    """Select dense views in original coordinates using measured or supplied row extents.

    Dense dict bucketing (`BucketedObservation` / `row_needs`); removed in 0.3.0.
    """
    if needs is None:
        needs = row_needs(obs, space)
    if needs is None:
        return None
    buckets = sizes_of(needs, space, objects=objects)
    memory = retained_memory(obs)
    trimmed = (
        PrefixObservation(obs, memory.prefixes, memory.capacities)
        if memory is not None
        else dict(obs)
    )
    for axis, buffers in space.axis_buffers.items():
        stop = buckets.sizes[axis]
        for key, dimension in buffers:
            prefix = (slice(None),) * (dimension + 1) + (slice(stop),)
            trimmed[key] = obs[key][prefix]
    return BucketedObservation(trimmed, buckets, needs)


def object_map(buckets: BucketSizes) -> np.ndarray | None:
    """Translate original zero-based object slots into bucket coordinates."""
    if not buckets.trims_objects:
        return None
    segments = object_segments(buckets.capacities)
    _, start, count = segments[-1]
    indexes = np.arange(start + count, dtype=np.int64)
    removed = 0
    for axis, start, count in segments:
        indexes[start : start + count] -= removed
        removed += count - buckets.sizes[axis]
    return indexes
