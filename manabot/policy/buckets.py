"""Bucket selection for bucketed padding: the sizes a batch's variable axes trim to.

CSR offsets give each row’s live extent. Bucket each axis to a power of two
within its configured capacity, preserving object padding for historical means
and belief rows. Object joins are translated into the selected coordinates.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig

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
    at its declared capacity.
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
