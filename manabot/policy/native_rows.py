"""Read the immutable v11 CSR export in the model's original object coordinates.

Each table has one span per batch row, or two viewer-relative seat spans.
Only outputs whose model shape includes padding need to materialize those slots.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.policy.buckets import AXES, BucketSizes, sizes_of
import managym


@dataclass(frozen=True)
class RowSpans:
    values: np.ndarray
    offsets: np.ndarray
    groups: tuple[int, ...]

    @cached_property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets).reshape(self.groups)

    @cached_property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Group and local slot of each packed row, in encoder order."""
        groups = np.repeat(np.arange(self.counts.size), self.counts.ravel())
        return groups, np.arange(len(groups)) - self.offsets[groups]

    def group(self, index: int) -> np.ndarray:
        return self.values[self.offsets[index] : self.offsets[index + 1]]

    def mask(self, capacity: int) -> np.ndarray:
        return np.arange(capacity) < self.counts[..., None]

    def scatter(self, values: np.ndarray, capacity: int, fill: int = 0) -> np.ndarray:
        shape = (self.counts.size, capacity, *values.shape[1:])
        out = np.full(shape, fill, dtype=values.dtype)
        out[self.coordinates] = values
        return out.reshape(*self.groups, capacity, *values.shape[1:])


@dataclass(frozen=True)
class NativeRows:
    tables: dict[str, RowSpans]
    capacities: dict[str, int]
    buckets: BucketSizes | None
    needs: np.ndarray | None

    def dense(self, key: str) -> np.ndarray:
        name = key.removesuffix("_valid")
        spans = self.tables[name]
        if name != key:
            return spans.mask(self.capacities[name]).astype(np.float32)
        return spans.scatter(
            spans.values, self.capacities[name], -1 if key == "action_focus" else 0
        )


def native_rows(
    batch: managym.ObservationBatch,
    space: ObservationSpace,
    *,
    bucketed: bool,
    objects: bool,
) -> NativeRows:
    """Bind a native batch to the exact model layout before trusting any join."""
    if (
        type(batch) is not managym.ObservationBatch
        or batch.layout.shapes != space.shapes
    ):
        raise ValueError("native observation layout differs from policy")
    tables: dict[str, RowSpans] = {}
    capacities: dict[str, int] = {}
    for key, (seat_axis, offsets, values) in batch.tables().items():
        axis = int(seat_axis)
        shape = space.shapes[key]
        capacities[key] = shape[axis]
        groups = (batch.batch_size, *shape[:axis])
        tables[key] = RowSpans(
            np.frombuffer(values, space.dtypes[key]).reshape(-1, *shape[axis + 1 :]),
            np.frombuffer(offsets, np.int32),
            groups,
        )
    needs = buckets = None
    if bucketed:
        needs = np.zeros((batch.batch_size, len(AXES)), dtype=np.int64)
        for axis, fields in space.axis_buffers.items():
            extent = needs[:, AXES.index(axis)]
            for key, _ in fields:
                if key in tables:
                    counts = tables[key].counts
                    np.maximum(
                        extent,
                        counts.max(1) if counts.ndim == 2 else counts,
                        out=extent,
                    )
        buckets = sizes_of(needs, space, objects=objects)
        for axis, fields in space.axis_buffers.items():
            for key, _ in fields:
                capacities[key] = buckets.sizes[axis]
    return NativeRows(tables, capacities, buckets, needs)
