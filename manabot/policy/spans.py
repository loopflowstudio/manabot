"""Read the immutable v11 CSR export in the model's original object coordinates.

Each table has one span per batch row, or two viewer-relative seat spans.
Only outputs whose model shape includes padding need to materialize those slots.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

import numpy as np

from manabot.env.memory_store import (
    MEMORY_FIELDS,
    CompactObservation,
    PrefixObservation,
)
from manabot.env.observation import (
    FACT_KIND_INDEX,
    FACT_VALUE_INDEX,
    ObservationSpace,
)
from manabot.policy.buckets import AXES, BucketSizes, sizes_of
from manabot.policy.readings import ColumnReading, input_width
import managym


@dataclass(frozen=True)
class RecordLayout:
    dtype: np.dtype
    # Positions in the checkpoint's logical column order, including full words.
    fields: tuple[tuple[str, int], ...]
    words: tuple[int, ...]
    width: int


@dataclass(frozen=True)
class BatchLayout:
    records: dict[str, RecordLayout]
    fact_divisors: np.ndarray


@lru_cache(maxsize=32)
def batch_layout(layout: managym.ObservationLayout) -> BatchLayout:
    """Bind NumPy views to the immutable producer layout once per layout."""
    arrays = layout.arrays
    records = {}
    for key, fields in layout.csr_fields.items():
        array = arrays[key]
        width = len(array["columns"])
        indexes = {index for _, _, index in fields}
        dtype = (
            np.dtype(fields[0][1])
            if "element" in array
            else np.dtype([(name, dtype) for name, dtype, _ in fields])
        )
        records[key] = RecordLayout(
            dtype,
            tuple((name, index) for name, _, index in fields),
            tuple(i for i in sorted(indexes) if i + 1 < width and i + 1 not in indexes),
            width,
        )
    scales = layout.csr_fact_divisors
    divisors = np.ones(max(scales) + 1, np.float32)
    for kind, divisor in scales.items():
        divisors[kind] = divisor
    return BatchLayout(records, divisors)


def full_word(rows: np.ndarray, index: int) -> np.ndarray:
    """Unsigned catalog key for an already complete signed int32 field."""
    return rows[..., index].view(np.uint32).astype(np.int64)


@lru_cache(maxsize=256)
def numeric_fields(
    layout: RecordLayout, readings: tuple[ColumnReading, ...] | None
) -> tuple[tuple[str, int], ...]:
    if readings is None:
        return layout.fields
    selected = {i for i, reading in enumerate(readings) if input_width(reading)}
    return tuple(
        (name, i)
        for name, i in layout.fields
        if i in selected or (i in layout.words and i + 1 in selected)
    )


@dataclass(frozen=True)
class RowSpans:
    records: np.ndarray
    layout: RecordLayout
    offsets: np.ndarray
    groups: tuple[int, ...]

    @cached_property
    def integers(self) -> np.ndarray:
        """Integer joins, flags and whole words in logical column order."""
        if self.records.dtype.names is None:
            return self.records
        if (
            all(self.records.dtype[name] == np.int32 for name, _ in self.layout.fields)
            and not self.layout.words
        ):
            return self.records.view(np.int32).reshape(
                len(self.records), self.layout.width
            )
        out = np.zeros((len(self.records), self.layout.width), np.int32)
        for name, index in self.layout.fields:
            field = self.records[name]
            if field.dtype.kind != "f":
                out[:, index] = field
        out.flags.writeable = False
        return out

    def features(self, readings: tuple[ColumnReading, ...] | None = None) -> np.ndarray:
        """Float projection for numeric readings. Lookups use the integer fields."""
        width = self.layout.width if readings is None else len(readings)
        out = np.zeros((len(self.records), width), np.float32)
        for name, index in numeric_fields(self.layout, readings):
            field = self.records[name]
            if index in self.layout.words:
                bits = field.view(np.uint32)
                out[:, index] = bits >> 16
                if index + 1 < width:
                    out[:, index + 1] = bits & 65535
            else:
                out[:, index] = field
        return out

    @cached_property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets).reshape(self.groups)

    @cached_property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Group and local slot of each packed row, in encoder order."""
        groups = np.repeat(np.arange(self.counts.size), self.counts.ravel())
        return groups, np.arange(len(groups)) - self.offsets[groups]

    def group(self, index: int) -> np.ndarray:
        return self.integers[self.offsets[index] : self.offsets[index + 1]]

    def mask(self, capacity: int) -> np.ndarray:
        return np.arange(capacity) < self.counts[..., None]

    def scatter(self, values: np.ndarray, capacity: int, fill: int = 0) -> np.ndarray:
        shape = (self.counts.size, capacity, *values.shape[1:])
        out = np.full(shape, fill, dtype=values.dtype)
        out[self.coordinates] = values
        return out.reshape(*self.groups, capacity, *values.shape[1:])


type BatchRow = tuple[managym.ObservationBatch, int]


@dataclass(frozen=True, eq=False)
class ObservationSelection(Mapping[str, np.ndarray]):
    """Immutable native coordinates in consumer order, with optional interned memory.

    A policy reads the selected CSR spans. Mapping access projects one logical
    field as a read-only array. Explicit exports produce editable dense rows.
    """

    space: ObservationSpace
    coordinates: tuple[BatchRow, ...]
    memory: PrefixObservation | None = None

    @cached_property
    def batch(self) -> managym.ObservationBatch:
        if not self.coordinates:
            return managym.encode_observation_batch([], self.space.layout)
        source = self.coordinates[0][0]
        if source.batch_size == len(self.coordinates) and all(
            batch is source and index == i
            for i, (batch, index) in enumerate(self.coordinates)
        ):
            return source
        return source.gather(self.coordinates)

    @cached_property
    def action_counts(self) -> np.ndarray:
        """Native legal prefix lengths in consumer order, without gathering other tables."""
        sources: dict[managym.ObservationBatch, list[int]] = {}
        counts = np.empty(len(self.coordinates), np.int32)
        for i, (batch, row) in enumerate(self.coordinates):
            if batch not in sources:
                sources[batch] = batch.action_counts
            counts[i] = sources[batch][row]
        counts.flags.writeable = False
        return counts

    @cached_property
    def spans(self) -> "ObservationSpans":
        return observation_spans(self.batch, self.space, bucketed=False, objects=False)

    def __iter__(self) -> Iterator[str]:
        return (
            k
            for k in self.space.shapes
            if self.memory is None or k not in MEMORY_FIELDS
        )

    def __len__(self) -> int:
        return len(self.space.shapes) - (
            len(MEMORY_FIELDS) if self.memory is not None else 0
        )

    def __contains__(self, key: object) -> bool:
        return key in self.space.shapes and (
            self.memory is None or key not in MEMORY_FIELDS
        )

    def __getitem__(self, key: str) -> np.ndarray:
        values = self.project(key)
        values.flags.writeable = False
        return values

    def project(self, key: str) -> np.ndarray:
        """Editable logical field in configured coordinates, independent of native storage."""
        if key not in self:
            raise KeyError(key)
        if key == "actions_valid":
            return (
                np.arange(self.space.config.max_actions) < self.action_counts[:, None]
            ).astype(np.float32)
        if not self.coordinates:
            return np.empty((0, *self.space.shapes[key]), self.space.dtypes[key])
        return self.spans.dense(key)

    def select(self, indexes: np.ndarray | slice) -> "ObservationSelection":
        coords = (
            self.coordinates[indexes]
            if isinstance(indexes, slice)
            else tuple(self.coordinates[i] for i in indexes)
        )
        memory = self.memory
        if memory is not None:
            prefixes = (
                memory.prefixes[indexes]
                if isinstance(indexes, slice)
                else tuple(memory.prefixes[i] for i in indexes)
            )
            memory = PrefixObservation({}, prefixes, memory.memory_capacities)
        return ObservationSelection(self.space, coords, memory)

    def export(self) -> PrefixObservation | dict[str, np.ndarray]:
        rows = {key: self.project(key) for key in self}
        if self.memory is None:
            return rows
        return PrefixObservation(
            rows, self.memory.prefixes, self.memory.memory_capacities
        )

    def materialize(self) -> CompactObservation | dict[str, np.ndarray]:
        rows = self.export()
        return rows.materialize() if isinstance(rows, PrefixObservation) else rows

    def action_columns(
        self, rows: np.ndarray, slots: np.ndarray, columns: Sequence[int]
    ) -> np.ndarray:
        """Read selected actions in their source batches without gathering other tables."""
        by_batch: dict[managym.ObservationBatch, list[tuple[int, int, int]]] = {}
        for dest, (row, slot) in enumerate(zip(rows, slots, strict=True)):
            batch, index = self.coordinates[row]
            by_batch.setdefault(batch, []).append((dest, index, int(slot)))
        out = np.empty((len(rows), len(columns)), np.float32)
        for batch, selected in by_batch.items():
            _, offsets, values = batch.tables()["actions"]
            layout = batch_layout(batch.layout).records["actions"]
            dest, index, slot = np.asarray(selected, np.intp).T
            records = np.frombuffer(values, layout.dtype)[
                np.frombuffer(offsets, np.int32)[index] + slot
            ]
            names = {i: name for name, i in layout.fields}
            for j, col in enumerate(columns):
                out[dest, j] = records[names[col]]
        return out

    def live_memory(
        self, index: int, starts: tuple[int, ...]
    ) -> tuple[np.ndarray, ...]:
        """Project the changed suffix after the last interned record boundary."""
        result = []
        for key, skip in zip(MEMORY_FIELDS, starts, strict=True):
            spans = self.spans.tables[key]
            start, stop = spans.offsets[index : index + 2]
            result.append(
                replace(spans, records=spans.records[start + skip : stop]).features()
            )
        return tuple(result)


@dataclass(frozen=True)
class ObservationSpans:
    fact_divisors: np.ndarray
    tables: dict[str, RowSpans]
    capacities: dict[str, int]
    buckets: BucketSizes | None
    needs: np.ndarray | None

    memory: PrefixObservation | None = None

    def fact_payloads(self, rows: np.ndarray) -> np.ndarray:
        values = rows[..., FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2].astype(np.float32)
        return values / self.fact_divisors[rows[..., FACT_KIND_INDEX], None]

    def dense(self, key: str) -> np.ndarray:
        name = key.removesuffix("_valid")
        spans = self.tables[name]
        if name != key:
            return spans.mask(self.capacities[name]).astype(np.float32)
        if spans.records.dtype.names is None:
            values = spans.records
        else:
            values = spans.features()
            if key == "facts":
                values[:, FACT_VALUE_INDEX : FACT_VALUE_INDEX + 2] = self.fact_payloads(
                    spans.integers
                )
        return spans.scatter(
            values, self.capacities[name], -1 if key == "action_focus" else 0
        ).astype(np.int32 if key == "action_focus" else np.float32, copy=False)


def observation_spans(
    batch: managym.ObservationBatch | ObservationSelection,
    space: ObservationSpace,
    *,
    bucketed: bool,
    objects: bool,
    minimum_needs: np.ndarray | None = None,
) -> ObservationSpans:
    """Bind a native batch to the exact model layout before trusting any join."""
    memory = None
    if type(batch) is ObservationSelection:
        memory = batch.memory
        batch = batch.batch
    if (
        type(batch) is not managym.ObservationBatch
        or batch.layout.shapes != space.shapes
    ):
        raise ValueError("native observation layout differs from policy")
    schema = batch_layout(batch.layout)
    tables: dict[str, RowSpans] = {}
    capacities: dict[str, int] = {}
    for key, (seat_axis, offsets, values) in batch.tables().items():
        axis = int(seat_axis)
        shape = space.shapes[key]
        capacities[key] = shape[axis]
        groups = (batch.batch_size, *shape[:axis])
        trailing = shape[axis + 1 :]
        layout = schema.records[key]
        records = np.frombuffer(values, layout.dtype)
        if layout.dtype.names is None:
            records = records.reshape(-1, *trailing)
        tables[key] = RowSpans(
            records,
            layout,
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
        if minimum_needs is not None:
            np.maximum(needs, minimum_needs, out=needs)
        buckets = sizes_of(needs, space, objects=objects)
        for axis, fields in space.axis_buffers.items():
            for key, _ in fields:
                capacities[key] = buckets.sizes[axis]
    return ObservationSpans(
        schema.fact_divisors, tables, capacities, buckets, needs, memory
    )
