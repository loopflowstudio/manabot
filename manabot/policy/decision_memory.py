"""Admission and host routes for shared historical reference snapshots.

Every local join is resolved here, before forward. Integer table identities
never become numeric features. Selected prefixes retain ordered repeated members.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS, MEMORY_WIDTHS, integer_words
from managym.decision_memory import (
    DefinitionProgram,
    decision_name_values,
    definition_programs,
)

if TYPE_CHECKING:
    from manabot.env.memory_store import PrefixObservation, RecordNode, Tail
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Packed, Tables

_FIELDS = MEMORY_FIELDS
_WIDTHS = tuple(MEMORY_WIDTHS[key] for key in _FIELDS)


@dataclass(frozen=True)
class MemoryFeatures[A]:
    """Prefix-independent features owned by one admitted run of records."""

    numbers: A
    kinds: A
    choices: A
    choice_names: A
    reference_numbers: A
    reference_definitions: A
    detail_fields: A
    detail_values: A
    detail_numbers: A
    detail_references: A
    member_references: A
    member_groups: A
    member_positions: A


@dataclass(frozen=True)
class MemoryLevel[A]:
    """Independent chains packed for one GRU call.

    Collection's record_nodes routes every packed output to its recurrent state;
    learner chunks retain only terminal states through nodes/schedule.restore.
    """

    records: A
    parents: A
    nodes: A
    schedule: Packed[A]
    record_nodes: A | None = None


@dataclass(frozen=True)
class MemoryFold[A]:
    """One causal node tree; index zero is the zero initial state.

    Collection can supply detached root states [roots, 64] and the references
    introduced through those roots. Local detail routes index only newly embedded
    references; member routes index cached references followed by new ones.
    Node keys/slices and the optional result sink are ephemeral serving metadata,
    never game facts. Learner folds have no cached roots or result sink.
    """

    levels: tuple[MemoryLevel[A], ...]
    tails: A
    node_count: int
    keys: tuple[str, ...] = ()
    reference_slices: tuple[tuple[int, int], ...] = ()
    record_slices: tuple[tuple[int, int], ...] = ()
    root_nodes: A | None = None
    root_states: A | None = None
    root_references: A | None = None
    retain: Callable[[A, A], None] | None = None


@dataclass(frozen=True)
class MemoryRows[A](MemoryFeatures[A]):
    """Features and sequence coordinates for selected policy rows."""

    record_batches: A
    record_slots: A
    nonempty_batches: A
    schedule: Packed[A]
    lengths: tuple[int, ...]
    width: int
    fold: MemoryFold[A] | None = None


def memory_lengths(obs: "ObservationBuffers", row: int) -> tuple[int, int, int, int]:
    headers, details, references = (obs[key][row] for key in _FIELDS[:3])
    return (
        int(headers[:, 19].sum()),
        int(details[:, 5].sum()),
        int(references[:, 12].sum()),
        int(headers[:, 15:18].sum()),
    )


def _integers(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError("invalid decision-memory integer values")


def _outside_integer_range(values: np.ndarray, low: int, high: int) -> bool:
    """Bounds replace set membership only after finite integer admission."""
    return not bool(((values >= low) & (values <= high)).all())


def _words(values: np.ndarray) -> None:
    if not ((values >= 0) & (values <= 65535)).all():
        raise ValueError("invalid decision-memory integer words")


def _live(rows: np.ndarray, valid: int) -> np.ndarray:
    if not np.isin(rows[:, valid], (0, 1)).all():
        raise ValueError("invalid decision-memory validity")
    count = int(rows[:, valid].sum())
    if not (rows[:count, valid] == 1).all() or rows[count:].any():
        raise ValueError(
            "decision-memory rows must be a chronological prefix with zero padding"
        )
    live = rows[:count]
    _integers(live)
    return live


@dataclass(frozen=True)
class MemoryCursor:
    """The complete boundary contract between consecutive record chunks."""

    records: int = 0
    span: int = 0
    members: int = 0
    references: int = 0
    details: int = 0


def live_memory(obs: "ObservationBuffers") -> list[tuple[np.ndarray, ...]]:
    """Validate arbitrary dense shapes, padding and integers before slicing."""
    present = [key in obs for key in _FIELDS]
    if not any(present):
        return []
    if not all(present):
        raise ValueError("decision memory requires all companion buffers")
    arrays = [obs[key] for key in _FIELDS]
    if any(
        a.ndim != 3 or a.shape[-1] != w or a.shape[0] != arrays[0].shape[0]
        for a, w in zip(arrays, _WIDTHS, strict=True)
    ):
        raise ValueError("invalid decision-memory shapes")
    result = []
    for headers, rows, snapshots, joins in zip(*arrays, strict=True):
        records, details, references = (
            _live(headers, 19),
            _live(rows, 5),
            _live(snapshots, 12),
        )
        counts = records[:, 15:18]
        total = counts.sum(dtype=np.float64)
        if (counts < 0).any() or total > len(joins):
            raise ValueError("invalid decision-memory member partition")
        members = joins[: int(total)]
        _integers(members)
        if joins[int(total) :].any():
            raise ValueError("unowned decision-memory member rows")
        result.append((records, details, references, members))
    return result


def record_chunks(
    rows: tuple[np.ndarray, ...], size: int = 32, cursor: MemoryCursor = MemoryCursor()
) -> list[tuple[np.ndarray, ...]]:
    """Partition records with their owned members and newly introduced snapshots.

    Check slicing bounds before conversion: malformed imports cannot allocate
    from counts or hide an unowned companion suffix outside the final chunk.
    """
    records, details, references, members = rows
    result = []
    member_start, reference_start, detail_start = (
        cursor.members,
        cursor.references,
        cursor.details,
    )
    for start in range(cursor.records, len(records), size):
        headers = records[start : start + size]
        count = headers[:, 15:18].sum(dtype=np.float64)
        if count < 0 or count > len(members) - member_start:
            raise ValueError("invalid decision-memory member partition")
        member_end = member_start + int(count)
        joins = members[member_start:member_end]
        last = joins[:, 0].max(initial=reference_start)
        if last < reference_start or last > len(references):
            raise ValueError("invalid decision-memory member join")
        reference_end = int(last)
        snapshots = references[reference_start:reference_end]
        counts = snapshots[:, 11]
        count = counts.sum(dtype=np.float64)
        if (counts < 0).any() or count > len(details) - detail_start:
            raise ValueError("invalid decision-memory reference detail partition")
        detail_end = detail_start + int(count)
        result.append((headers, details[detail_start:detail_end], snapshots, joins))
        member_start, reference_start, detail_start = (
            member_end,
            reference_end,
            detail_end,
        )
    if (member_start, reference_start, detail_start) != (
        len(members),
        len(references),
        len(details),
    ):
        raise ValueError("unowned decision-memory companion rows")
    return result


def admit_memory(obs: "ObservationBuffers") -> None:
    for rows in live_memory(obs):
        cursor = MemoryCursor()
        for chunk in record_chunks(rows):
            cursor = admit_chunk(chunk, cursor)


def admit_chunk(rows: tuple[np.ndarray, ...], cursor: MemoryCursor) -> MemoryCursor:
    """Admit one immutable record chunk against its exact predecessor boundary."""
    records, details, references, joins = rows
    for block in rows:
        _integers(block)
    if (
        not (records[:, 19] == 1).all()
        or not (details[:, 5] == 1).all()
        or not (references[:, 12] == 1).all()
    ):
        raise ValueError("invalid decision-memory chunk validity")
    _words(
        records[:, [*range(10), 12, 13, *range(20, 60), 61, 62, 64, 65, 67, 68, 70, 71]]
    )
    _words(references[:, 2:6])
    _words(details[:, 2:4])
    if (
        _outside_integer_range(records[:, 10], 0, 4)
        or _outside_integer_range(records[:, 11], 0, 11)
        or _outside_integer_range(records[:, 18], 1, 2)
    ):
        raise ValueError("invalid decision-memory phase, step or chooser")
    if _outside_integer_range(records[:, [60, 63, 66, 69, 72, 73]], 0, 1):
        raise ValueError("invalid decision-memory selector presence")
    for offset in (60, 63, 66, 69):
        if records[records[:, offset] == 0, offset + 1 : offset + 3].any():
            raise ValueError("unavailable decision-memory selector has a value")
    if records[records[:, 72] == 0, 73].any():
        raise ValueError("unavailable decision-memory declaration has a value")
    # Word arithmetic must be exact even above float32's integer range.
    temporal = records[:, :10].astype(np.int64).reshape(-1, 5, 2)
    temporal = temporal[:, :, 0] * 65536 + temporal[:, :, 1]
    if (
        not np.array_equal(
            temporal[:, 0], np.arange(cursor.records, cursor.records + len(records))
        )
        or (temporal[:, 1] < 1).any()
        or not np.array_equal(
            temporal[:, 3], np.r_[cursor.span, temporal[:-1, 4] + 1][: len(records)]
        )
        or (temporal[:, 4] - temporal[:, 3] + 1 != temporal[:, 1]).any()
    ):
        raise ValueError("invalid decision-memory ordinal, span or multiplicity")

    # Bound before integer conversion/repeat: imported counts can be finite
    # integers far beyond int64, and may never drive an allocation.
    raw_counts = records[:, 15:18]
    member_count = raw_counts.sum(dtype=np.float64)
    if (raw_counts < 0).any() or member_count > len(joins):
        raise ValueError("invalid decision-memory member partition")
    counts = raw_counts.astype(np.int64)
    record_ends = counts.sum(1).cumsum()
    if not np.array_equal(records[:, 14], cursor.members + record_ends - counts.sum(1)):
        raise ValueError("invalid decision-memory member partition")
    members = joins[: int(member_count)]
    _integers(members)
    group_counts = counts.ravel()
    group_starts = group_counts.cumsum() - group_counts
    if (
        not np.array_equal(
            members[:, 1],
            np.arange(len(members)) - np.repeat(group_starts, group_counts),
        )
        or (
            (members[:, 0] < 1) | (members[:, 0] > cursor.references + len(references))
        ).any()
    ):
        raise ValueError("invalid decision-memory member join or ordinal")
    identities, first = np.unique(
        members[members[:, 0] > cursor.references, 0], return_index=True
    )
    if len(identities) and (
        not np.array_equal(
            identities,
            np.arange(cursor.references + 1, cursor.references + len(identities) + 1),
        )
        or (np.diff(first) <= 0).any()
    ):
        raise ValueError(
            "decision-memory snapshots must be ordered by first occurrence"
        )
    if joins[len(members) :].any() or len(identities) != len(references):
        raise ValueError("unowned decision-memory member or snapshot rows")
    if (
        _outside_integer_range(references[:, 13], 0, 1)
        or ((references[:, 13] == 1) & (references[:, 1] == 0)).any()
    ):
        raise ValueError("invalid decision-memory definition shortcut")

    raw_counts = references[:, 11]
    detail_count = raw_counts.sum(dtype=np.float64)
    if (raw_counts < 0).any() or detail_count > len(details):
        raise ValueError("invalid decision-memory reference detail partition")
    counts = raw_counts.astype(np.int64)
    starts = counts.cumsum() - counts
    if not np.array_equal(references[:, 10], starts + cursor.details):
        raise ValueError("invalid decision-memory reference detail partition")
    if detail_count != len(details):
        raise ValueError("unowned decision-memory reference details")
    if (
        (references[:, 1] < 0).any()
        or _outside_integer_range(references[:, 0], 0, 2)
        or _outside_integer_range(references[:, 6:10], 0, 1)
    ):
        raise ValueError("invalid decision-memory reference features")
    unknown = references[:, 0] == 0
    if (
        references[unknown, :10].any()
        or references[unknown, 11].any()
        or references[unknown, 13].any()
    ):
        raise ValueError("unavailable decision-memory reference has features")

    # Route every owned detail once. Empty reference slices retain their
    # place in counts but own no rows; each program must precede its mana.
    owners = np.repeat(np.arange(len(references)), counts)
    program = details[:, 0] == 0
    programs = np.bincount(owners[program], minlength=len(references))
    positions = np.arange(len(details)) - np.repeat(starts, counts)
    expected_program = positions < programs[owners]
    if (
        not np.array_equal(program, expected_program)
        or not (details[~program, 0] == 1).all()
        or not ((details[program, 1] >= 1) & (details[program, 1] <= 9)).all()
        or not (details[~program, 1] == -1).all()
    ):
        raise ValueError("invalid decision-memory program or mana kind")
    coordinates = np.where(program, positions, positions - programs[owners])
    if (
        not np.array_equal(details[:, 4], coordinates)
        or ((counts - programs) % 6).any()
    ):
        raise ValueError("invalid decision-memory program or mana coordinates")
    if ((references[:, 13] == 1) & (programs > 0)).any():
        raise ValueError("definition-only snapshot also owns a program")
    boolean = details[:, 1] == 5
    if details[boolean, 2].any() or _outside_integer_range(details[boolean, 3], 0, 1):
        raise ValueError("invalid decision-memory boolean token")
    return MemoryCursor(
        cursor.records + len(records),
        int(temporal[-1, 4]) + 1 if len(records) else cursor.span,
        cursor.members + len(joins),
        cursor.references + len(references),
        cursor.details + len(details),
    )


def validate_memory_vocabulary(obs: "ObservationBuffers", tables: "Tables") -> None:
    """Validate only owned values, including for a control without learned heads."""
    from manabot.policy.prepare import record_word

    if _FIELDS[0] not in obs:
        return
    programs = definition_programs()
    for batch in range(obs[_FIELDS[0]].shape[0]):
        count, detail_count, reference_count, _ = memory_lengths(obs, batch)
        references = obs[_FIELDS[2]][batch, :reference_count]
        for definition in references[references[:, 13] == 1, 1].astype(np.int64):
            if (
                not 0 < definition <= len(programs)
                or not programs[int(definition) - 1].eligible
            ):
                raise ValueError("ineligible decision-memory definition shortcut")
        headers = obs[_FIELDS[0]][batch, :count]
        details = obs[_FIELDS[1]][batch, :detail_count]
        if not np.isin(record_word(headers[:, 12:14], 0), decision_name_values()).all():
            raise ValueError("unadmitted decision-memory kind")
        chosen = headers[:, 60] == 1
        if not np.isin(
            record_word(headers[chosen, 61:63], 0), decision_name_values()
        ).all():
            raise ValueError("unadmitted decision-memory choice")
        program = details[:, 0] == 0
        tables.tokens(
            details[program, 1].astype(np.int64), record_word(details[program, 2:4], 0)
        )
        named = record_word(headers[:, 70:72], 0)
        if (named < 0).any() or (named >= len(tables.name_definitions)).any():
            raise ValueError("invalid decision-memory name row")


def _scaled(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (np.sign(values) * np.log1p(np.abs(values)) / 10).astype(np.float32)


@dataclass(frozen=True)
class _ProgramRows:
    source: tuple[DefinitionProgram, ...]
    rows: np.ndarray
    starts: np.ndarray
    counts: np.ndarray


_program_rows: _ProgramRows | None = None


def _catalog_program_rows() -> _ProgramRows:
    """Expand immutable native definitions once, without a policy-token lookup.

    Lookup admission still sees only definitions used by the supplied prefixes.
    Native catalog cache replacement invalidates this derived cache by identity.
    """
    global _program_rows
    source = definition_programs()
    if _program_rows is None or _program_rows.source is not source:
        counts = np.array([len(program.tokens) for program in source], np.int64)
        rows = np.array(
            [
                [0, kind, *integer_words(value), ordinal, 1]
                for program in source
                for ordinal, (kind, value) in enumerate(program.tokens)
            ],
            np.float32,
        ).reshape(-1, 6)
        _program_rows = _ProgramRows(source, rows, counts.cumsum() - counts, counts)
    return _program_rows


def _expand_details(
    references: np.ndarray, details: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Route catalog and explicit detail rows in reference order, without a loop.

    Catalog programs precede each reference's explicit mana rows. Empty slices
    keep their reference index, so unknown snapshots and repeated joins retain
    exactly the same ownership as the native transport.
    """
    counts = references[:, 11].astype(np.int64)
    shortcut_counts = np.zeros(len(references), np.int64)
    shortcut_starts = np.zeros(len(references), np.int64)
    shortcuts = references[:, 13] == 1
    programs = None
    if shortcuts.any():
        programs = _catalog_program_rows()
        definitions = references[shortcuts, 1].astype(np.int64) - 1
        shortcut_counts[shortcuts] = programs.counts[definitions]
        shortcut_starts[shortcuts] = programs.starts[definitions]
    expanded_counts = counts + shortcut_counts
    owners = np.repeat(np.arange(len(references), dtype=np.int64), expanded_counts)
    positions = np.arange(len(owners)) - np.repeat(
        expanded_counts.cumsum() - expanded_counts, expanded_counts
    )
    catalog = positions < shortcut_counts[owners]
    expanded = np.empty((len(owners), 6), np.float32)
    if programs is not None:
        expanded[catalog] = programs.rows[
            shortcut_starts[owners[catalog]] + positions[catalog]
        ]
    explicit = ~catalog
    expanded[explicit] = details[
        references[owners[explicit], 10].astype(np.int64)
        + positions[explicit]
        - shortcut_counts[owners[explicit]]
    ]
    return expanded, owners


def _node_rows(node: Tail | RecordNode) -> tuple[np.ndarray, ...]:
    from manabot.env.memory_store import RecordNode

    return node.rows if isinstance(node, RecordNode) else node.chunk.rows


def chunk_observation(node: "Tail | RecordNode") -> dict[str, np.ndarray]:
    """Local companion offsets for featurization; temporal words stay absolute."""
    cursor = node.previous.cursor if node.previous else MemoryCursor()
    arrays = dict(zip(_FIELDS, (a[None] for a in _node_rows(node)), strict=True))
    references = arrays[_FIELDS[2]].copy()
    references[..., 10] -= cursor.details
    arrays[_FIELDS[2]] = references
    members = arrays[_FIELDS[3]].copy()
    members[..., 0] -= cursor.references
    arrays[_FIELDS[3]] = members
    return arrays


def retained_memory(obs: "ObservationBuffers") -> "PrefixObservation | None":
    from manabot.env.memory_store import PrefixObservation

    return (
        obs
        if isinstance(obs, PrefixObservation) and not any(key in obs for key in _FIELDS)
        else None
    )


def validate_retained_memory(
    obs: "PrefixObservation", tables: "Tables", definition_count: int, include_own: bool
) -> None:
    from manabot.policy.admission import definition_ids
    from manabot.policy.prepare import record_word

    binding = (tables.memory_binding, definition_count, include_own)
    lookup = tables.definition_rows
    size = definition_count if lookup is None else len(lookup)
    pending: dict[str, Tail] = {}
    for prefix in obs.prefixes:
        node = prefix.tail
        while (
            node is not None
            and node.features.admission != binding
            and node.key not in pending
        ):
            pending[node.key] = node
            node = node.previous
    if not pending:
        return
    # Chunk certificates already prove per-chain temporal/partition joins.
    # Vocabulary, own-seat and definition checks are independent of those joins,
    # so admit the new chunks across all selected rows in one arithmetic batch.
    headers, details, references = (
        np.concatenate([_node_rows(node)[index] for node in pending.values()])
        for index in range(3)
    )
    if not include_own and (headers[:, 18] == 1).any():
        raise ValueError("own-seat decision memory is disabled")
    definition_ids(references[:, 1], size, lookup)
    definition_ids(
        record_word(details, 2)[(details[:, 0] == 0) & (details[:, 1] == 7)] + 1,
        size,
        lookup,
    )
    validate_memory_vocabulary(
        dict(
            zip(
                _FIELDS[:3],
                (a[None] for a in (headers, details, references)),
                strict=True,
            )
        ),
        tables,
    )
    for node in pending.values():
        node.features.admission = binding


def _prepare_chunk_features(nodes: list[Tail | RecordNode], tables: Tables) -> None:
    """Featurize each record once; raw learner chunks only assemble its fragments."""
    from manabot.env.memory_store import RecordNode, chunk_records

    pending = [
        node
        for node in nodes
        if node.features.prepared is None
        or node.features.prepared[0] != tables.memory_binding
    ]
    if not pending:
        return
    chunks = {
        node: chunk_records(node)
        for node in pending
        if not isinstance(node, RecordNode)
    }
    records = {
        record.key: record
        for node in pending
        for record in ([node] if isinstance(node, RecordNode) else chunks[node])
    }
    _featurize_records(list(records.values()), tables)
    for chunk, selected in chunks.items():
        cursor = chunk.previous.cursor if chunk.previous else MemoryCursor()
        fragments = []
        reference_offsets = []
        for record in selected:
            cached = record.features.prepared
            assert cached is not None and cached[0] == tables.memory_binding
            fragments.append(cached[1])
            previous = record.previous.cursor if record.previous else MemoryCursor()
            reference_offsets.append(previous.references - cursor.references)
        values = {
            f.name: np.concatenate(
                [getattr(fragment, f.name) for fragment in fragments]
            )
            for f in fields(MemoryFeatures)
        }
        # Offset joins in bulk, rather than allocating three arrays per record.
        members = [len(fragment.member_references) for fragment in fragments]
        values["detail_references"] += np.repeat(
            reference_offsets,
            [len(fragment.detail_references) for fragment in fragments],
        )
        values["member_references"] += np.repeat(reference_offsets, members)
        values["member_groups"] += np.repeat(np.arange(len(selected)) * 3, members)
        for value in values.values():
            value.setflags(write=False)
        chunk.features.prepared = (tables.memory_binding, MemoryFeatures(**values))


def _featurize_records(nodes: list[RecordNode], tables: Tables) -> None:
    """Featurize new nodes together; cached fragments contribute no raw features.

    Concatenated records are only a host arithmetic batch, never a GRU sequence.
    Split the immutable results back onto their original nodes, with local joins.
    """
    pending = [
        node
        for node in nodes
        if node.features.prepared is None
        or node.features.prepared[0] != tables.memory_binding
    ]
    if not pending:
        return
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in _FIELDS}
    detail_base = 0
    for node in pending:
        local = chunk_observation(node)
        local[_FIELDS[2]][..., 10] += detail_base
        for key in blocks:
            blocks[key].append(local[key])
        detail_base += len(_node_rows(node)[1])
    prepared = _prepare_dense_memory(
        {key: np.concatenate(parts, axis=1) for key, parts in blocks.items()}, tables
    )
    assert prepared is not None
    record_base = reference_base = member_base = 0
    record_fields = {"numbers", "kinds", "choices", "choice_names"}
    reference_fields = {"reference_numbers", "reference_definitions"}
    for node in pending:
        record_end = record_base + len(_node_rows(node)[0])
        reference_end = reference_base + len(_node_rows(node)[2])
        member_end = member_base + len(_node_rows(node)[3])
        detail_start, detail_end = np.searchsorted(
            prepared.detail_references, [reference_base, reference_end]
        )
        values: dict[str, np.ndarray] = {}
        for feature in fields(MemoryFeatures):
            name = feature.name
            value = getattr(prepared, name)
            if name in record_fields:
                value = value[record_base:record_end]
            elif name in reference_fields:
                value = value[reference_base:reference_end]
            elif name.startswith("detail_"):
                value = value[detail_start:detail_end]
                if name == "detail_references":
                    value = value - reference_base
            else:
                value = value[member_base:member_end]
                if name == "member_groups":
                    value = value - record_base * 3
            # Own only this node's bytes, never the entire transient batch.
            value = value.copy()
            value.setflags(write=False)
            values[name] = value
        node.features.prepared = (tables.memory_binding, MemoryFeatures(**values))
        record_base, reference_base, member_base = record_end, reference_end, member_end


def _chunk_features(
    node: Tail | RecordNode, tables: Tables
) -> MemoryFeatures[np.ndarray]:
    """Read one immutable record fragment or its assembled learner chunk."""
    cached = node.features.prepared
    if cached is None or cached[0] != tables.memory_binding:
        _prepare_chunk_features([node], tables)
        cached = node.features.prepared
    assert cached is not None
    return cached[1]


def _empty_memory(tables: Tables) -> MemoryRows[np.ndarray]:
    """The dense encoder owns the canonical empty shapes and dtypes."""
    empty = _prepare_dense_memory(
        {
            key: np.zeros((0, 0, width), np.float32)
            for key, width in zip(_FIELDS, _WIDTHS, strict=True)
        },
        tables,
    )
    assert empty is not None
    return empty


def _prepare_retained(
    obs: "PrefixObservation", tables: "Tables"
) -> MemoryRows[np.ndarray]:
    from manabot.env.memory_store import record_tail
    from manabot.policy.prepare import pack

    lengths = tuple(p.record_count for p in obs.prefixes)
    arrays: dict[str, list[np.ndarray]] = {f.name: [] for f in fields(MemoryFeatures)}
    batches: list[np.ndarray] = []
    slots: list[np.ndarray] = []
    reference_base = record_base = 0
    for batch, prefix in enumerate(obs.prefixes):
        record_tail(prefix)
        for node in prefix.tail.chunks() if prefix.tail else ():
            fragment = _chunk_features(node, tables)
            cursor = node.previous.cursor if node.previous else MemoryCursor()
            for name in arrays:
                value = getattr(fragment, name)
                if name == "detail_references":
                    value = value + reference_base + cursor.references
                elif name == "member_references":
                    value = value + reference_base + cursor.references
                elif name == "member_groups":
                    value = value + (record_base + cursor.records) * 3
                arrays[name].append(value)
        batches.append(np.full(prefix.record_count, batch, np.int64))
        slots.append(np.arange(prefix.record_count, dtype=np.int64))
        reference_base += prefix.counts[2]
        record_base += prefix.record_count
    empty = _empty_memory(tables)
    values = {
        name: np.concatenate(parts) if parts else getattr(empty, name)
        for name, parts in arrays.items()
    }
    nonempty = np.flatnonzero(lengths).astype(np.int64)
    return replace(
        empty,
        **values,
        record_batches=np.concatenate(batches) if batches else empty.record_batches,
        record_slots=np.concatenate(slots) if slots else empty.record_slots,
        lengths=lengths,
        width=max(1, max(lengths, default=0)),
        nonempty_batches=nonempty,
        schedule=pack(np.array(lengths, np.int64)[nonempty]),
    )


def _prepare_fold(
    obs: PrefixObservation, tables: Tables, roots: Mapping[str, int] | None = None
) -> MemoryRows[np.ndarray]:
    """Embed each selected node once, with exact joins to predecessor references.

    No neural tensors or raw prefix owners escape preparation. The tree and its
    autograd state belong to this execution chunk, never to the rollout cache.
    """
    from manabot.env.memory_store import record_tail
    from manabot.policy.prepare import pack

    collection = roots is not None
    if not collection:
        for prefix in obs.prefixes:
            record_tail(prefix)
    tails = [record_tail(p) if collection else p.tail for p in obs.prefixes]
    roots = roots or {}
    selected_nodes: dict[str, Tail | RecordNode] = {}
    for tail in tails:
        pending: list[Tail | RecordNode] = []
        node = tail
        while node is not None and node.key not in selected_nodes:
            pending.append(node)
            if node.key in roots:
                break
            node = node.previous
        for node in reversed(pending):
            selected_nodes[node.key] = node
    cached = [node for node in selected_nodes.values() if node.key in roots]
    root_offsets: dict[str, int] = {}
    reference_base = 0
    for node in cached:
        root_offsets[node.key] = reference_base
        reference_base += roots[node.key]
    root_reference_count = reference_base
    nodes: list[Tail | RecordNode] = []
    reference_slices: list[tuple[int, int]] = []
    record_slices: list[tuple[int, int]] = []
    indexes: dict[str, int] = {}
    depths: list[int] = []
    references: list[np.ndarray] = [np.empty(0, np.int64)]
    record_starts: list[int] = []
    arrays: dict[str, list[np.ndarray]] = {f.name: [] for f in fields(MemoryFeatures)}
    record_base = 0
    parents: list[int] = []
    _prepare_chunk_features(
        [node for node in selected_nodes.values() if node.key not in roots], tables
    )
    for node in selected_nodes.values():
        parent = (
            indexes[node.previous.key] if node.previous and node.key not in roots else 0
        )
        if node.key in roots:
            count = roots[node.key]
            start = root_offsets[node.key]
            joined = np.concatenate(
                (references[parent], np.arange(start, start + count))
            )
            record_starts.append(-1)
            reference_slices.append((start, count))
            record_slices.append((-1, 0))
            nodes.append(node)
            indexes[node.key] = len(nodes)
            references.append(joined)
            parents.append(parent)
            depths.append(-1)
            continue
        fragment = _chunk_features(node, tables)
        count = len(fragment.reference_numbers)
        reference_slices.append((reference_base, count))
        record_slices.append((record_base, len(fragment.numbers)))
        joined = np.concatenate(
            (references[parent], np.arange(reference_base, reference_base + count))
        )
        # Fragment member joins are relative to the newly introduced rows;
        # negative values name a reference owned by a predecessor chunk.
        cursor = node.previous.cursor if node.previous else MemoryCursor()
        for name in arrays:
            value = getattr(fragment, name)
            if name == "detail_references":
                value = value + reference_base - root_reference_count
            elif name == "member_references":
                value = joined[value + cursor.references]
            elif name == "member_groups":
                value = value + record_base * 3
            arrays[name].append(value)
        record_starts.append(record_base)
        reference_base += count
        record_base += len(fragment.numbers)
        nodes.append(node)
        indexes[node.key] = len(nodes)
        references.append(joined)
        parents.append(parent)
        depths.append(depths[parent - 1] + 1 if parent else 0)
    # Sequence lengths retain original policy row order; the fold owns routes.
    empty = _empty_memory(tables)
    levels: list[MemoryLevel[np.ndarray]] = []
    if collection:
        # One packed sequence per unbranched suffix, retaining the GRU output
        # after every record. Per-record state caching needs no per-record call.
        children: dict[int, list[int]] = {}
        for i, parent in enumerate(parents):
            if record_starts[i] >= 0:
                children.setdefault(parent, []).append(i)
        paths: dict[int, list[list[int]]] = {}
        ends: dict[int, int] = {0: -1}
        ends.update((i + 1, -1) for i, start in enumerate(record_starts) if start < 0)
        for i, start in enumerate(record_starts):
            if start < 0 or (
                parents[i] not in ends and len(children.get(parents[i], ())) == 1
            ):
                continue
            path = [i]
            while len(children.get(path[-1] + 1, ())) == 1:
                path.append(children[path[-1] + 1][0])
            depth = ends[parents[i]] + 1
            paths.setdefault(depth, []).append(path)
            ends[path[-1] + 1] = depth
        for paths_at_depth in paths.values():
            schedule = pack(np.array([len(path) for path in paths_at_depth], np.int64))
            selected = np.array([i for path in paths_at_depth for i in path], np.int64)
            levels.append(
                MemoryLevel(
                    np.array([record_starts[i] for i in selected], np.int64)[
                        schedule.gather
                    ],
                    np.array([parents[path[0]] for path in paths_at_depth], np.int64)[
                        np.argsort(schedule.restore)
                    ],
                    np.array([path[-1] + 1 for path in paths_at_depth], np.int64),
                    schedule,
                    (selected + 1)[schedule.gather],
                )
            )
    else:
        for depth in range(max(depths, default=-1) + 1):
            selected = [i for i, value in enumerate(depths) if value == depth]
            lengths = np.array(
                [len(_node_rows(nodes[i])[0]) for i in selected], np.int64
            )
            schedule = pack(lengths)
            records = np.concatenate(
                [
                    np.arange(record_starts[i], record_starts[i] + length)
                    for i, length in zip(selected, lengths, strict=True)
                ]
            )
            levels.append(
                MemoryLevel(
                    records[schedule.gather],
                    np.array([parents[i] for i in selected], np.int64)[
                        np.argsort(schedule.restore)
                    ],
                    np.array(selected, np.int64) + 1,
                    schedule,
                )
            )
    lengths = tuple(p.record_count for p in obs.prefixes)
    return replace(
        empty,
        **{
            name: np.concatenate(parts) if parts else getattr(empty, name)
            for name, parts in arrays.items()
        },
        lengths=lengths,
        fold=MemoryFold(
            tuple(levels),
            np.array([indexes[tail.key] if tail else 0 for tail in tails], np.int64),
            len(nodes),
            keys=tuple(node.key for node in nodes),
            reference_slices=tuple(reference_slices),
            record_slices=tuple(record_slices),
            root_nodes=np.array([indexes[node.key] for node in cached], np.int64),
        ),
    )


def prepare_memory(
    obs: "ObservationBuffers",
    tables: "Tables",
    *,
    fold: bool = False,
    roots: Mapping[str, int] | None = None,
) -> MemoryRows[np.ndarray] | None:
    retained = retained_memory(obs)
    return (
        (
            _prepare_fold(retained, tables, roots)
            if fold
            else _prepare_retained(retained, tables)
        )
        if retained is not None
        else _prepare_dense_memory(obs, tables)
    )


def _prepare_dense_memory(
    obs: "ObservationBuffers", tables: "Tables"
) -> MemoryRows[np.ndarray] | None:
    from manabot.policy.prepare import NUMBER_TOKEN_KINDS, pack, record_word

    if _FIELDS[0] not in obs:
        return None
    headers = obs[_FIELDS[0]]
    lengths = headers[..., 19].sum(1).astype(np.int64)
    batches, slots = np.nonzero(headers[..., 19])
    selected = headers[batches, slots]
    temporal = np.stack(
        [record_word(selected[:, i : i + 2], 0) for i in (0, 2, 4, 6, 8)], -1
    )
    context = np.stack(
        [record_word(selected[:, i : i + 2], 0) for i in range(20, 60, 2)], -1
    )
    context[:, :2] = context[:, :2].astype(np.uint32).view(np.int32)
    selectors = np.zeros((len(selected), 10), np.float32)
    for index, offset in enumerate((60, 63, 66, 69)):
        selectors[:, index * 2] = selected[:, offset]
        if offset in (63, 66):
            selectors[:, index * 2 + 1] = _scaled(
                record_word(selected[:, offset + 1 : offset + 3], 0)
            )
    selectors[:, 8:] = selected[:, 72:74]
    numbers = np.concatenate(
        (
            _scaled(temporal),
            selected[:, 10:12] / [4, 12],
            selected[:, 18:19] / 2,
            _scaled(context),
            selectors,
        ),
        -1,
    ).astype(np.float32)
    kinds = tables.names(1, record_word(selected[:, 12:14], 0))
    choices = np.zeros(len(selected), np.int64)
    present = selected[:, 60] == 1
    choices[present] = tables.names(1, record_word(selected[present, 61:63], 0))
    choice_names = record_word(selected[:, 70:72], 0)
    references_list: list[np.ndarray] = []
    details_list: list[np.ndarray] = []
    detail_routes: list[np.ndarray] = []
    member_routes: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    reference_base = record_base = 0
    for batch, length in enumerate(lengths):
        _, detail_count, reference_count, member_count = memory_lengths(obs, batch)
        references = obs[_FIELDS[2]][batch, :reference_count]
        references_list.append(references)
        raw_details = obs[_FIELDS[1]][batch, :detail_count]
        expanded, owners = _expand_details(references, raw_details)
        details_list.append(expanded)
        detail_routes.append(owners + reference_base)
        members = obs[_FIELDS[3]][batch, :member_count]
        member_routes.append(members[:, 0].astype(np.int64) - 1 + reference_base)
        counts = headers[batch, :length, 15:18].astype(np.int64).ravel()
        groups.append(
            np.repeat(np.arange(length * 3, dtype=np.int64) + record_base * 3, counts)
        )
        positions.append((members[:, 1:2] / 128).astype(np.float32))
        reference_base += reference_count
        record_base += int(length)
    references = (
        np.concatenate(references_list)
        if references_list
        else np.zeros((0, 14), np.float32)
    )
    details = (
        np.concatenate(details_list) if details_list else np.zeros((0, 6), np.float32)
    )
    numeric = (
        np.stack([record_word(references[:, i : i + 2], 0) for i in (2, 4)], -1)
        .astype(np.uint32)
        .view(np.int32)
    )
    reference_numbers = np.concatenate(
        (
            references[:, :1] / 2,
            _scaled(numeric),
            references[:, 6:10],
            (references[:, 1:2] > 0).astype(np.float32),
        ),
        -1,
    ).astype(np.float32)
    payloads = record_word(details[:, 2:4], 0)
    token_kinds = details[:, 1].astype(np.int64)
    values = np.zeros(len(details), np.int64)
    program = details[:, 0] == 0
    values[program] = tables.tokens(token_kinds[program], payloads[program])
    signed = payloads.astype(np.uint32).view(np.int32)
    numeric_values = np.where(
        (token_kinds == -1) | np.isin(token_kinds, NUMBER_TOKEN_KINDS),
        _scaled(np.where(program, signed, payloads)),
        0,
    )
    # Field labels have one owner in the native vocabulary. No table indexes
    # or appearance/provenance identifiers enter these feature coordinates.
    # Native stable hashes of DecisionMemory::Program and ::Mana. The lookup
    # is against the supplied checkpoint vocabulary, not a current-world join.
    field_values = np.array([30649973, 1006277364], np.int64)
    fields = tables.names(1, field_values)[details[:, 0].astype(np.int64)]

    def cat(arrays: list[np.ndarray], width: int | None = None) -> np.ndarray:
        return (
            np.concatenate(arrays)
            if arrays
            else np.zeros((0,) if width is None else (0, width), np.int64)
        )

    return MemoryRows(
        numbers,
        kinds,
        choices,
        choice_names,
        reference_numbers,
        tables.rows(references[:, 1].astype(np.int64)),
        fields,
        values,
        np.stack(
            (
                token_kinds / 9,
                np.where(program, details[:, 4] / 128, 0),
                np.where(program, 0, np.floor_divide(details[:, 4], 6) / 128),
                np.where(program, 0, np.remainder(details[:, 4], 6) / 5),
                numeric_values,
            ),
            -1,
        ).astype(np.float32),
        cat(detail_routes),
        cat(member_routes),
        cat(groups),
        cat(positions, 1),
        batches.astype(np.int64),
        slots.astype(np.int64),
        np.flatnonzero(lengths).astype(np.int64),
        pack(lengths[lengths > 0]),
        tuple(map(int, lengths)),
        max(1, int(lengths.max(initial=0))),
    )
