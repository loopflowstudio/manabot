"""Admission and host routes for shared historical reference snapshots.

Every local join is resolved here, before forward. Integer table identities
never become numeric features. Selected prefixes retain ordered repeated members.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS, MEMORY_WIDTHS
from manabot.env.observation import column, column_names
from manabot.env.shared_effects import NUMBER_TOKEN_KINDS
from manabot.policy.readings import signed_log
from managym.decision_memory import (
    DefinitionProgram,
    decision_name_values,
    definition_programs,
)

if TYPE_CHECKING:
    from manabot.env.memory_store import PrefixObservation, RecordNode, Tail
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Packed, Tables
    from manabot.policy.spans import ObservationSpans


def _span(array: str, first: str, last: str) -> slice:
    return slice(column(array, first), column(array, last) + 1)


# Native columns of the four memory tables, by name.
_RECORDS, _DETAILS, _REFERENCES, _MEMBERS = MEMORY_FIELDS
# Ordinal, count, turn, first and last: five 16-bit word pairs.
_RECORD_TEMPORAL = _span(_RECORDS, "ordinal_high", "last_low")
_RECORD_PHASE = column(_RECORDS, "phase")
_RECORD_STEP = column(_RECORDS, "step")
_RECORD_PHASE_STEP = _span(_RECORDS, "phase", "step")
_RECORD_KIND = column(_RECORDS, "kind_high")
_RECORD_MEMBER_OFFSET = column(_RECORDS, "member_offset")
# Mana, response and choice member counts.
_RECORD_COUNTS = _span(_RECORDS, "mana_count", "choice_count")
_RECORD_CHOOSER = column(_RECORDS, "chooser_role")
_RECORD_VALID = column(_RECORDS, "valid")
# Life, hand and board context of both seats: word pairs.
_RECORD_CONTEXT = _span(_RECORDS, "life_viewer_high", "board_opponent_battle_low")
# Each selector's presence flag, followed by its value's two words.
_RECORD_SELECTORS = tuple(
    column(_RECORDS, f"{name}_present")
    for name in ("choice_kind", "mode", "ability", "name")
)
_RECORD_NUMBERED_SELECTORS = (
    column(_RECORDS, "mode_present"),
    column(_RECORDS, "ability_present"),
)
_RECORD_CHOICE_PRESENT = column(_RECORDS, "choice_kind_present")
_RECORD_CHOICE_KIND = _span(_RECORDS, "choice_kind_high", "choice_kind_low")
_RECORD_NAME = column(_RECORDS, "name_high")
_RECORD_DECLARED_PRESENT = column(_RECORDS, "declared_present")
_RECORD_DECLARED = column(_RECORDS, "declared")
_RECORD_DECLARATION = _span(_RECORDS, "declared_present", "declared")
_RECORD_DECLARATION_COLUMNS = (_RECORD_DECLARED_PRESENT, _RECORD_DECLARED)
# Every 16-bit word column.
_RECORD_WORDS = [
    column(_RECORDS, name)
    for name in column_names(_RECORDS)
    if name.endswith(("_high", "_low"))
]
_REFERENCE_ROLE = column(_REFERENCES, "role")
_REFERENCE_DEFINITION = column(_REFERENCES, "definition")
_REFERENCE_WORDS = _span(_REFERENCES, "power_high", "toughness_low")
# Tapped, ready, attacking and blocking flags.
_REFERENCE_FLAGS = _span(_REFERENCES, "tapped", "blocking")
_REFERENCE_DETAIL_OFFSET = column(_REFERENCES, "detail_offset")
_REFERENCE_DETAIL_COUNT = column(_REFERENCES, "detail_count")
_REFERENCE_VALID = column(_REFERENCES, "valid")
_REFERENCE_DEFINITION_ONLY = column(_REFERENCES, "definition_only")
_DETAIL_FIELD = column(_DETAILS, "field")
_DETAIL_TOKEN_KIND = column(_DETAILS, "token_kind")
_DETAIL_VALUE_HIGH = column(_DETAILS, "value_high")
_DETAIL_VALUE_LOW = column(_DETAILS, "value_low")
_DETAIL_VALUE = _span(_DETAILS, "value_high", "value_low")
_DETAIL_ORDINAL = column(_DETAILS, "ordinal")
_DETAIL_VALID = column(_DETAILS, "valid")
_MEMBER_REFERENCE = column(_MEMBERS, "reference")
_MEMBER_ORDINAL = column(_MEMBERS, "ordinal")


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
    keys: tuple[str, ...] = field(default=(), metadata={"host_only": True})
    reference_slices: tuple[tuple[int, int], ...] = field(
        default=(), metadata={"host_only": True}
    )
    record_slices: tuple[tuple[int, int], ...] = field(
        default=(), metadata={"host_only": True}
    )
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
    lengths: tuple[int, ...] = field(metadata={"host_only": True})
    width: int
    fold: MemoryFold[A] | None = None


def memory_lengths(obs: "ObservationBuffers", row: int) -> tuple[int, int, int, int]:
    headers, details, references = (obs[key][row] for key in MEMORY_FIELDS[:3])
    return (
        int(headers[:, _RECORD_VALID].sum()),
        int(details[:, _DETAIL_VALID].sum()),
        int(references[:, _REFERENCE_VALID].sum()),
        int(headers[:, _RECORD_COUNTS].sum()),
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
    present = [key in obs for key in MEMORY_FIELDS]
    if not any(present):
        return []
    if not all(present):
        raise ValueError("decision memory requires all companion buffers")
    arrays = [obs[key] for key in MEMORY_FIELDS]
    if any(
        a.ndim != 3 or a.shape[-1] != w or a.shape[0] != arrays[0].shape[0]
        for a, w in zip(arrays, MEMORY_WIDTHS.values(), strict=True)
    ):
        raise ValueError("invalid decision-memory shapes")
    result = []
    for headers, rows, snapshots, joins in zip(*arrays, strict=True):
        records, details, references = (
            _live(headers, _RECORD_VALID),
            _live(rows, _DETAIL_VALID),
            _live(snapshots, _REFERENCE_VALID),
        )
        counts = records[:, _RECORD_COUNTS]
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
    rows: tuple[np.ndarray, ...], size: int = 32, *, reference_offset: int = 0
) -> list[tuple[np.ndarray, ...]]:
    """Partition records with their owned members and newly introduced snapshots.

    Rows begin at one record boundary. Member joins retain absolute reference
    indexes, including references introduced before this suffix.
    Check slicing bounds before conversion: malformed imports cannot allocate
    from counts or hide an unowned companion suffix outside the final chunk.
    """
    records, details, references, members = rows
    result = []
    member_start = reference_start = detail_start = 0
    for start in range(0, len(records), size):
        headers = records[start : start + size]
        count = headers[:, _RECORD_COUNTS].sum(dtype=np.float64)
        if count < 0 or count > len(members) - member_start:
            raise ValueError("invalid decision-memory member partition")
        member_end = member_start + int(count)
        joins = members[member_start:member_end]
        last = (
            joins[:, _MEMBER_REFERENCE].max(initial=reference_offset + reference_start)
            - reference_offset
        )
        if last < reference_start or last > len(references):
            raise ValueError("invalid decision-memory member join")
        reference_end = int(last)
        snapshots = references[reference_start:reference_end]
        counts = snapshots[:, _REFERENCE_DETAIL_COUNT]
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
        not (records[:, _RECORD_VALID] == 1).all()
        or not (details[:, _DETAIL_VALID] == 1).all()
        or not (references[:, _REFERENCE_VALID] == 1).all()
    ):
        raise ValueError("invalid decision-memory chunk validity")
    _words(records[:, _RECORD_WORDS])
    _words(references[:, _REFERENCE_WORDS])
    _words(details[:, _DETAIL_VALUE])
    if (
        _outside_integer_range(records[:, _RECORD_PHASE], 0, 4)
        or _outside_integer_range(records[:, _RECORD_STEP], 0, 11)
        or _outside_integer_range(records[:, _RECORD_CHOOSER], 1, 2)
    ):
        raise ValueError("invalid decision-memory phase, step or chooser")
    if _outside_integer_range(
        records[:, [*_RECORD_SELECTORS, *_RECORD_DECLARATION_COLUMNS]], 0, 1
    ):
        raise ValueError("invalid decision-memory selector presence")
    for offset in _RECORD_SELECTORS:
        if records[records[:, offset] == 0, offset + 1 : offset + 3].any():
            raise ValueError("unavailable decision-memory selector has a value")
    if records[records[:, _RECORD_DECLARED_PRESENT] == 0, _RECORD_DECLARED].any():
        raise ValueError("unavailable decision-memory declaration has a value")
    # Word arithmetic must be exact even above float32's integer range.
    temporal = records[:, _RECORD_TEMPORAL].astype(np.int64).reshape(-1, 5, 2)
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
    raw_counts = records[:, _RECORD_COUNTS]
    member_count = raw_counts.sum(dtype=np.float64)
    if (raw_counts < 0).any() or member_count > len(joins):
        raise ValueError("invalid decision-memory member partition")
    counts = raw_counts.astype(np.int64)
    record_ends = counts.sum(1).cumsum()
    if not np.array_equal(
        records[:, _RECORD_MEMBER_OFFSET], cursor.members + record_ends - counts.sum(1)
    ):
        raise ValueError("invalid decision-memory member partition")
    members = joins[: int(member_count)]
    _integers(members)
    group_counts = counts.ravel()
    group_starts = group_counts.cumsum() - group_counts
    if (
        not np.array_equal(
            members[:, _MEMBER_ORDINAL],
            np.arange(len(members)) - np.repeat(group_starts, group_counts),
        )
        or (
            (members[:, _MEMBER_REFERENCE] < 1)
            | (members[:, _MEMBER_REFERENCE] > cursor.references + len(references))
        ).any()
    ):
        raise ValueError("invalid decision-memory member join or ordinal")
    identities, first = np.unique(
        members[members[:, _MEMBER_REFERENCE] > cursor.references, _MEMBER_REFERENCE],
        return_index=True,
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
        _outside_integer_range(references[:, _REFERENCE_DEFINITION_ONLY], 0, 1)
        or (
            (references[:, _REFERENCE_DEFINITION_ONLY] == 1)
            & (references[:, _REFERENCE_DEFINITION] == 0)
        ).any()
    ):
        raise ValueError("invalid decision-memory definition shortcut")

    raw_counts = references[:, _REFERENCE_DETAIL_COUNT]
    detail_count = raw_counts.sum(dtype=np.float64)
    if (raw_counts < 0).any() or detail_count > len(details):
        raise ValueError("invalid decision-memory reference detail partition")
    counts = raw_counts.astype(np.int64)
    starts = counts.cumsum() - counts
    if not np.array_equal(
        references[:, _REFERENCE_DETAIL_OFFSET], starts + cursor.details
    ):
        raise ValueError("invalid decision-memory reference detail partition")
    if detail_count != len(details):
        raise ValueError("unowned decision-memory reference details")
    if (
        (references[:, _REFERENCE_DEFINITION] < 0).any()
        or _outside_integer_range(references[:, _REFERENCE_ROLE], 0, 2)
        or _outside_integer_range(references[:, _REFERENCE_FLAGS], 0, 1)
    ):
        raise ValueError("invalid decision-memory reference features")
    unknown = references[:, _REFERENCE_ROLE] == 0
    if (
        references[unknown, :_REFERENCE_DETAIL_OFFSET].any()
        or references[unknown, _REFERENCE_DETAIL_COUNT].any()
        or references[unknown, _REFERENCE_DEFINITION_ONLY].any()
    ):
        raise ValueError("unavailable decision-memory reference has features")

    # Route every owned detail once. Empty reference slices retain their
    # place in counts but own no rows; each program must precede its mana.
    owners = np.repeat(np.arange(len(references)), counts)
    program = details[:, _DETAIL_FIELD] == 0
    programs = np.bincount(owners[program], minlength=len(references))
    positions = np.arange(len(details)) - np.repeat(starts, counts)
    expected_program = positions < programs[owners]
    if (
        not np.array_equal(program, expected_program)
        or not (details[~program, _DETAIL_FIELD] == 1).all()
        or not (
            (details[program, _DETAIL_TOKEN_KIND] >= 1)
            & (details[program, _DETAIL_TOKEN_KIND] <= 9)
        ).all()
        or not (details[~program, _DETAIL_TOKEN_KIND] == -1).all()
    ):
        raise ValueError("invalid decision-memory program or mana kind")
    coordinates = np.where(program, positions, positions - programs[owners])
    if (
        not np.array_equal(details[:, _DETAIL_ORDINAL], coordinates)
        or ((counts - programs) % 6).any()
    ):
        raise ValueError("invalid decision-memory program or mana coordinates")
    if ((references[:, _REFERENCE_DEFINITION_ONLY] == 1) & (programs > 0)).any():
        raise ValueError("definition-only snapshot also owns a program")
    boolean = details[:, _DETAIL_TOKEN_KIND] == 5
    if details[boolean, _DETAIL_VALUE_HIGH].any() or _outside_integer_range(
        details[boolean, _DETAIL_VALUE_LOW], 0, 1
    ):
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
    if _RECORDS not in obs:
        return
    for batch in range(obs[_RECORDS].shape[0]):
        count, detail_count, reference_count, _ = memory_lengths(obs, batch)
        memory_vocabulary(
            obs[_RECORDS][batch, :count],
            obs[_DETAILS][batch, :detail_count],
            obs[_REFERENCES][batch, :reference_count],
            tables,
        )


def memory_vocabulary(
    headers: np.ndarray, details: np.ndarray, references: np.ndarray, tables: "Tables"
) -> None:
    from manabot.policy.prepare import record_word

    programs = definition_programs()
    for definition in references[
        references[:, _REFERENCE_DEFINITION_ONLY] == 1, _REFERENCE_DEFINITION
    ].astype(np.int64):
        if (
            not 0 < definition <= len(programs)
            or not programs[int(definition) - 1].eligible
        ):
            raise ValueError("ineligible decision-memory definition shortcut")
    if not np.isin(record_word(headers, _RECORD_KIND), decision_name_values()).all():
        raise ValueError("unadmitted decision-memory kind")
    chosen = headers[:, _RECORD_CHOICE_PRESENT] == 1
    if not np.isin(
        record_word(headers[chosen, _RECORD_CHOICE_KIND], 0), decision_name_values()
    ).all():
        raise ValueError("unadmitted decision-memory choice")
    program = details[:, _DETAIL_FIELD] == 0
    tables.tokens(
        details[program, _DETAIL_TOKEN_KIND].astype(np.int64),
        record_word(details[program, _DETAIL_VALUE], 0),
    )
    named = record_word(headers, _RECORD_NAME)
    if (named < 0).any() or (named >= len(tables.name_definitions)).any():
        raise ValueError("invalid decision-memory name row")


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
                [0, kind, value, 0, ordinal, 1]
                for program in source
                for ordinal, (kind, value) in enumerate(program.tokens)
            ],
            np.int32,
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
    counts = references[:, _REFERENCE_DETAIL_COUNT].astype(np.int64)
    shortcut_counts = np.zeros(len(references), np.int64)
    shortcut_starts = np.zeros(len(references), np.int64)
    shortcuts = references[:, _REFERENCE_DEFINITION_ONLY] == 1
    programs = None
    if shortcuts.any():
        programs = _catalog_program_rows()
        definitions = references[shortcuts, _REFERENCE_DEFINITION].astype(np.int64) - 1
        shortcut_counts[shortcuts] = programs.counts[definitions]
        shortcut_starts[shortcuts] = programs.starts[definitions]
    expanded_counts = counts + shortcut_counts
    owners = np.repeat(np.arange(len(references), dtype=np.int64), expanded_counts)
    positions = np.arange(len(owners)) - np.repeat(
        expanded_counts.cumsum() - expanded_counts, expanded_counts
    )
    catalog = positions < shortcut_counts[owners]
    expanded = np.empty((len(owners), 6), np.int32)
    if programs is not None:
        expanded[catalog] = programs.rows[
            shortcut_starts[owners[catalog]] + positions[catalog]
        ]
    explicit = ~catalog
    expanded[explicit] = details[
        references[owners[explicit], _REFERENCE_DETAIL_OFFSET].astype(np.int64)
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
    arrays = dict(zip(MEMORY_FIELDS, (a[None] for a in _node_rows(node)), strict=True))
    references = arrays[_REFERENCES].copy()
    references[..., _REFERENCE_DETAIL_OFFSET] -= cursor.details
    arrays[_REFERENCES] = references
    members = arrays[_MEMBERS].copy()
    members[..., _MEMBER_REFERENCE] -= cursor.references
    arrays[_MEMBERS] = members
    return arrays


def retained_memory(obs: object) -> "PrefixObservation | None":
    from manabot.env.memory_store import PrefixObservation
    from manabot.policy.spans import ObservationSelection

    if type(obs) is ObservationSelection:
        return obs.memory
    return (
        obs
        if isinstance(obs, PrefixObservation)
        and not any(key in obs for key in MEMORY_FIELDS)
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
    if not include_own and (headers[:, _RECORD_CHOOSER] == 1).any():
        raise ValueError("own-seat decision memory is disabled")
    definition_ids(references[:, _REFERENCE_DEFINITION], size, lookup)
    definition_ids(
        record_word(details, _DETAIL_VALUE_HIGH)[
            (details[:, _DETAIL_FIELD] == 0) & (details[:, _DETAIL_TOKEN_KIND] == 7)
        ]
        + 1,
        size,
        lookup,
    )
    validate_memory_vocabulary(
        dict(
            zip(
                MEMORY_FIELDS[:3],
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
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in MEMORY_FIELDS}
    detail_base = 0
    for node in pending:
        local = chunk_observation(node)
        local[_REFERENCES][..., _REFERENCE_DETAIL_OFFSET] += detail_base
        for key in blocks:
            blocks[key].append(local[key])
        detail_base += len(_node_rows(node)[1])
    prepared = _prepare_memory_rows(
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
    empty = _prepare_memory_rows(
        {
            key: np.zeros((0, 0, width), np.float32)
            for key, width in MEMORY_WIDTHS.items()
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
    native: "ObservationSpans | None" = None,
) -> MemoryRows[np.ndarray] | None:
    retained = retained_memory(obs)
    return (
        (
            _prepare_fold(retained, tables, roots)
            if fold
            else _prepare_retained(retained, tables)
        )
        if retained is not None
        else _prepare_memory_rows(obs, tables, native=native)
    )


def _prepare_memory_rows(
    obs: "ObservationBuffers", tables: "Tables", *, native: "ObservationSpans | None" = None
) -> MemoryRows[np.ndarray] | None:
    from manabot.policy.prepare import pack, record_word
    from manabot.policy.spans import full_word

    if native is None:
        if _RECORDS not in obs:
            return None
        headers = obs[_RECORDS]
        lengths = headers[..., _RECORD_VALID].sum(1).astype(np.int64)
        batches, slots = np.nonzero(headers[..., _RECORD_VALID])
        selected = headers[batches, slots]
    else:
        if _RECORDS not in native.tables:
            return None
        spans = native.tables[_RECORDS]
        lengths = spans.counts.astype(np.int64)
        batches, slots = spans.coordinates
        selected = spans.integers

    word = record_word if native is None else full_word

    def word_pairs(rows: np.ndarray, span: slice) -> np.ndarray:
        halves = rows[:, span]
        if native is not None:
            return halves[:, ::2].view(np.uint32).astype(np.int64)
        return record_word(halves.reshape(len(rows), halves.shape[1] // 2, 2), 0)

    temporal = word_pairs(selected, _RECORD_TEMPORAL)
    context = word_pairs(selected, _RECORD_CONTEXT)
    context[:, :2] = context[:, :2].astype(np.uint32).view(np.int32)
    selectors = np.zeros((len(selected), 10), np.float32)
    for index, offset in enumerate(_RECORD_SELECTORS):
        selectors[:, index * 2] = selected[:, offset]
        if offset in _RECORD_NUMBERED_SELECTORS:
            selectors[:, index * 2 + 1] = signed_log(word(selected, offset + 1))
    selectors[:, 8:] = selected[:, _RECORD_DECLARATION]
    numbers = np.concatenate(
        (
            signed_log(temporal),
            selected[:, _RECORD_PHASE_STEP] / [4, 12],
            selected[:, _RECORD_CHOOSER, None] / 2,
            signed_log(context),
            selectors,
        ),
        -1,
    ).astype(np.float32)
    kinds = tables.names(1, word(selected, _RECORD_KIND))
    choices = np.zeros(len(selected), np.int64)
    present = selected[:, _RECORD_CHOICE_PRESENT] == 1
    choices[present] = tables.names(1, word(selected[present, _RECORD_CHOICE_KIND], 0))
    choice_names = word(selected, _RECORD_NAME)
    references_list: list[np.ndarray] = []
    details_list: list[np.ndarray] = []
    detail_routes: list[np.ndarray] = []
    member_routes: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    reference_base = record_base = 0
    for batch, length in enumerate(lengths):
        if native is None:
            _, detail_count, reference_count, member_count = memory_lengths(obs, batch)
            records = obs[_RECORDS][batch, :length]
            references = obs[_REFERENCES][batch, :reference_count]
            raw_details = obs[_DETAILS][batch, :detail_count].astype(np.int32)
            raw_details[:, _DETAIL_VALUE_HIGH] = (
                record_word(raw_details, _DETAIL_VALUE_HIGH)
                .astype(np.uint32)
                .view(np.int32)
            )
            members = obs[_MEMBERS][batch, :member_count]
        else:
            records, raw_details, references, members = (
                native.tables[key].group(batch) for key in MEMORY_FIELDS
            )
        references_list.append(references)
        expanded, owners = _expand_details(references, raw_details)
        details_list.append(expanded)
        detail_routes.append(owners + reference_base)
        member_routes.append(
            members[:, _MEMBER_REFERENCE].astype(np.int64) - 1 + reference_base
        )
        counts = records[:, _RECORD_COUNTS].astype(np.int64).ravel()
        groups.append(
            np.repeat(np.arange(length * 3, dtype=np.int64) + record_base * 3, counts)
        )
        positions.append((members[:, _MEMBER_ORDINAL, None] / 128).astype(np.float32))
        reference_base += len(references)
        record_base += int(length)
    references = (
        np.concatenate(references_list)
        if references_list
        else np.zeros((0, MEMORY_WIDTHS[_REFERENCES]), np.int32)
    )
    details = (
        np.concatenate(details_list)
        if details_list
        else np.zeros((0, MEMORY_WIDTHS[_DETAILS]), np.int32)
    )
    numeric = word_pairs(references, _REFERENCE_WORDS).astype(np.uint32).view(np.int32)
    reference_numbers = np.concatenate(
        (
            references[:, _REFERENCE_ROLE, None] / 2,
            signed_log(numeric),
            references[:, _REFERENCE_FLAGS],
            (references[:, _REFERENCE_DEFINITION, None] > 0).astype(np.float32),
        ),
        -1,
    ).astype(np.float32)
    payloads = full_word(details, _DETAIL_VALUE_HIGH)
    token_kinds = details[:, _DETAIL_TOKEN_KIND].astype(np.int64)
    values = np.zeros(len(details), np.int64)
    program = details[:, _DETAIL_FIELD] == 0
    values[program] = tables.tokens(token_kinds[program], payloads[program])
    signed = payloads.astype(np.uint32).view(np.int32)
    numeric_values = np.where(
        (token_kinds == -1) | np.isin(token_kinds, NUMBER_TOKEN_KINDS),
        signed_log(np.where(program, signed, payloads)),
        0,
    )
    # Field labels have one owner in the native vocabulary. No table indexes
    # or appearance/provenance identifiers enter these feature coordinates.
    # Native stable hashes of DecisionMemory::Program and ::Mana. The lookup
    # is against the supplied checkpoint vocabulary, not a current-world join.
    field_values = np.array([30649973, 1006277364], np.int64)
    fields = tables.names(1, field_values)[details[:, _DETAIL_FIELD].astype(np.int64)]

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
        tables.rows(references[:, _REFERENCE_DEFINITION].astype(np.int64)),
        fields,
        values,
        np.stack(
            (
                token_kinds / 9,
                np.where(program, details[:, _DETAIL_ORDINAL] / 128, 0),
                np.where(
                    program, 0, np.floor_divide(details[:, _DETAIL_ORDINAL], 6) / 128
                ),
                np.where(program, 0, np.remainder(details[:, _DETAIL_ORDINAL], 6) / 5),
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
