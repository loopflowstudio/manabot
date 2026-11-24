"""Host-side index preparation shared by every framework backend.

A policy forward needs integer bookkeeping beside its float arithmetic:
definition-row lookups, record-vocabulary `searchsorted`, legal-action
coordinates, packed GRU schedules and shared-effect CSR routes. That
bookkeeping runs once per batch in NumPy (`prepare`), producing a `Prepared`
batch of float features and integer indexes, so a backend's graph
(`manabot.backends.torch.policy`, `manabot.backends.mlx.policy`) holds only
gathers, scatters and arithmetic.

`Tables` holds the checkpoint's lookup buffers (catalog programs, record
vocabulary, naming joins, ownership template) and the observation layout; it is
read from the checkpoint container (`manabot.policy.checkpoint`), never from
the running engine. This module imports NumPy only.

Inputs must already be admitted (`manabot.policy.admission`): nothing here
validates ids or partitions.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from hashlib import sha256

import numpy as np

from manabot.env.observation import (
    ACTION_CONTINUATION_INDEX,
    ACTION_TYPE_DIM,
    FACT_COMBAT_RESTRICTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_COPY_EXCEPTION,
    FACT_DETAIL_INDEX,
    FACT_DURATION_INDEX,
    FACT_EFFECT,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DEFINITION_INDEX,
    RELATION_VALID_INDEX,
    STACK_DEFINITION_INDEX,
    STACK_IS_MINE_INDEX,
    STACK_SOURCE_INDEX,
    ObservationBuffers,
    column,
    columns,
)
from manabot.env.shared_effects import (
    DETAIL_KIND,
    DETAIL_VALUE,
    EFFECT_LINKS,
    EFFECT_MODIFICATION,
    EFFECT_SOURCE_DEFINITION,
    EFFECT_SPANS,
    EFFECT_VALID,
    MEMBER_OBJECT,
    NUMBER_TOKEN_KINDS,
    PROGRAM_TOKEN_KIND,
    effect_span_rows,
)
from manabot.policy.buckets import BucketedObservation, has_history_events, object_map
from manabot.policy.decision_memory import MemoryRows, prepare_memory
from manabot.policy.readings import (
    DURATION_FLAGS,
    PROGRAM_KINDS,
    RANK_SCALE,
    ColumnReading,
    Reading,
    Readings,
    duration_flags,
    read_numeric,
    reading_plan,
)
from manabot.policy.spans import ObservationSpans, full_word

Array = np.ndarray


class RecordNamespace(IntEnum):
    """Id column families of fact and shared-effect rows (native namespaces)."""

    TOKEN = 1
    NAME = 2
    RESTRICTION = 3


# Fact kinds by how admission and preparation read their id and detail words.
ORDINAL_ID_KINDS = (
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_CONTINUATION_PROGRAM,
    FACT_COMBAT_RESTRICTION,
)
NAMED_DETAIL_KINDS = (FACT_EFFECT, FACT_COPY_EXCEPTION)
_RELATION_OBJECT_COLUMNS = columns("relations", "source", "reference")
_RELATION_DEFINITIONS = columns(
    "relations", "source_definition", "reference_definition"
)
_RELATION_FRAME, _RELATION_FRAME_KIND = columns("relations", "frame", "frame_kind")
_RELATION_FRAME_IDS = slice(_RELATION_FRAME, _RELATION_FRAME_KIND)

_NAME, _NAME_DEFINITION, _GROUP_FIRST, _GROUP_LAST = columns(
    "name_candidates", "name", "definition", "group_first", "group_last"
)
_HAND_DEFINITION, _HAND_COUNT = columns("known_hand", "definition", "count")
_LIBRARY_DEFINITION, _LIBRARY_POSITION = columns(
    "known_library", "definition", "position"
)
_DECK_DEFINITION, _DECK_MAIN = columns("decklists", "definition", "main")

_FACT_VALUE_2 = column("facts", "value_2")

_EVENT_KIND, _EVENT_AMOUNT = columns("events", "kind", "amount")
_EVENT_DEFINITIONS = columns("events", "source_definition", "target_definition")
_EVENT_OWNERS = columns("events", "source_owner", "target_owner")
_EVENT_ZONES = columns("events", "source_zone", "target_zone")
_EVENT_CONTEXT = columns("events", "source_context", "target_context")
_EVENT_STATUS = columns("events", "source_status", "target_status")

# Viewer-relative seat prefixes of the per-seat buffers.
SEATS = ("agent", "opponent")

_PERMANENT_KEYS = ("agent_permanents", "opponent_permanents")
# The two seats' object rows in gather order. V11 appends the stack.
OBJECT_KEYS = (
    "agent_player",
    "opponent_player",
    "agent_cards",
    "opponent_cards",
    *_PERMANENT_KEYS,
)
OBJECT_VALIDITY_KEYS = tuple(f"{key}_valid" for key in OBJECT_KEYS)


# -- checkpoint lookup tables ----------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """How the policy reads its input buffers: one reading per column of each
    buffer's saved prefix (`manabot.policy.readings`). Widths are the lengths."""

    readings: Readings

    def __post_init__(self) -> None:
        for buffer in ("player", "cards", "permanents"):
            agent, opponent = (self.readings.get(f"{seat}_{buffer}") for seat in SEATS)
            if agent != opponent:
                raise ValueError(f"{buffer} readings differ between seats")

    @cached_property
    def widths(self) -> dict[str, int]:
        return {key: len(values) for key, values in self.readings.items()}

    def numeric_width(self, key: str) -> int:
        """How many inputs a buffer's projection receives: the sum of its
        columns' input widths."""
        return reading_plan(self.readings[key]).width

    def fact_width(self, *, catalog_ids: bool) -> int:
        """Inputs of the fact projection: the numeric payloads and timestamp,
        the duration flags, and with catalog ids the ordinal id."""
        readings = self.readings["facts"]
        return (
            self.numeric_width("facts")
            + sum(DURATION_FLAGS.get(reading, 0) for reading in readings)
            + int(catalog_ids)
        )

    @cached_property
    def link_columns(self) -> tuple[int, ...]:
        """Permanent columns joining another object (attached to, blocking)."""
        return tuple(
            index
            for index, reading in enumerate(self.readings["agent_permanents"])
            if reading is Reading.GATHER
        )


@dataclass(frozen=True)
class Tables:
    """The checkpoint's integer buffers, under their PyTorch state-dict names."""

    definition_rows: Array | None  # [ids + 1] transport id -> catalog row + 1 (0: none)
    catalog_tokens: Array  # [S, W] token ids of each definition's program
    catalog_lengths: Array  # [S]
    record_name_keys: Array
    record_name_rows: Array
    record_token_keys: Array
    record_token_rows: Array
    record_number_rows: Array  # [6]
    name_definitions: Array  # [names + 1] definition id + 1 per name (0: none)
    name_program_rows: Array  # [names + 1] catalog row + 1 per name
    is_agent: Array  # [N] bool ownership of each object slot
    layout: Layout
    record_buckets: int = 4096
    # Preserve a backend's existing tie order while sorting only host lengths.
    order_lengths: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        # Memory certificates bind exact immutable lookup bytes, never weights
        # or a mutable backend buffer's Python identity.
        for name in self._MEMORY_LOOKUPS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    np.frombuffer(value.tobytes(), value.dtype).reshape(value.shape),
                )

    _MEMORY_LOOKUPS = (
        "definition_rows",
        "record_name_keys",
        "record_name_rows",
        "record_token_keys",
        "record_token_rows",
        "record_number_rows",
        "name_definitions",
    )

    @cached_property
    def memory_binding(self) -> str:
        from managym.decision_memory import definition_programs

        digest = sha256(str(self.record_buckets).encode())
        digest.update(repr(definition_programs()).encode())
        for name in self._MEMORY_LOOKUPS:
            value = getattr(self, name)
            digest.update(name.encode())
            if value is not None:
                digest.update(str((value.dtype, value.shape)).encode())
                digest.update(value.tobytes())
        return digest.hexdigest()

    @cached_property
    def full_programs(self) -> "CatalogPrograms[Array]":
        return catalog_programs(
            self.catalog_tokens, self.catalog_lengths, order_lengths=self.order_lengths
        )

    def programs(self, rows: Array | None = None) -> "CatalogPrograms[Array]":
        if rows is None:
            return self.full_programs
        return catalog_programs(
            self.catalog_tokens,
            self.catalog_lengths,
            rows,
            order_lengths=self.order_lengths,
        )

    # Torch buffer name of each field.
    BUFFERS = {
        "definition_rows": "semantic_cards.definition_rows",
        "catalog_tokens": "semantic_cards.tokens",
        "catalog_lengths": "semantic_cards.lengths",
        "record_name_keys": "semantic_cards.record_name_keys",
        "record_name_rows": "semantic_cards.record_name_rows",
        "record_token_keys": "semantic_cards.record_token_keys",
        "record_token_rows": "semantic_cards.record_token_rows",
        "record_number_rows": "semantic_cards.record_number_rows",
        "name_definitions": "name_candidates.definitions",
        "name_program_rows": "name_candidates.program_rows",
        "is_agent": "_is_agent_template",
    }

    @classmethod
    def from_buffers(cls, buffers: Mapping[str, Array], layout: Layout) -> "Tables":
        required = {"name_definitions", "is_agent"}
        catalog = {
            "definition_rows",
            "catalog_tokens",
            "catalog_lengths",
            "name_program_rows",
        }
        records = {
            "record_name_keys",
            "record_name_rows",
            "record_token_keys",
            "record_token_rows",
            "record_number_rows",
        }
        for group in (catalog, records):
            if any(cls.BUFFERS[name] in buffers for name in group):
                required.update(group)
        missing = sorted(
            cls.BUFFERS[name] for name in required if cls.BUFFERS[name] not in buffers
        )
        if missing:
            raise KeyError(f"checkpoint lacks lookup buffers {missing}")
        values = {
            field: np.asarray(buffers[key]) if key in buffers else np.empty(0, np.int64)
            for field, key in cls.BUFFERS.items()
        }
        if cls.BUFFERS["definition_rows"] not in buffers:
            values["definition_rows"] = None
        values["is_agent"] = values["is_agent"].reshape(-1).astype(bool)
        return cls(**values, layout=layout)

    def rows(self, ids: Array) -> Array:
        """Catalog rows + 1 of admitted transport ids (`definition_rows`)."""
        return (
            ids.astype(np.int64)
            if self.definition_rows is None
            else self.definition_rows[ids.astype(np.int64)]
        )

    def names(self, namespace: Array | int, words: Array) -> Array:
        """Token-embedding rows of record names (`record_names`)."""
        if not self.catalog_records:
            return words.astype(np.int64) % self.record_buckets
        return _lookup(
            self.record_name_keys, self.record_name_rows, record_key(namespace, words)
        )

    def tokens(self, kinds: Array, words: Array) -> Array:
        """Token-embedding rows of program learning tokens (`record_tokens`)."""
        if not self.catalog_records:
            return words.astype(np.int64) % self.record_buckets
        kinds = kinds.astype(np.int64)
        number = np.isin(kinds, NUMBER_TOKEN_KINDS)
        rows = np.empty_like(kinds)
        rows[number] = self.record_number_rows[kinds[number]]
        if (~number).any():
            rows[~number] = _lookup(
                self.record_token_keys,
                self.record_token_rows,
                record_key(kinds[~number], words[~number]),
            )
        return rows

    @property
    def catalog_records(self) -> bool:
        return self.record_name_keys.size > 0


def record_key(high: Array | int, low: Array) -> Array:
    """One int64 key from a family (namespace or token kind) and a u32 word."""
    return (np.asarray(high, dtype=np.int64) << 32) | (
        low.astype(np.int64) & 0xFFFFFFFF
    )


def _lookup(keys: Array, rows: Array, query: Array) -> Array:
    if query.size == 0:
        return np.zeros(query.shape, dtype=np.int64)
    return rows[np.searchsorted(keys, query).clip(max=len(keys) - 1)]


def record_word(rows: Array, index: int) -> Array:
    """The u32 id at columns `index`, `index + 1` (two 16-bit halves)."""
    return (rows[..., index].astype(np.int64) << 16) | rows[..., index + 1].astype(
        np.int64
    )


# -- packed recurrent schedules ----------------------------------------------------


@dataclass(frozen=True)
class Packed[A]:
    """Variable-length sequences over a flat token table, packed step-major.

    Sequences are sorted by descending length (PyTorch's packing order). Step t
    reads `sizes[t]` live sequences, the prefix of that order. `gather` is the
    flat token of each packed row; `unpack` the packed row of each token;
    `restore` each original sequence's position in the sorted order. `lengths`
    keeps the original sequence boundaries on the host, including after transfer.
    """

    gather: A
    sizes: tuple[int, ...] = field(metadata={"host_only": True})
    unpack: A
    restore: A
    lengths: tuple[int, ...] = field(metadata={"host_only": True})


def pack(
    lengths: Sequence[int] | Array,
    order_lengths: Callable[[Array], Array] | None = None,
) -> Packed[Array]:
    """Schedule sequences of `lengths`; `order_lengths` returns a descending
    order when a backend's tie order differs from the stable default."""
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    order = (
        np.argsort(-lengths, kind="stable")
        if order_lengths is None
        else order_lengths(lengths)
    )
    ordered = lengths[order]
    sizes: list[int] = []
    gather: list[Array] = []
    for step in range(int(lengths.max(initial=0))):
        live = order[: int((ordered > step).sum())]
        sizes.append(len(live))
        gather.append(offsets[live] + step)
    flat = np.concatenate(gather) if gather else np.zeros(0, np.int64)
    unpack = np.empty_like(flat)
    unpack[flat] = np.arange(len(flat))
    restore = np.empty(len(order), dtype=np.int64)
    restore[order] = np.arange(len(order))
    return Packed(flat, tuple(sizes), unpack, restore, tuple(int(n) for n in lengths))


@dataclass(frozen=True)
class CatalogPrograms[A]:
    """Packed definition programs and their one-based output rows."""

    rows: A
    tokens: A
    schedule: Packed[A]


def catalog_programs(
    tokens: Array,
    lengths: Array,
    rows: Array | None = None,
    *,
    order_lengths: Callable[[Array], Array] | None = None,
) -> CatalogPrograms[Array]:
    if rows is None:
        rows = np.arange(1, len(lengths) + 1, dtype=np.int64)
    selected_lengths = lengths[rows - 1]
    schedule = pack(selected_lengths, order_lengths)
    if len(rows):
        flat = np.concatenate(
            [
                tokens[row - 1, :length]
                for row, length in zip(rows, selected_lengths, strict=True)
            ]
        )
    else:
        flat = np.zeros(0, np.int64)
    return CatalogPrograms(rows, flat[schedule.gather], schedule)


# -- prepared batches --------------------------------------------------------------


@dataclass(frozen=True)
class FactPrograms[A]:
    """The program subset of a fact table, absent as a whole when none is live."""

    rows: A  # [T] flat fact row of each program token
    numbers: A  # [T] signed integers; log1p stays in the backend
    positions: A  # [B*F] token position of each fact row
    mask: A  # [B*F] bool
    schedule: Packed[A]


@dataclass(frozen=True)
class FactRoutes[A]:
    kinds: A  # [B,F]
    columns: A  # [B,F,6] numeric columns
    id_rows: A  # [B,F] token rows of named ids
    id_mask: A  # [B,F]
    detail_rows: A
    detail_mask: A
    token_rows: A  # program token rows
    token_mask: A
    source_rows: A  # [B,F] catalog row + 1
    targets: A  # [B*F] flat object index
    linked: A  # [B,F] float
    programs: FactPrograms[A] | None


@dataclass(frozen=True)
class EffectRoutes[A]:
    numeric: A  # [L,8]
    controller: A  # [L] flat object index
    controller_mask: A
    source_object: A
    source_object_mask: A
    modification_rows: A  # [L]
    source_rows: A  # [L] catalog row + 1
    detail_rows: A  # [Dt]
    detail_kinds: A  # [Dt]
    detail_numbers: A  # [Dt] signed integers, before backend log1p
    detail_is_number: A  # [Dt] float
    token_indexes: A  # [sum lengths] rows of the header+detail token table
    schedule: Packed[A]
    routes: A  # [M] flat object index
    member_records: A  # [M]


@dataclass(frozen=True)
class BeliefRows[A]:
    """Explicit belief inputs; no hidden state persists between forwards."""

    cards: A
    owners: A
    zones: A
    probabilities: A
    validity: A
    globals: A


@dataclass(frozen=True)
class EventRows[A]:
    """Public history joins and integer ages; log1p stays in the backend."""

    kinds: A
    amounts: A
    ages: A
    validity: A
    counts: A
    definitions: A
    owners: A
    zones: A
    focus: A
    focus_mask: A
    status: A


@dataclass(frozen=True)
class Prepared[A]:
    """One admitted observation batch, ready for a backend policy."""

    batch: int
    agent_player: A
    opponent_player: A
    agent_cards: A
    opponent_cards: A
    permanents: A  # [B,2,P,numeric]
    validity: A  # [B,N]
    card_rows: A  # [B,2,C]
    known_rows: A  # [B,2,K]
    known_counts: A  # [B,2,K] raw counts; divide after the embedding product
    library_rows: A  # [B,2,L], catalog definitions at known positions
    library_weights: A  # [B,2,L], reciprocal one-based position; zero padding
    deck_rows: A  # [B,2,R]
    deck_counts: A  # [B,2,R,4] already / 60
    permanent_rows: A  # [B,2,P]
    link_targets: tuple[A, ...]  # per link column [B*2P] flat object index
    link_masks: tuple[A, ...]  # [B,2P] float
    facts: FactRoutes[A]
    effects: EffectRoutes[A] | None
    relation_features: A  # [B,R,44]
    relation_valid: A  # [B,R]
    relation_rows: A  # [B,R,2] catalog rows + 1
    relation_objects: tuple[A, ...]  # per object column [B*R] flat index
    relation_object_masks: tuple[A, ...]
    frame_indexes: A  # [B*R] flat frame slot
    # Legal actions, row-major over [B,A].
    legal: A  # [B,A] bool
    legal_batches: A  # [K]
    legal_slots: A  # [K] original action coordinates
    logit_positions: A  # [B,A] index into K (0 where illegal)
    action_features: A  # [K, numeric action columns]
    action_frames: A  # [K] flat frame slot
    action_frame_mask: A  # [K] float
    name_rows: A  # [K]
    name_mask: A
    group_first: A
    group_last: A
    group_mask: A
    group_counts: A
    candidate_rows: A  # [K] catalog row + 1
    focus: A  # [K,F] flat object index
    focus_mask: A  # [K,F] float
    is_agent: A  # [B,N] includes explicit belief ownership
    # Inference only: object columns any row uses (None: attend over all).
    attention_columns: A | None = None
    # Configuration identity, independent of a particular model instance.
    binding: str = ""
    programs: CatalogPrograms[A] | None = None
    belief: BeliefRows[A] | None = None
    events: EventRows[A] | None = None
    memory: MemoryRows[A] | None = None
    live_index: A | None = None  # flat valid slots, including memory/value tokens
    live_offsets: A | None = None  # [B+1] cumulative live-token counts
    legal_support: bool = True  # every observation has a legal action
    # A serving scope may bound this batch's lifetime without changing its rows.
    guard: Callable[[object], None] | None = None
    bucket_needs: np.ndarray | None = field(default=None, metadata={"host_only": True})
    # Input v11 appends one shared stack segment before optional belief rows.
    stack: A | None = None
    stack_rows: A | None = None
    stack_targets: A | None = None
    stack_mask: A | None = None


def object_slots(values: Array, remap: Array | None, *, base: int = 1) -> Array:
    """Zero-based gather slots. An absent join gathers player zero under its mask."""
    slots = (values.astype(np.int64, copy=False) - base).clip(min=0)
    return slots if remap is None else remap[slots]


def object_ownership(cards: int, permanents: int) -> np.ndarray:
    """Ownership of the two players and their card and permanent slots."""
    return np.repeat([True, False] * 3, [1, 1, cards, cards, permanents, permanents])


def padded_rows(values: Array, selected: Array) -> Array:
    """Restore gathered rows to their original coordinates, zero elsewhere.

    Packed native spans list each row's slots in order, so they scatter with
    the same row-major mask as rows gathered from a dense buffer."""
    out = np.zeros((*selected.shape, *values.shape[1:]), dtype=values.dtype)
    out[selected] = values
    return out


def read_object_numeric(values: Array, readings: Sequence[ColumnReading]) -> Array:
    """Skip zero rows, which every object reading maps to zero. An external
    observation can carry numeric data even in a masked object slot."""
    present = values.any(axis=-1)
    if present.all():
        return read_numeric(values, readings)
    return padded_rows(read_numeric(values[present], readings), present)


def prepare(
    obs: ObservationBuffers | BucketedObservation | ObservationSpans,
    tables: Tables,
    *,
    inference: bool = False,
    bucketed: bool = False,
    live_tokens: bool = False,
    value_token: bool = False,
    memory_summaries: int = 0,
    memory_fold: bool = False,
    memory_roots: Mapping[str, int] | None = None,
    binding: str = "",
    select_programs: bool = False,
) -> Prepared[Array]:
    """Build every index a backend forward reads from one admitted batch.

    `inference` enables CPU inference compaction: unused fact columns are
    dropped unless bucketed, and unused attention columns unless live tokens
    are packed. Bucketed observations retain their original action capacity.
    Counts and signed integers stay raw where backend arithmetic determines
    rounding. The input belongs to `binding`, not to a particular model instance.
    """
    action_capacity = None
    bucket_needs = object_remap = None
    # Dense `ObservationBuffers` / `BucketedObservation` inputs are deprecated and
    # removed in 0.3.0 once `ObservationSpans` is the sole policy input.
    native = obs if isinstance(obs, ObservationSpans) else None
    if isinstance(obs, ObservationSpans):
        if obs.buckets is not None:
            bucketed = True
            bucket_needs = obs.needs
            object_remap = object_map(obs.buckets)
            action_capacity = obs.buckets.capacities["actions"]
        # Fact and effect routing consumes packed values directly.
        packed_keys = {
            "facts",
            "effects",
            "effect_details",
            "effect_members",
            "actions",
            "action_focus",
            "name_candidates",
            "agent_cards",
            "opponent_cards",
        }
        dense = {}
        for key, span in obs.tables.items():
            if key in packed_keys or key.startswith("decision_"):
                continue
            if key in ("agent_player", "opponent_player", "player_turn_state"):
                values = span.features(tables.layout.readings[key])
            else:
                values = span.integers
            dense[key] = span.scatter(values, obs.capacities[key])
        for key in (*OBJECT_KEYS, "stack", "actions", "events"):
            dense[key + "_valid"] = obs.dense(key + "_valid")
        obs = dense
    elif isinstance(obs, BucketedObservation):
        bucketed = True
        bucket_needs = obs.needs
        object_remap = object_map(obs.buckets)
        action_capacity = obs.buckets.capacities["actions"]
        obs = obs.rows
    memory = (
        prepare_memory(
            native.memory if native is not None and native.memory is not None else obs,
            tables,
            fold=memory_fold,
            roots=memory_roots,
            native=native,
        )
        if memory_summaries
        else None
    )
    obs = {key: np.asarray(value) for key, value in obs.items()}
    history = has_history_events(obs)
    for key, width in tables.layout.widths.items():
        if key in obs and obs[key].shape[-1] > width:
            obs[key] = obs[key][..., :width]
    if native is not None:
        span = native.tables["facts"]
        stop = (
            int(span.counts.max(initial=0))
            if inference and not bucketed
            else native.capacities["facts"]
        )
        fact_valid = span.mask(stop)
        live_facts = span.integers[:, : tables.layout.widths["facts"]]
    else:
        facts = obs["facts"]
        if inference and not bucketed:
            facts = facts[:, (facts[..., FACT_VALID_INDEX] == 1).any(0)]
        fact_valid = facts[..., FACT_VALID_INDEX] == 1
        live_facts = facts[fact_valid]
    stack = obs.get("stack")
    validity_rows = [obs[key] for key in OBJECT_VALIDITY_KEYS]
    if stack is not None:
        validity_rows.append(obs["stack_valid"])
    belief = _belief(obs)
    if belief is not None:
        validity_rows.extend(
            (belief.validity, np.ones((len(belief.validity), 1), np.float32))
        )
    validity = np.concatenate(validity_rows, 1)
    columns = None
    if inference and not live_tokens:
        if native is None:
            used = np.nonzero((validity != 0).any(0))[0]
        else:
            start = 0
            segments = []
            for key in (*OBJECT_KEYS, "stack"):
                segments.append(
                    start + np.arange(native.tables[key].counts.max(initial=0))
                )
                start += native.capacities[key]
            used = np.concatenate(segments)
        if 0 < len(used) < validity.shape[1]:
            columns = used
    layout = tables.layout
    batch = obs["actions_valid"].shape[0]
    objects = validity.shape[1]
    f32 = np.float32
    permanents = np.stack([obs[key] for key in _PERMANENT_KEYS], axis=1)

    def object_features(key: str) -> Array:
        if native is None:
            return read_object_numeric(obs[key], layout.readings[key])
        spans = native.tables[key]
        values = spans.features(layout.readings[key])
        return spans.scatter(
            read_numeric(values, layout.readings[key]), native.capacities[key]
        )

    numeric = (
        read_object_numeric(permanents, layout.readings["agent_permanents"])
        if native is None
        else np.stack([object_features(key) for key in _PERMANENT_KEYS], axis=1)
    )
    knowledge = obs["known_hand"]
    decklists = obs["decklists"]
    # Deprecated v9/v10 carry no positional rows. Current input must supply
    # the admitted buffer; missing rows never become an empty-knowledge claim.
    library = (
        obs["known_library"]
        if stack is not None
        else np.zeros((batch, 2, 0, 2), dtype=f32)
    )
    link_targets: list[Array] = []
    link_masks: list[Array] = []
    base = np.arange(batch)[:, None] * objects
    for link_column in layout.link_columns:
        links = np.concatenate(
            [obs[key][..., link_column] for key in _PERMANENT_KEYS], 1
        )
        link_targets.append((base + object_slots(links, object_remap)).reshape(-1))
        link_masks.append((links > 0).astype(f32))

    relations = obs["relations"]
    valid = (
        relations[..., RELATION_VALID_INDEX] == 1
        if native is None
        else native.tables["relations"].mask(relations.shape[1])
    )
    relation_valid = relations[..., RELATION_VALID_INDEX]
    live = (
        relations[valid]
        if native is None
        else native.tables["relations"].integers[:, : relations.shape[-1]].copy()
    )
    pending = np.isin(live[:, _RELATION_FRAME_KIND], (7, 8, 10))
    live[pending, _RELATION_FRAME_IDS] = 0
    relation_features = padded_rows(
        read_numeric(live, layout.readings["relations"]), valid
    )
    relation_count = relations.shape[1]
    relation_objects: list[Array] = []
    relation_masks: list[Array] = []
    for link_column in _RELATION_OBJECT_COLUMNS:
        indexes = np.where(valid, relations[..., link_column], 0)
        relation_objects.append(
            (base + object_slots(indexes, object_remap)).reshape(-1)
        )
        relation_masks.append((indexes > 0).astype(f32))
    frame_ids = np.where(valid, relations[..., _RELATION_FRAME], 1).astype(np.int64)
    frame_base = np.arange(batch)[:, None] * relation_count
    frame_indexes = (frame_base + (frame_ids - 1).clip(min=0)).reshape(-1)

    legal = obs["actions_valid"] > 0
    if action_capacity is not None and action_capacity != legal.shape[1]:
        legal = np.pad(legal, ((0, 0), (0, action_capacity - legal.shape[1])))
    legal_batches, legal_slots = np.nonzero(legal)
    positions = np.zeros(legal.shape, dtype=np.int64)
    positions[legal_batches, legal_slots] = np.arange(len(legal_batches))
    if native is None:
        action_rows = obs["actions"][legal_batches, legal_slots]
        names = obs["name_candidates"][legal_batches, legal_slots]
        focus = obs["action_focus"][legal_batches, legal_slots]
    else:
        action_rows = native.tables["actions"].features(layout.readings["actions"])
        names = native.tables["name_candidates"].integers[
            :, : layout.widths["name_candidates"]
        ]
        focus = native.tables["action_focus"].integers
    action_features = read_numeric(action_rows, layout.readings["actions"])
    frames = (
        action_rows[:, ACTION_CONTINUATION_INDEX]
        if native is None
        else native.tables["actions"].records["continuation"]
    ).astype(np.int64, copy=False)
    candidates = names.astype(np.int64)
    focus = focus.astype(np.int64)
    if candidates.shape[-1] > max(_GROUP_FIRST, _GROUP_LAST):
        first, last = candidates[:, _GROUP_FIRST], candidates[:, _GROUP_LAST]
    else:
        first = last = np.zeros(len(candidates), np.int64)

    players = [
        read_numeric(obs[f"{seat}_player"], layout.readings[f"{seat}_player"])
        for seat in SEATS
    ]
    cards = [object_features(f"{seat}_cards") for seat in SEATS]
    if "player_turn_state" in obs:
        turn_state = read_numeric(
            obs["player_turn_state"], layout.readings["player_turn_state"]
        )
        players = [
            np.concatenate((row, turn_state[:, seat : seat + 1]), -1)
            for seat, row in enumerate(players)
        ]
        cards = [
            np.concatenate((row, obs["card_turn_state"][:, seat, :, None]), -1)
            for seat, row in enumerate(cards)
        ]
    c, p = cards[0].shape[1], permanents.shape[2]
    seat_ownership = object_ownership(c, p)
    ownership = [np.broadcast_to(seat_ownership, (batch, len(seat_ownership)))]
    if stack is not None:
        ownership.append(stack[..., STACK_IS_MINE_INDEX] > 0)
    if belief is not None:
        ownership.extend((belief.owners == 0, np.zeros((batch, 1), bool)))
    live_index = live_offsets = None
    if live_tokens:
        if native is None:
            present = validity != 0
            if value_token:
                present = np.concatenate((present, np.ones((batch, 1), bool)), 1)
            live_index = np.flatnonzero(present)
            lengths = present.sum(1)
        else:
            # An object's padded index is its packed index plus the number
            # of unused slots in all preceding (batch, segment) groups.
            keys = (*OBJECT_KEYS, "stack")
            counts = np.column_stack([native.tables[key].counts for key in keys])
            capacities = np.array([native.capacities[key] for key in keys])
            if value_token:
                counts = np.column_stack((counts, np.ones(batch, dtype=np.int64)))
                capacities = np.append(capacities, 1)
            gaps = (capacities - counts).ravel()
            skipped = np.cumsum(gaps) - gaps
            lengths = counts.sum(1)
            live_index = np.arange(lengths.sum()) + np.repeat(skipped, counts.ravel())
        live_offsets = np.concatenate(([0], lengths.cumsum()))
    return Prepared(
        batch=batch,
        legal_support=bool(legal.any(-1).all()),
        agent_player=players[0],
        opponent_player=players[1],
        agent_cards=cards[0],
        opponent_cards=cards[1],
        permanents=numeric,
        validity=validity.astype(f32),
        card_rows=tables.rows(obs["semantic_cards"]),
        known_rows=tables.rows(knowledge[..., _HAND_DEFINITION]),
        known_counts=knowledge[..., _HAND_COUNT].astype(f32),
        library_rows=tables.rows(library[..., _LIBRARY_DEFINITION]),
        library_weights=np.divide(
            np.float32(1),
            library[..., _LIBRARY_POSITION],
            out=np.zeros(library.shape[:-1], dtype=f32),
            where=library[..., _LIBRARY_POSITION] > 0,
        ).astype(f32),
        deck_rows=tables.rows(decklists[..., _DECK_DEFINITION]),
        deck_counts=(decklists[..., _DECK_MAIN:] / f32(60)).astype(f32),
        permanent_rows=tables.rows(permanents[..., PERMANENT_DEFINITION_INDEX]),
        stack=object_features("stack") if stack is not None else None,
        stack_rows=tables.rows(stack[..., STACK_DEFINITION_INDEX])
        if stack is not None
        else None,
        stack_targets=(
            base + object_slots(stack[..., STACK_SOURCE_INDEX], object_remap)
        ).reshape(-1)
        if stack is not None
        else None,
        stack_mask=(stack[..., STACK_SOURCE_INDEX] > 0).astype(f32)
        if stack is not None
        else None,
        link_targets=tuple(link_targets),
        link_masks=tuple(link_masks),
        facts=_facts(
            live_facts, fact_valid, tables, objects, object_remap, native=native
        ),
        effects=_effects(obs, tables, objects, object_remap, native=native),
        relation_features=relation_features,
        relation_valid=relation_valid.astype(f32),
        relation_rows=tables.rows(relations[..., _RELATION_DEFINITIONS]),
        relation_objects=tuple(relation_objects),
        relation_object_masks=tuple(relation_masks),
        frame_indexes=frame_indexes,
        legal=legal,
        legal_batches=legal_batches.astype(np.int64),
        legal_slots=legal_slots.astype(np.int64),
        logit_positions=positions,
        action_features=action_features.astype(f32),
        action_frames=legal_batches * relation_count + (frames - 1).clip(min=0),
        action_frame_mask=(frames != 0).astype(f32),
        name_rows=candidates[:, _NAME],
        name_mask=(candidates[:, _NAME] > 0).astype(f32),
        group_first=(first - 1).clip(min=0),
        group_last=last,
        group_mask=(first > 0).astype(f32),
        group_counts=(last - first + 1).clip(min=1).astype(f32),
        candidate_rows=tables.rows(candidates[:, _NAME_DEFINITION]),
        focus=legal_batches[:, None] * objects
        + object_slots(focus, object_remap, base=0),
        focus_mask=(focus != -1).astype(f32),
        attention_columns=columns,
        binding=binding,
        bucket_needs=bucket_needs,
        is_agent=np.concatenate(ownership, axis=1),
        # Full-catalog preparation skips the scan for referenced definitions.
        programs=tables.programs(
            _referenced_definitions(
                obs, tables, candidates, history=history, memory=memory, native=native
            )
            if select_programs
            else None
        )
        if tables.definition_rows is not None
        else None,
        belief=belief,
        events=_events(obs, tables, objects, object_remap) if history else None,
        memory=memory,
        live_index=live_index,
        live_offsets=live_offsets,
    )


def _referenced_definitions(
    obs: ObservationBuffers,
    tables: Tables,
    candidates: Array,
    *,
    history: bool,
    memory: MemoryRows[Array] | None,
    native: ObservationSpans | None = None,
) -> Array:
    """Catalog subset read by learning, in sorted row order (zero is padding)."""
    references = [
        obs["semantic_cards"],
        *(obs[key][..., PERMANENT_DEFINITION_INDEX] for key in _PERMANENT_KEYS),
        obs["known_hand"][..., _HAND_DEFINITION],
        obs["decklists"][..., _DECK_DEFINITION],
        obs["relations"][..., _RELATION_DEFINITIONS],
        candidates[:, _NAME_DEFINITION],
    ]
    if "known_library" in obs:
        references.append(obs["known_library"][..., _LIBRARY_DEFINITION])
    if history:
        references.extend(
            obs["events"][..., col][obs["events_valid"] == 1]
            for col in _EVENT_DEFINITIONS
        )
    if "stack" in obs:
        references.append(obs["stack"][..., STACK_DEFINITION_INDEX])
    if tables.catalog_records:
        for key, source, valid in (
            ("facts", FACT_SOURCE_INDEX, FACT_VALID_INDEX),
            ("effects", EFFECT_SOURCE_DEFINITION, EFFECT_VALID),
        ):
            references.append(
                native.tables[key].integers[:, source]
                if native is not None
                else obs[key][..., source][obs[key][..., valid] == 1]
            )
    referenced = np.unique(
        tables.rows(np.concatenate([ids.reshape(-1) for ids in references]))
    )
    if memory is not None:
        referenced = np.union1d(referenced, memory.reference_definitions)
    return referenced[referenced > 0]


def _facts(
    live: Array,
    valid: Array,
    tables: Tables,
    objects: int,
    object_remap: Array | None = None,
    *,
    native: ObservationSpans | None = None,
) -> FactRoutes[Array]:
    """Route live fact rows; `valid` places them in their padded slots.

    Index only live facts on the host. Restore padding before the forward:
    its matrix shapes and gradient reduction order must remain unchanged."""
    word = record_word if native is None else full_word
    batch = len(valid)

    kinds = live[..., FACT_KIND_INDEX].astype(np.int64)
    ident, detail = (
        word(live, FACT_ID_INDEX),
        word(live, FACT_DETAIL_INDEX),
    )
    ordinal = np.isin(kinds, ORDINAL_ID_KINDS)
    # Numeric columns in parameter order: the two payloads, the duration flags,
    # the timestamp, then the ordinal id (`fact_id`), / 128 like a rank.
    readings = tables.layout.readings["facts"]
    payload = slice(FACT_VALUE_INDEX, FACT_VALUE_INDEX + 2)
    timestamp = slice(FACT_ORDER_INDEX, FACT_ORDER_INDEX + 1)
    values = native.fact_payloads(live) if native is not None else live[..., payload]
    payloads = read_numeric(values, readings[payload], kinds)
    order = read_numeric(live[..., timestamp], readings[timestamp])
    columns = np.concatenate(
        [
            payloads,
            duration_flags(
                live[..., FACT_DURATION_INDEX], readings[FACT_DURATION_INDEX]
            ),
            order,
            (np.where(ordinal, ident, 0).astype(np.float32) / np.float32(RANK_SCALE))[
                :, None
            ],
        ],
        -1,
    )
    named = (kinds > 0) & ~ordinal
    program = np.isin(kinds, PROGRAM_KINDS)
    restriction = kinds == FACT_COMBAT_RESTRICTION
    named_detail = (np.isin(kinds, NAMED_DETAIL_KINDS) & (detail != 0)) | restriction
    id_space = np.where(
        kinds == FACT_SET_NAME, RecordNamespace.NAME, RecordNamespace.TOKEN
    )
    detail_space = np.where(
        restriction, RecordNamespace.RESTRICTION, RecordNamespace.TOKEN
    )

    token_kind = live[..., FACT_VALUE_INDEX].astype(np.int64)
    token_rows = np.zeros(kinds.shape, np.int64)
    if tables.catalog_records:
        id_rows = np.zeros(kinds.shape, np.int64)
        detail_rows = np.zeros_like(id_rows)
        id_rows[named] = tables.names(id_space[named], ident[named])
        detail_rows[named_detail] = tables.names(
            detail_space[named_detail], detail[named_detail]
        )
        token_rows[program] = tables.tokens(token_kind[program], detail[program])
        sources = tables.rows(live[..., FACT_SOURCE_INDEX])
        id_mask = padded_rows(named.astype(np.float32), valid)
        detail_mask = padded_rows(named_detail.astype(np.float32), valid)
    else:
        # Torch linear chooses a different accumulation path for strided inputs.
        columns = np.ascontiguousarray(columns[..., :-1])
        id_rows = ident % tables.record_buckets
        detail_rows = detail % tables.record_buckets
        sources = live[..., FACT_SOURCE_INDEX].astype(np.int64) % tables.record_buckets
        # Hashed records embed every row, padding included.
        id_mask = np.ones(valid.shape, np.float32)
        detail_mask = np.ones(valid.shape, np.float32)
    links = padded_rows(live[..., FACT_OBJECT_INDEX].astype(np.int64), valid)
    base = np.arange(batch)[:, None] * objects
    all_programs = padded_rows(program, valid)
    return FactRoutes(
        kinds=padded_rows(kinds, valid),
        columns=padded_rows(columns, valid),
        id_rows=padded_rows(id_rows, valid),
        id_mask=id_mask,
        detail_rows=padded_rows(detail_rows, valid),
        detail_mask=detail_mask,
        token_rows=padded_rows(token_rows, valid),
        token_mask=all_programs.astype(np.float32),
        source_rows=padded_rows(sources, valid),
        targets=(base + object_slots(links, object_remap)).reshape(-1),
        linked=(valid & (links > 0)).astype(np.float32),
        programs=_fact_programs(
            live[program],
            detail[program],
            all_programs,
            np.flatnonzero(valid)[program],
            tables.order_lengths,
        ),
    )


def _fact_programs(
    program_facts: Array,
    word: Array,
    program: Array,
    rows: Array,
    order_lengths: Callable[[Array], Array] | None = None,
) -> FactPrograms[Array] | None:
    """Preserve admitted program boundaries and raw integers for the backend GRU."""
    flat = program.reshape(-1)
    if not len(rows):
        return None
    starts = np.nonzero(program_facts[:, _FACT_VALUE_2] == 0)[0]
    lengths = np.diff(np.append(starts, len(rows)))
    signed = np.where(word >= 2**31, word - 2**32, word).astype(np.float32)
    kind = program_facts[:, FACT_VALUE_INDEX]
    numbers = np.where((kind == 4) | (kind == 5), signed, 0)
    positions = np.zeros(flat.size, dtype=np.int64)
    positions[rows] = np.arange(len(rows))
    return FactPrograms(
        rows=rows,
        numbers=numbers,
        positions=positions,
        mask=flat,
        schedule=pack(lengths, order_lengths),
    )


def _effects(
    obs: ObservationBuffers,
    tables: Tables,
    objects: int,
    object_remap: Array | None = None,
    *,
    native: ObservationSpans | None = None,
) -> EffectRoutes[Array] | None:
    if native is None:
        headers = obs["effects"]
        live_batches, live_slots = np.nonzero(headers[..., EFFECT_VALID] == 1)
        selected = headers[live_batches, live_slots]
    else:
        spans = native.tables["effects"]
        live_batches, _ = spans.coordinates
        selected = spans.integers[:, : tables.layout.widths["effects"]]
    if not len(selected):
        return None
    live = selected.astype(np.int64)
    modification = (
        record_word(live, EFFECT_MODIFICATION[0])
        if native is None
        else full_word(selected, EFFECT_MODIFICATION[0])
    )
    source = live[:, EFFECT_SOURCE_DEFINITION]
    links = live[:, EFFECT_LINKS]
    detail_offsets, counts, member_offsets, member_counts = live[:, EFFECT_SPANS].T
    if native is None:
        detail_records, detail_rows = effect_span_rows(detail_offsets, counts)
        detail_values = obs["effect_details"][live_batches[detail_records], detail_rows]
    else:
        detail_values = native.tables["effect_details"].integers
    detail_host = detail_values.astype(np.int64)
    kinds = detail_host[:, DETAIL_KIND]
    records, member_rows = effect_span_rows(member_offsets, member_counts)
    member_batches = live_batches[records]
    targets = (
        obs["effect_members"][member_batches, member_rows, MEMBER_OBJECT]
        if native is None
        else native.tables["effect_members"].integers[:, MEMBER_OBJECT]
    ).astype(np.int64)

    # Each sequence reads its three header tokens followed by its detail tokens.
    token_indexes: list[int] = []
    detail_start = 3 * len(live)
    for i, count in enumerate(counts.tolist()):
        token_indexes.extend(range(3 * i, 3 * i + 3))
        token_indexes.extend(range(detail_start, detail_start + count))
        detail_start += count
    effect_readings = tables.layout.readings["effects"]
    if native is not None:
        selected = spans.features(effect_readings)
    numeric = read_numeric(selected, effect_readings)
    base = live_batches * objects
    unsigned = (
        record_word(detail_host, DETAIL_VALUE[0])
        if native is None
        else full_word(detail_values, DETAIL_VALUE[0])
    )
    token_kinds = PROGRAM_TOKEN_KIND[kinds]
    program = token_kinds != 0
    detail_embedding_rows = np.zeros(len(kinds), dtype=np.int64)
    if (~program).any():
        detail_embedding_rows[~program] = tables.names(
            RecordNamespace.TOKEN, unsigned[~program]
        )
    if program.any():
        detail_embedding_rows[program] = tables.tokens(
            token_kinds[program], unsigned[program]
        )
    signed = np.where(unsigned >= 2**31, unsigned - 2**32, unsigned)
    return EffectRoutes(
        numeric=numeric,
        controller=base + object_slots(links[:, 0], object_remap),
        controller_mask=(links[:, 0] > 0).astype(np.float32),
        source_object=base + object_slots(links[:, 1], object_remap),
        source_object_mask=(links[:, 1] > 0).astype(np.float32),
        modification_rows=tables.names(RecordNamespace.TOKEN, modification),
        source_rows=tables.rows(source)
        if tables.catalog_records
        else source % tables.record_buckets,
        detail_rows=detail_embedding_rows,
        detail_kinds=kinds,
        detail_numbers=signed.astype(np.float32),
        detail_is_number=np.isin(token_kinds, NUMBER_TOKEN_KINDS).astype(np.float32),
        token_indexes=np.asarray(token_indexes, dtype=np.int64),
        schedule=pack(3 + counts),
        routes=member_batches * objects + object_slots(targets, object_remap),
        member_records=records,
    )


def _belief(obs: ObservationBuffers) -> BeliefRows[Array] | None:
    if "belief_card_def_ids" not in obs:
        return None
    return BeliefRows(
        cards=obs["belief_card_def_ids"].astype(np.int64),
        owners=obs["belief_owner_role_ids"].astype(np.int64),
        zones=obs["belief_hidden_zone_ids"].astype(np.int64),
        probabilities=obs["belief_count_probabilities"].astype(np.float32),
        validity=obs["belief_validity"].astype(np.float32),
        globals=obs["belief_globals"].astype(np.float32),
    )


def _events(
    obs: ObservationBuffers,
    tables: Tables,
    objects: int,
    object_remap: Array | None = None,
) -> EventRows[Array]:
    valid = obs["events_valid"]
    rows = np.where((valid != 0)[..., None], obs["events"], 0)
    counts = valid.sum(1, keepdims=True)
    focus = rows[..., _EVENT_CONTEXT].astype(np.int64)
    return EventRows(
        kinds=rows[..., _EVENT_KIND].astype(np.int64),
        amounts=rows[..., _EVENT_AMOUNT, None].astype(np.float32) / np.float32(20),
        ages=(counts - valid.cumsum(1)).clip(min=0).astype(np.float32),
        validity=valid.astype(bool),
        counts=counts.astype(np.float32),
        definitions=tables.rows(rows[..., _EVENT_DEFINITIONS]),
        owners=rows[..., _EVENT_OWNERS].astype(np.int64),
        zones=rows[..., _EVENT_ZONES].astype(np.int64),
        focus=np.arange(len(rows))[:, None, None] * objects
        + object_slots(focus, object_remap, base=0),
        focus_mask=(focus >= 0) & (valid != 0)[..., None],
        status=rows[..., _EVENT_STATUS].astype(np.int64),
    )


def reference_distribution(obs: ObservationBuffers, kind: str) -> Array:
    """The full-support magnet [B,A] float32 a reverse KL pulls toward.

    `"uniform"` spreads mass evenly over the legal offers; `"action_type_uniform"`
    spreads it evenly over the legal action types, then over offers within a
    type. Reads `actions_valid` [B,A] and `actions` [B,A,F]; every row needs a
    legal offer.
    """
    if kind not in {"uniform", "action_type_uniform"}:
        raise ValueError(f"unknown reference: {kind}")
    valid = np.asarray(obs["actions_valid"]) > 0
    if not valid.any(-1).all():
        raise ValueError("reference requires at least one legal offer per row")
    if kind == "uniform":
        return (valid / valid.sum(-1, keepdims=True)).astype(np.float32)
    action_types = np.asarray(obs["actions"])[..., :ACTION_TYPE_DIM]
    types = action_types.argmax(-1)
    # Legal offers of each type per row, [B,types]: exact small integers.
    width = action_types.shape[-1]
    slots = types + width * np.arange(valid.shape[0])[:, None]
    counts = np.bincount(
        slots.ravel(), weights=valid.ravel(), minlength=valid.shape[0] * width
    )
    counts = counts.reshape(valid.shape[0], width).astype(np.float32)
    weights = valid / np.take_along_axis(counts, types, -1).clip(min=1)
    return (weights / weights.sum(-1, keepdims=True)).astype(np.float32)
