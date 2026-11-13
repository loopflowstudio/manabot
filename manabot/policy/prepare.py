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
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property

import numpy as np

from manabot.env.observation import (
    ACTION_TYPE_DIM,
    FACT_COMBAT_RESTRICTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_COPY_EXCEPTION,
    FACT_DETAIL_INDEX,
    FACT_DURATION_INDEX,
    FACT_EFFECT,
    FACT_EFFECT_PROGRAM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_TIMESTAMP_SCALE,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DEFINITION_INDEX,
    RELATION_VALID_INDEX,
    ObservationBuffers,
)
from manabot.env.shared_effects import DETAIL_DIM, EFFECT_ROWS, MEMBER_DIM
from manabot.policy.buckets import has_history_events

Array = np.ndarray


class RecordNamespace(IntEnum):
    """Id column families of fact and shared-effect rows (native namespaces)."""

    TOKEN = 1
    NAME = 2
    RESTRICTION = 3


# Learning token kinds whose value is a number, not an id.
NUMBER_TOKEN_KINDS = (4, 5)
# Fact kinds by how admission and preparation read their id and detail words.
ORDINAL_ID_KINDS = (
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_CONTINUATION_PROGRAM,
    FACT_COMBAT_RESTRICTION,
)
PROGRAM_KINDS = (FACT_EFFECT_PROGRAM, FACT_CONTINUATION_PROGRAM)
NAMED_DETAIL_KINDS = (FACT_EFFECT, FACT_COPY_EXCEPTION)
# One-based object joins of a relation row.
_RELATION_OBJECT_COLUMNS = (6, 24)
_PERMANENT_KEYS = ("agent_permanents", "opponent_permanents")
# Validity of the visible object table, in its concatenation order.
OBJECT_VALIDITY_KEYS = (
    "agent_player_valid",
    "opponent_player_valid",
    "agent_cards_valid",
    "opponent_cards_valid",
    "agent_permanents_valid",
    "opponent_permanents_valid",
)


# -- checkpoint lookup tables ----------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Observation layout the policy reads, from the checkpoint's encoder."""

    cards_per_player: int
    perms_per_player: int
    max_actions: int
    numeric_permanent_columns: tuple[int, ...]
    link_columns: tuple[int, ...]

    @property
    def objects(self) -> int:
        return 2 + 2 * self.cards_per_player + 2 * self.perms_per_player


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
        rows = self.record_number_rows[kinds.clip(0, 5)].copy()
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
    sizes: tuple[int, ...]
    unpack: A
    restore: A
    lengths: tuple[int, ...]


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
    action_features: A  # [K, action_dim - 2]
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
    live_index: A | None = None  # flat valid object slots, including value token
    live_offsets: A | None = None  # [B+1] cumulative live-token counts
    legal_support: bool = True  # every observation has a legal action
    # A serving scope may bound this batch's lifetime without changing its rows.
    guard: Callable[[object], None] | None = None
    bucket_needs: tuple[tuple[int, ...], ...] | None = None


def prepare(
    obs: ObservationBuffers,
    tables: Tables,
    *,
    inference: bool = False,
    bucketed: bool = False,
    live_tokens: bool = False,
    value_token: bool = False,
    binding: str = "",
    action_capacity: int | None = None,
    select_programs: bool = False,
) -> Prepared[Array]:
    """Build every index a backend forward reads from one admitted batch.

    `inference` enables CPU inference compaction: unused fact columns are
    dropped unless bucketed, and unused attention columns unless live tokens
    are packed. `action_capacity` retains full logit coordinates after trimming.
    Counts and signed integers stay raw where backend arithmetic determines
    rounding. The input belongs to `binding`, not to a particular model instance.
    """
    obs = {key: np.asarray(value) for key, value in obs.items()}
    facts = obs["facts"]
    if inference and not bucketed:
        facts = facts[:, (facts[..., FACT_VALID_INDEX] == 1).any(0)]
    validity = np.concatenate([obs[key] for key in OBJECT_VALIDITY_KEYS], 1)
    belief = _belief(obs)
    if belief is not None:
        validity = np.concatenate(
            (validity, belief.validity, np.ones((len(validity), 1), np.float32)), 1
        )
    columns = None
    if inference and not live_tokens:
        used = np.nonzero((validity != 0).any(0))[0]
        if 0 < len(used) < validity.shape[1]:
            columns = used
    layout = tables.layout
    batch = obs["actions_valid"].shape[0]
    objects = validity.shape[1]
    f32 = np.float32
    permanents = np.stack([obs[key] for key in _PERMANENT_KEYS], axis=1)
    numeric = permanents[..., list(layout.numeric_permanent_columns)].astype(f32)
    knowledge = obs["known_hand"]
    decklists = obs["decklists"]
    link_targets: list[Array] = []
    link_masks: list[Array] = []
    base = np.arange(batch)[:, None] * objects
    for column in layout.link_columns:
        links = np.concatenate([obs[key][..., column] for key in _PERMANENT_KEYS], 1)
        link_targets.append(
            (base + (links.astype(np.int64) - 1).clip(min=0)).reshape(-1)
        )
        link_masks.append((links > 0).astype(f32))

    relations = obs["relations"]
    relation_valid = relations[..., RELATION_VALID_INDEX]
    clean = np.where(relation_valid[..., None] == 1, relations, 0)
    features = clean.copy()
    pending = np.isin(clean[..., 3], (7, 8, 10))
    features[..., 1:3] = np.where(pending[..., None], 0, clean[..., 1:3])
    relation_features = np.concatenate(
        [
            features[..., :5],
            features[..., 7:23],
            features[..., 25:RELATION_VALID_INDEX],
            features[..., RELATION_VALID_INDEX + 1 :],
        ],
        -1,
    ).astype(f32)
    relation_count = relations.shape[1]
    relation_objects: list[Array] = []
    relation_masks: list[Array] = []
    for column in _RELATION_OBJECT_COLUMNS:
        indexes = clean[..., column].astype(np.int64)
        relation_objects.append((base + (indexes - 1).clip(min=0)).reshape(-1))
        relation_masks.append((indexes > 0).astype(f32))
    frame_ids = np.where(relation_valid == 1, relations[..., 1], 1).astype(np.int64)
    frame_base = np.arange(batch)[:, None] * relation_count
    frame_indexes = (frame_base + (frame_ids - 1).clip(min=0)).reshape(-1)

    legal = obs["actions_valid"] > 0
    if action_capacity is not None:
        legal = np.pad(legal, ((0, 0), (0, action_capacity - legal.shape[1])))
    legal_batches, legal_slots = np.nonzero(legal)
    positions = np.zeros(legal.shape, dtype=np.int64)
    positions[legal_batches, legal_slots] = np.arange(len(legal_batches))
    action_rows = obs["actions"][legal_batches, legal_slots]
    frames = action_rows[:, -2].astype(np.int64)
    candidates = obs["name_candidates"][legal_batches, legal_slots].astype(np.int64)
    focus = obs["action_focus"][legal_batches, legal_slots].astype(np.int64)
    if candidates.shape[-1] == 4:
        first, last = candidates[:, 2], candidates[:, 3]
    else:
        first = last = np.zeros(len(candidates), np.int64)

    players = [obs[key].astype(f32) for key in ("agent_player", "opponent_player")]
    cards = [obs[key].astype(f32) for key in ("agent_cards", "opponent_cards")]
    if "player_turn_state" in obs:
        players = [
            np.concatenate((row, obs["player_turn_state"][:, seat : seat + 1]), -1)
            for seat, row in enumerate(players)
        ]
        cards = [
            np.concatenate((row, obs["card_turn_state"][:, seat, :, None]), -1)
            for seat, row in enumerate(cards)
        ]
    c, p = cards[0].shape[1], permanents.shape[2]
    ownership = np.broadcast_to(
        np.repeat([True, False, True, False, True, False], [1, 1, c, c, p, p]),
        (batch, 2 + 2 * c + 2 * p),
    )
    if belief is not None:
        ownership = np.concatenate(
            (ownership, belief.owners == 0, np.zeros((batch, 1), bool)), 1
        )
    present = validity != 0
    if value_token:
        present = np.concatenate((present, np.ones((batch, 1), bool)), 1)
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
        known_rows=tables.rows(knowledge[..., 0]),
        known_counts=knowledge[..., 1].astype(f32),
        deck_rows=tables.rows(decklists[..., 0]),
        deck_counts=(decklists[..., 1:] / f32(60)).astype(f32),
        permanent_rows=tables.rows(permanents[..., PERMANENT_DEFINITION_INDEX]),
        link_targets=tuple(link_targets),
        link_masks=tuple(link_masks),
        facts=_facts(facts, tables, objects),
        effects=_effects(obs, tables, objects),
        relation_features=relation_features,
        relation_valid=relation_valid.astype(f32),
        relation_rows=tables.rows(relations[..., [5, 23]]),
        relation_objects=tuple(relation_objects),
        relation_object_masks=tuple(relation_masks),
        frame_indexes=frame_indexes,
        legal=legal,
        legal_batches=legal_batches.astype(np.int64),
        legal_slots=legal_slots.astype(np.int64),
        logit_positions=positions,
        action_features=action_rows[:, :-2].astype(f32),
        action_frames=legal_batches * relation_count + (frames - 1).clip(min=0),
        action_frame_mask=(frames != 0).astype(f32),
        name_rows=candidates[:, 0],
        name_mask=(candidates[:, 0] > 0).astype(f32),
        group_first=(first - 1).clip(min=0),
        group_last=last,
        group_mask=(first > 0).astype(f32),
        group_counts=(last - first + 1).clip(min=1).astype(f32),
        candidate_rows=tables.rows(candidates[:, 1]),
        focus=legal_batches[:, None] * objects + focus.clip(min=0),
        focus_mask=(focus != -1).astype(f32),
        attention_columns=columns,
        binding=binding,
        is_agent=ownership,
        # Full-catalog preparation skips the scan for referenced definitions.
        programs=tables.programs(
            _referenced_definitions(obs, tables, candidates)
            if select_programs
            else None
        )
        if tables.definition_rows is not None
        else None,
        belief=belief,
        events=_events(obs, tables, objects),
        live_index=np.flatnonzero(present) if live_tokens else None,
        live_offsets=np.concatenate(([0], present.sum(1).cumsum()))
        if live_tokens
        else None,
    )


def _referenced_definitions(
    obs: ObservationBuffers, tables: Tables, candidates: Array
) -> Array:
    """Catalog subset read by learning, in sorted row order (zero is padding)."""
    references = [
        obs["semantic_cards"],
        *(obs[key][..., PERMANENT_DEFINITION_INDEX] for key in _PERMANENT_KEYS),
        obs["known_hand"][..., 0],
        obs["decklists"][..., 0],
        obs["relations"][..., [5, 23]],
        candidates[:, 1],
    ]
    if has_history_events(obs):
        references.extend(
            obs["events"][..., col][obs["events_valid"] == 1] for col in (2, 7)
        )
    if tables.catalog_records:
        references.extend(
            (
                obs["facts"][..., FACT_SOURCE_INDEX][
                    obs["facts"][..., FACT_VALID_INDEX] == 1
                ],
                obs["effects"][..., 11][obs["effects"][..., 17] == 1],
            )
        )
    referenced = np.unique(
        tables.rows(np.concatenate([ids.reshape(-1) for ids in references]))
    )
    return referenced[referenced > 0]


def _facts(facts: Array, tables: Tables, objects: int) -> FactRoutes[Array]:
    batch = len(facts)
    valid = facts[..., FACT_VALID_INDEX] == 1
    # Index only live facts on the host. Restore padding before the forward:
    # its matrix shapes and gradient reduction order must remain unchanged.
    live = facts[valid]

    def padded(values: Array) -> Array:
        shape = (*valid.shape, *values.shape[1:])
        result = np.zeros(shape, dtype=values.dtype)
        result[valid] = values
        return result

    kinds = live[..., FACT_KIND_INDEX].astype(np.int64)
    duration = live[..., FACT_DURATION_INDEX]
    ident, detail = (
        record_word(live, FACT_ID_INDEX),
        record_word(live, FACT_DETAIL_INDEX),
    )
    ordinal = np.isin(kinds, ORDINAL_ID_KINDS)
    scale = np.float32(FACT_TIMESTAMP_SCALE)
    columns = np.stack(
        [
            live[..., FACT_VALUE_INDEX],
            live[..., FACT_VALUE_INDEX + 1],
            (duration == 1).astype(np.float32),
            (duration == 2).astype(np.float32),
            live[..., FACT_ORDER_INDEX] / scale,
            np.where(ordinal, ident, 0).astype(np.float32) / scale,
        ],
        -1,
    ).astype(np.float32)
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
        id_mask = padded(named.astype(np.float32))
        detail_mask = padded(named_detail.astype(np.float32))
    else:
        # Torch linear chooses a different accumulation path for strided inputs.
        columns = np.ascontiguousarray(columns[..., :5])
        id_rows = ident % tables.record_buckets
        detail_rows = detail % tables.record_buckets
        sources = live[..., FACT_SOURCE_INDEX].astype(np.int64) % tables.record_buckets
        # Hashed records embed every row, padding included.
        id_mask = np.ones(valid.shape, np.float32)
        detail_mask = np.ones(valid.shape, np.float32)
    links = padded(live[..., FACT_OBJECT_INDEX].astype(np.int64))
    base = np.arange(batch)[:, None] * objects
    all_programs = padded(program)
    return FactRoutes(
        kinds=padded(kinds),
        columns=padded(columns),
        id_rows=padded(id_rows),
        id_mask=id_mask,
        detail_rows=padded(detail_rows),
        detail_mask=detail_mask,
        token_rows=padded(token_rows),
        token_mask=all_programs.astype(np.float32),
        source_rows=padded(sources),
        targets=(base + (links - 1).clip(min=0)).reshape(-1),
        linked=(valid & (links > 0)).astype(np.float32),
        programs=_fact_programs(facts, all_programs, tables.order_lengths),
    )


def _fact_programs(
    facts: Array, program: Array, order_lengths: Callable[[Array], Array] | None = None
) -> FactPrograms[Array] | None:
    """Preserve admitted program boundaries and raw integers for the backend GRU."""
    flat = program.reshape(-1)
    if not flat.any():
        return None
    rows = np.nonzero(flat)[0]
    program_facts = facts.reshape(-1, facts.shape[-1])[rows]
    starts = np.nonzero(program_facts[:, FACT_VALUE_INDEX + 1] == 0)[0]
    lengths = np.diff(np.append(starts, len(rows)))
    word = record_word(program_facts, FACT_DETAIL_INDEX)
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
    obs: ObservationBuffers, tables: Tables, objects: int
) -> EffectRoutes[Array] | None:
    headers = obs["effects"]
    details = obs["effect_details"]
    members = obs["effect_members"]
    detail_rows, member_rows = details.shape[1], members.shape[1]
    live_batches, live_slots = np.nonzero(headers[..., 17] == 1)
    if len(live_batches) == 0:
        return None
    # Select before casting: padding may contain arbitrary payloads, including NaN.
    selected = headers[live_batches, live_slots]
    live = selected.astype(np.int64)
    halves, source, links = live[:, :2], live[:, 11], live[:, [10, 12]]
    detail_indexes: list[int] = []
    member_indexes: list[int] = []
    member_records: list[int] = []
    token_indexes: list[int] = []
    lengths: list[int] = []
    header_tokens = 3 * len(live)
    for i, (b, (d, count, m, targets)) in enumerate(
        zip(live_batches.tolist(), live[:, 13:17].tolist(), strict=True)
    ):
        detail_start = header_tokens + len(detail_indexes)
        token_indexes.extend(range(3 * i, 3 * i + 3))
        token_indexes.extend(range(detail_start, detail_start + count))
        lengths.append(3 + count)
        detail_indexes.extend(range(b * detail_rows + d, b * detail_rows + d + count))
        member_indexes.extend(range(b * member_rows + m, b * member_rows + m + targets))
        member_records.extend([i] * targets)
    detail_host = details.reshape(-1, DETAIL_DIM)[detail_indexes].astype(np.int64)
    kinds, ids = detail_host[:, 0], detail_host[:, 1:]
    targets = members.reshape(-1, MEMBER_DIM)[member_indexes][:, 0].astype(np.int64)
    records = np.asarray(member_records, dtype=np.int64)
    numeric = selected[:, 2:10].astype(np.float32)
    numeric[:, -1] /= np.float32(EFFECT_ROWS)
    base = live_batches * objects
    unsigned = (ids[:, 0] << 16) | ids[:, 1]
    program = kinds > 10
    detail_embedding_rows = np.zeros(len(kinds), dtype=np.int64)
    if (~program).any():
        detail_embedding_rows[~program] = tables.names(
            RecordNamespace.TOKEN, unsigned[~program]
        )
    if program.any():
        detail_embedding_rows[program] = tables.tokens(
            kinds[program] - 10, unsigned[program]
        )
    signed = np.where(unsigned >= 2**31, unsigned - 2**32, unsigned)
    return EffectRoutes(
        numeric=numeric,
        controller=base + (links[:, 0] - 1).clip(min=0),
        controller_mask=(links[:, 0] > 0).astype(np.float32),
        source_object=base + (links[:, 1] - 1).clip(min=0),
        source_object_mask=(links[:, 1] > 0).astype(np.float32),
        modification_rows=tables.names(
            RecordNamespace.TOKEN, (halves[:, 0] << 16) | halves[:, 1]
        ),
        source_rows=tables.rows(source)
        if tables.catalog_records
        else source % tables.record_buckets,
        detail_rows=detail_embedding_rows,
        detail_kinds=kinds,
        detail_numbers=signed.astype(np.float32),
        detail_is_number=((kinds == 14) | (kinds == 15)).astype(np.float32),
        token_indexes=np.asarray(token_indexes, dtype=np.int64),
        schedule=pack(lengths),
        routes=live_batches[records] * objects + targets - 1,
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
    obs: ObservationBuffers, tables: Tables, objects: int
) -> EventRows[Array] | None:
    if not has_history_events(obs):
        return None
    valid = obs["events_valid"]
    rows = np.where((valid != 0)[..., None], obs["events"], 0)
    counts = valid.sum(1, keepdims=True)
    focus = rows[..., [5, 10]].astype(np.int64)
    return EventRows(
        kinds=rows[..., 0].astype(np.int64),
        amounts=rows[..., 1:2].astype(np.float32) / np.float32(20),
        ages=(counts - valid.cumsum(1)).clip(min=0).astype(np.float32),
        validity=valid.astype(bool),
        counts=counts.astype(np.float32),
        definitions=tables.rows(rows[..., [2, 7]]),
        owners=rows[..., [3, 8]].astype(np.int64),
        zones=rows[..., [4, 9]].astype(np.int64),
        focus=np.arange(len(rows))[:, None, None] * objects + focus.clip(min=0),
        focus_mask=(focus >= 0) & (valid != 0)[..., None],
        status=rows[..., [6, 11]].astype(np.int64),
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
