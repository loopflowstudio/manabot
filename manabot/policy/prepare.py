"""Host-side index preparation shared by every framework backend.

The PyTorch forward mixes float arithmetic with integer bookkeeping it does on
tensors: definition-row lookups, record-vocabulary `searchsorted`, legal-action
coordinates, packed GRU schedules and shared-effect CSR routes. Here that
bookkeeping runs once per batch in numpy (`prepare`), producing a `Prepared`
batch of float features and integer indexes, so a backend's graph (for example
`manabot.backends.mlx.policy`) holds only gathers, scatters and arithmetic.

`Tables` holds the checkpoint's lookup buffers (catalog programs, record
vocabulary, naming joins, ownership template) and the observation layout; it is
read from the checkpoint container (`manabot.policy.checkpoint`), never from
the running engine. This module imports NumPy only.

Inputs must already be admitted (`manabot.model.admission`): nothing here
validates ids or partitions, exactly as the PyTorch forward does not.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
)
from manabot.env.shared_effects import DETAIL_DIM, EFFECT_DIM, EFFECT_ROWS, MEMBER_DIM

Array = np.ndarray

# Record namespaces (`manabot.model.semantic_cards.RecordNamespace`).
_TOKEN, _NAME, _RESTRICTION = 1, 2, 3
_NUMBER_KINDS = (4, 5)
_ORDINAL_ID_KINDS = (
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_CONTINUATION_PROGRAM,
    FACT_COMBAT_RESTRICTION,
)
_PROGRAM_KINDS = (FACT_EFFECT_PROGRAM, FACT_CONTINUATION_PROGRAM)
_NAMED_DETAIL_KINDS = (FACT_EFFECT, FACT_COPY_EXCEPTION)
# One-based object joins of a relation row.
_RELATION_OBJECT_COLUMNS = (6, 24)
_PERMANENT_KEYS = ("agent_permanents", "opponent_permanents")
_VALID_KEYS = (
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

    definition_rows: Array  # [ids + 1] transport id -> catalog row + 1 (0: none)
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
        values = {field: np.asarray(buffers[key]) for field, key in cls.BUFFERS.items()}
        values["is_agent"] = values["is_agent"].reshape(-1).astype(bool)
        return cls(**values, layout=layout)

    def rows(self, ids: Array) -> Array:
        """Catalog rows + 1 of admitted transport ids (`definition_rows`)."""
        return self.definition_rows[ids.astype(np.int64)]

    def names(self, namespace: Array | int, words: Array) -> Array:
        """Token-embedding rows of record names (`record_names`)."""
        return _lookup(
            self.record_name_keys,
            self.record_name_rows,
            _key(np.broadcast_to(np.asarray(namespace), words.shape), words),
        )

    def tokens(self, kinds: Array, words: Array) -> Array:
        """Token-embedding rows of program learning tokens (`record_tokens`)."""
        kinds = kinds.astype(np.int64)
        number = np.isin(kinds, _NUMBER_KINDS)
        rows = self.record_number_rows[kinds.clip(0, 5)].copy()
        if (~number).any():
            rows[~number] = _lookup(
                self.record_token_keys,
                self.record_token_rows,
                _key(kinds[~number], words[~number]),
            )
        return rows


def _key(high: Array, low: Array) -> Array:
    return (high.astype(np.int64) << 32) | (low.astype(np.int64) & 0xFFFFFFFF)


def _lookup(keys: Array, rows: Array, query: Array) -> Array:
    if query.size == 0:
        return np.zeros(query.shape, dtype=np.int64)
    return rows[np.searchsorted(keys, query).clip(max=len(keys) - 1)]


def _word(rows: Array, index: int) -> Array:
    """The u32 id at `index` (two 16-bit halves)."""
    return (rows[..., index].astype(np.int64) << 16) | rows[..., index + 1].astype(
        np.int64
    )


def program_number(value: Array) -> Array:
    """`manabot.model.records.program_number` in float32."""
    value = value.astype(np.float32)
    return (np.sign(value) * np.log1p(np.abs(value)) / np.float32(10)).astype(
        np.float32
    )


# -- packed recurrent schedules ----------------------------------------------------


@dataclass(frozen=True)
class Packed:
    """Variable-length sequences over a flat token table, packed step-major.

    Sequences are sorted by descending length (PyTorch's packing order). Step t
    reads `sizes[t]` live sequences, the prefix of that order. `gather` is the
    flat token of each packed row; `unpack` the packed row of each token;
    `restore` each original sequence's position in the sorted order.
    """

    gather: Array
    sizes: tuple[int, ...]
    unpack: Array
    restore: Array


def pack(lengths: Sequence[int] | Array) -> Packed:
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    order = np.argsort(-lengths, kind="stable")
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
    return Packed(flat, tuple(sizes), unpack, restore)


def catalog_schedule(tables: Tables) -> tuple[Array, Packed]:
    """Packed token ids of every catalog program, encoded independently."""
    lengths = tables.catalog_lengths.astype(np.int64)
    schedule = pack(lengths)
    width = tables.catalog_tokens.shape[1]
    rows = np.repeat(np.arange(len(lengths)), lengths)
    steps = np.concatenate([np.arange(n) for n in lengths])
    flat = tables.catalog_tokens.reshape(-1)[rows * width + steps]
    return flat[schedule.gather], schedule


# -- prepared batches --------------------------------------------------------------


@dataclass(frozen=True)
class FactRoutes:
    kinds: Array  # [B,F]
    columns: Array  # [B,F,6] numeric columns
    id_rows: Array  # [B,F] token rows of named ids
    id_mask: Array  # [B,F]
    detail_rows: Array
    detail_mask: Array
    token_rows: Array  # program token rows
    token_mask: Array
    source_rows: Array  # [B,F] catalog row + 1
    targets: Array  # [B*F] flat object index
    linked: Array  # [B,F] float
    # Programs (absent when the batch has none).
    program_rows: Array | None  # [T] flat fact row of each program token
    program_numbers: Array | None  # [T]
    program_positions: Array | None  # [B*F] token position of each fact row
    program_mask: Array | None  # [B*F] bool
    program_schedule: Packed | None


@dataclass(frozen=True)
class EffectRoutes:
    numeric: Array  # [L,8]
    controller: Array  # [L] flat object index
    controller_mask: Array
    source_object: Array
    source_object_mask: Array
    modification_rows: Array  # [L]
    source_rows: Array  # [L] catalog row + 1
    detail_rows: Array  # [Dt]
    detail_kinds: Array  # [Dt]
    detail_numbers: Array  # [Dt]
    detail_is_number: Array  # [Dt] float
    token_indexes: Array  # [sum lengths] rows of the header+detail token table
    schedule: Packed
    routes: Array  # [M] flat object index
    member_records: Array  # [M]


@dataclass(frozen=True)
class Prepared:
    """One admitted observation batch, ready for a backend policy."""

    batch: int
    agent_player: Array
    opponent_player: Array
    agent_cards: Array
    opponent_cards: Array
    permanents: Array  # [B,2,P,numeric]
    validity: Array  # [B,N]
    card_rows: Array  # [B,2,C]
    known_rows: Array  # [B,2,K]
    known_counts: Array  # [B,2,K] already / 60
    deck_rows: Array  # [B,2,R]
    deck_counts: Array  # [B,2,R,4] already / 60
    permanent_rows: Array  # [B,2,P]
    link_targets: tuple[Array, ...]  # per link column [B*2P] flat object index
    link_masks: tuple[Array, ...]  # [B,2P] float
    facts: FactRoutes
    effects: EffectRoutes | None
    relation_features: Array  # [B,R,44]
    relation_valid: Array  # [B,R]
    relation_rows: Array  # [B,R,2] catalog rows + 1
    relation_objects: tuple[Array, ...]  # per object column [B*R] flat index
    relation_object_masks: tuple[Array, ...]
    frame_indexes: Array  # [B*R] flat frame slot
    # Legal actions, row-major over [B,A].
    legal: Array  # [B,A] bool
    legal_batches: Array  # [K]
    logit_positions: Array  # [B,A] index into K (0 where illegal)
    action_features: Array  # [K, action_dim - 2]
    action_frames: Array  # [K] flat frame slot
    action_frame_mask: Array  # [K] float
    name_rows: Array  # [K]
    name_mask: Array
    group_first: Array
    group_last: Array
    group_mask: Array
    group_counts: Array
    candidate_rows: Array  # [K] catalog row + 1
    focus: Array  # [K,F] flat object index
    focus_mask: Array  # [K,F] float
    # Inference only: object columns any row uses (None: attend over all).
    attention_columns: Array | None = None


def prepare(
    obs: Mapping[str, Array], tables: Tables, *, inference: bool = False
) -> Prepared:
    """Build every index a backend forward reads from one admitted batch.

    `inference` applies the PyTorch CPU inference compaction: fact columns no
    row uses are dropped, and attention runs over the object columns some row
    uses. Padding is never an attention key and every layer zeros its output,
    so both are exact; learning keeps fixed shapes, as PyTorch does.
    """
    obs = {key: np.asarray(value) for key, value in obs.items()}
    facts = obs["facts"]
    if inference:
        facts = facts[:, (facts[..., FACT_VALID_INDEX] == 1).any(0)]
    validity = np.concatenate([obs[key] for key in _VALID_KEYS], 1)
    columns = None
    if inference:
        used = np.nonzero((validity != 0).any(0))[0]
        if 0 < len(used) < validity.shape[1]:
            columns = used
    layout = tables.layout
    batch = obs["actions_valid"].shape[0]
    objects = layout.objects
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

    return Prepared(
        batch=batch,
        agent_player=obs["agent_player"].astype(f32),
        opponent_player=obs["opponent_player"].astype(f32),
        agent_cards=obs["agent_cards"].astype(f32),
        opponent_cards=obs["opponent_cards"].astype(f32),
        permanents=numeric,
        validity=validity.astype(f32),
        card_rows=tables.rows(obs["semantic_cards"]),
        known_rows=tables.rows(knowledge[..., 0]),
        known_counts=(knowledge[..., 1] / f32(60)).astype(f32),
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
    )


def _facts(facts: Array, tables: Tables, objects: int) -> FactRoutes:
    batch, count, _ = facts.shape
    valid = facts[..., FACT_VALID_INDEX] == 1
    facts = np.where(valid[..., None], facts, 0)
    kinds = facts[..., FACT_KIND_INDEX].astype(np.int64)
    duration = facts[..., FACT_DURATION_INDEX]
    ident, detail = _word(facts, FACT_ID_INDEX), _word(facts, FACT_DETAIL_INDEX)
    ordinal = np.isin(kinds, _ORDINAL_ID_KINDS)
    scale = np.float32(FACT_TIMESTAMP_SCALE)
    columns = np.stack(
        [
            facts[..., FACT_VALUE_INDEX],
            facts[..., FACT_VALUE_INDEX + 1],
            (duration == 1).astype(np.float32),
            (duration == 2).astype(np.float32),
            facts[..., FACT_ORDER_INDEX] / scale,
            np.where(ordinal, ident, 0).astype(np.float32) / scale,
        ],
        -1,
    ).astype(np.float32)
    named = (kinds > 0) & ~ordinal
    program = np.isin(kinds, _PROGRAM_KINDS)
    restriction = kinds == FACT_COMBAT_RESTRICTION
    named_detail = (np.isin(kinds, _NAMED_DETAIL_KINDS) & (detail != 0)) | restriction
    id_space = np.where(kinds == FACT_SET_NAME, _NAME, _TOKEN)
    detail_space = np.where(restriction, _RESTRICTION, _TOKEN)

    def masked(mask: Array, rows: Array) -> Array:
        return np.where(mask, rows, 0)

    first_name = tables.record_name_keys[0]
    first_token = tables.record_token_keys[0]
    id_rows = _lookup(
        tables.record_name_keys,
        tables.record_name_rows,
        np.where(named, _key(id_space, ident), first_name),
    )
    detail_rows = _lookup(
        tables.record_name_keys,
        tables.record_name_rows,
        np.where(named_detail, _key(detail_space, detail), first_name),
    )
    token_kind = facts[..., FACT_VALUE_INDEX].astype(np.int64)
    number = np.isin(token_kind, _NUMBER_KINDS)
    token_rows = np.where(
        number,
        tables.record_number_rows[token_kind.clip(0, 5)],
        _lookup(
            tables.record_token_keys,
            tables.record_token_rows,
            np.where(program & ~number, _key(token_kind, detail), first_token),
        ),
    )
    links = facts[..., FACT_OBJECT_INDEX].astype(np.int64)
    base = np.arange(batch)[:, None] * objects
    routes = dict(
        kinds=kinds,
        columns=columns,
        id_rows=masked(named, id_rows),
        id_mask=named.astype(np.float32),
        detail_rows=masked(named_detail, detail_rows),
        detail_mask=named_detail.astype(np.float32),
        token_rows=masked(program, token_rows),
        token_mask=program.astype(np.float32),
        source_rows=tables.rows(facts[..., FACT_SOURCE_INDEX]),
        targets=(base + (links - 1).clip(min=0)).reshape(-1),
        linked=(valid & (links > 0)).astype(np.float32),
    )
    flat = program.reshape(-1)
    if not flat.any():
        return FactRoutes(
            **routes,
            program_rows=None,
            program_numbers=None,
            program_positions=None,
            program_mask=None,
            program_schedule=None,
        )
    rows = np.nonzero(flat)[0]
    program_facts = facts.reshape(batch * count, -1)[rows]
    starts = np.nonzero(program_facts[:, FACT_VALUE_INDEX + 1] == 0)[0]
    lengths = np.diff(np.append(starts, len(rows)))
    word = _word(program_facts, FACT_DETAIL_INDEX)
    signed = np.where(word >= 2**31, word - 2**32, word).astype(np.float32)
    kind = program_facts[:, FACT_VALUE_INDEX]
    numbers = program_number(np.where((kind == 4) | (kind == 5), signed, 0))
    positions = np.zeros(batch * count, dtype=np.int64)
    positions[rows] = np.arange(len(rows))
    return FactRoutes(
        **routes,
        program_rows=rows,
        program_numbers=numbers,
        program_positions=positions,
        program_mask=flat,
        program_schedule=pack(lengths),
    )


def _effects(
    obs: Mapping[str, Array], tables: Tables, objects: int
) -> EffectRoutes | None:
    headers = obs["effects"]
    details = obs["effect_details"]
    members = obs["effect_members"]
    host = np.where(headers[..., 17:18] == 1, headers, 0)
    header_rows, detail_rows, member_rows = (
        headers.shape[1],
        details.shape[1],
        members.shape[1],
    )
    live_batches, live_slots = np.nonzero(host[..., 17])
    if len(live_batches) == 0:
        return None
    live = host[live_batches, live_slots].astype(np.int64)
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
    numeric = headers.reshape(-1, EFFECT_DIM)[
        live_batches * header_rows + live_slots, 2:10
    ].astype(np.float32)
    numeric[:, -1] /= np.float32(EFFECT_ROWS)
    base = live_batches * objects
    unsigned = (ids[:, 0] << 16) | ids[:, 1]
    program = kinds > 10
    detail_embedding_rows = np.zeros(len(kinds), dtype=np.int64)
    if (~program).any():
        detail_embedding_rows[~program] = tables.names(_TOKEN, unsigned[~program])
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
        modification_rows=tables.names(_TOKEN, (halves[:, 0] << 16) | halves[:, 1]),
        source_rows=tables.rows(source),
        detail_rows=detail_embedding_rows,
        detail_kinds=kinds,
        detail_numbers=program_number(signed),
        detail_is_number=((kinds == 14) | (kinds == 15)).astype(np.float32),
        token_indexes=np.asarray(token_indexes, dtype=np.int64),
        schedule=pack(lengths),
        routes=live_batches[records] * objects + targets - 1,
        member_records=records,
    )


def reference_distribution(obs: Mapping[str, Array], kind: str) -> Array:
    """`manabot.training.learning.references.reference_distribution` in numpy."""
    valid = np.asarray(obs["actions_valid"]) > 0
    if kind == "uniform":
        return (valid / valid.sum(-1, keepdims=True)).astype(np.float32)
    if kind != "action_type_uniform":
        raise ValueError(f"unknown reference: {kind}")
    action_types = np.asarray(obs["actions"])[..., :ACTION_TYPE_DIM]
    types = action_types.argmax(-1)
    counts = np.zeros((valid.shape[0], action_types.shape[-1]), dtype=np.float32)
    for row in range(valid.shape[0]):
        np.add.at(counts[row], types[row], valid[row].astype(np.float32))
    weights = valid / np.take_along_axis(counts, types, -1).clip(min=1)
    return (weights / weights.sum(-1, keepdims=True)).astype(np.float32)
