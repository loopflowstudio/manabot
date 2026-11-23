"""Host validation shared by policy backends before preparing observation indexes.

The checks here own definition/vocabulary joins, fact programs, effect partitions,
continuation rows and legal-action references. Every backend runs `check` on raw
NumPy rows before `prepare` turns validity into booleans or performs lookups.
Padding is ignored where a validity column owns it and checked otherwise.
"""

from dataclasses import dataclass

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import CompactObservation
from manabot.env.observation import (
    ACTION_CONTINUATION_INDEX,
    FACT_COMBAT_RESTRICTION,
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_DURATION_INDEX,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS,
    FACT_OBJECT_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    RELATION_ROWS,
    RELATION_VALID_INDEX,
    STACK_DEFINITION_INDEX,
    STACK_SOURCE_INDEX,
    ObservationBuffers,
    ObservationSpace,
    column,
    columns,
)
from manabot.env.shared_effects import (
    DETAIL_DIM,
    DETAIL_KIND,
    DETAIL_KINDS,
    DETAIL_ROWS,
    DETAIL_VALUE,
    EFFECT_DIM,
    EFFECT_LINKS,
    EFFECT_MODIFICATION,
    EFFECT_ROWS,
    EFFECT_SOURCE_DEFINITION,
    EFFECT_SPANS,
    EFFECT_VALID,
    MEMBER_DIM,
    MEMBER_OBJECT,
    MEMBER_ROWS,
    NUMBER_TOKEN_KINDS,
    PROGRAM_TOKEN_KIND,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.buckets import (
    BucketedObservation,
    bucket_observation,
    object_trimming,
)
from manabot.policy.decision_memory import (
    admit_memory,
    retained_memory,
    validate_memory_vocabulary,
    validate_retained_memory,
)
from manabot.policy.prepare import (
    NAMED_DETAIL_KINDS,
    ORDINAL_ID_KINDS,
    RecordNamespace,
    Tables,
    record_key,
    record_word,
)
from manabot.policy.readings import PROGRAM_KINDS, Readings

# Native columns these checks read, by name.
_FACT_ORDINAL = column("facts", "value_2")  # a program row's ordinal
_MEMBER_ORDER = column("effect_members", "order")
_RELATION_FRAME = column("relations", "frame")
_RELATION_OBJECTS = columns("relations", "source", "reference")
_RELATION_DEFINITIONS = columns(
    "relations", "source_definition", "reference_definition"
)
_NAME_ROW = column("name_candidates", "name")
_NAME_DEFINITION = column("name_candidates", "definition")
_NAME_GROUP_FIRST = column("name_candidates", "group_first")
_NAME_GROUP_LAST = column("name_candidates", "group_last")
_DETAIL_FIELD = column("decision_reference_details", "field")
_DETAIL_TOKEN_KIND = column("decision_reference_details", "token_kind")
_DETAIL_VALUE = column("decision_reference_details", "value_high")
_EVENT_KIND = column("events", "kind")
_EVENT_SUBJECTS = columns("events", "source_definition", "target_definition")


def binding(model: ModelConfig, space: ObservationSpace, readings: Readings) -> str:
    """The configuration a prepared batch belongs to, independent of weights.

    Compared in process and stored nowhere; the readings (and so the saved
    widths) are part of it, since they decide every prepared number."""
    key = model.model_dump_json() + "\n" + space.config.model_dump_json()
    return key + f"\nreadings={sorted(readings.items())}"


class BindingCache:
    """Reuse a binding while the model, observation config and readings match its snapshot."""

    def __init__(self) -> None:
        self.snapshot: tuple[ModelConfig, ObservationConfig, Readings, str] | None = (
            None
        )

    def get(
        self, model: ModelConfig, space: ObservationSpace, readings: Readings
    ) -> str:
        cached = self.snapshot
        if cached is None or cached[:3] != (model, space.config, readings):
            cached = (
                model.model_copy(deep=True),
                space.config.model_copy(deep=True),
                dict(readings),
                binding(model, space, readings),
            )
            self.snapshot = cached
        return cached[3]


@dataclass(frozen=True)
class Spec:
    """Configuration and immutable vocabularies shared by admission and preparation."""

    model: ModelConfig
    space: ObservationSpace
    tables: Tables
    definition_count: int

    @property
    def binding(self) -> str:
        return binding(self.model, self.space, self.tables.layout.readings)


def legal_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a binary host mask before preparation turns it into booleans."""
    if not (np.isfinite(mask) & ((mask == 0) | (mask == 1))).all():
        raise ValueError("invalid_action_mask")
    return mask.astype(bool)


def admit(
    obs: ObservationBuffers, spec: Spec, legal: np.ndarray, *, bucketed: bool
) -> BucketedObservation | None:
    """Validate in original coordinates and return the measured views for preparation."""
    measured = (
        bucket_observation(obs, spec.space, None, objects=object_trimming(spec.model))
        if bucketed
        else None
    )
    check(obs, spec, legal, measured)
    return measured


def check(
    obs: ObservationBuffers,
    spec: Spec,
    legal: np.ndarray,
    bucketed: BucketedObservation | None = None,
) -> None:
    """Admit raw coordinates. Measured prefixes exclude only unreferenced zero rows."""
    retained = bool(spec.space.config.max_decision_records)
    prefixes = retained_memory(obs)
    if retained != (prefixes is not None or "decision_records" in obs):
        raise ValueError("decision-memory model and input contract disagree")
    if prefixes is None:
        admit_memory(obs)
    else:
        validate_retained_memory(
            prefixes,
            spec.tables,
            spec.definition_count,
            spec.space.config.decision_memory_include_own,
        )
    if (
        retained
        and prefixes is None
        and not spec.space.config.decision_memory_include_own
        and (
            obs["decision_records"][..., column("decision_records", "chooser_role")]
            == 1
        ).any()
    ):
        raise ValueError("own-seat decision memory is disabled")
    if retained:
        for axis, key in enumerate(MEMORY_FIELDS):
            expected = spec.space.shapes[key]
            if isinstance(obs, CompactObservation):
                if (
                    obs.memory_capacities[axis] != expected[0]
                    or obs[key].shape[1] > expected[0]
                ):
                    raise ValueError(f"decision-memory capacity mismatch: {key}")
            elif prefixes is not None:
                if prefixes.memory_capacities[axis] != expected[0]:
                    raise ValueError(f"decision-memory capacity mismatch: {key}")
            elif obs[key].shape[1:] != expected:
                raise ValueError(f"decision-memory capacity mismatch: {key}")
    lookup = spec.tables.definition_rows
    if spec.space.one_token and any(key not in obs for key in ("stack", "stack_valid")):
        raise ValueError("policy input v11 requires stack and stack_valid")
    if spec.space.one_token:
        if "known_library" not in obs:
            raise ValueError("policy input v11 requires known_library")
        library = obs["known_library"]
        shape = spec.space.shapes["known_library"]
        if library.shape != (len(legal), *shape):
            raise ValueError("known_library must match the declared observation shape")
        positions = library[..., column("known_library", "position")]
        known = library[..., column("known_library", "definition")]
        if not (
            np.isfinite(positions)
            & (positions == np.floor(positions))
            & (positions >= 0)
            & ((known == 0) == (positions == 0))
        ).all():
            raise ValueError(
                "known_library requires whole positive positions or zero padding"
            )

    raw = obs
    if bucketed is not None:
        obs = bucketed.rows
        legal = legal[:, : obs["actions_valid"].shape[1]]
    size = spec.definition_count if lookup is None else len(lookup)
    ids = [
        obs["semantic_cards"],
        np.stack(
            [
                obs[k][..., PERMANENT_DEFINITION_INDEX]
                for k in ("agent_permanents", "opponent_permanents")
            ],
            1,
        ),
    ]
    if lookup is not None:
        ids.extend(
            (
                obs["known_hand"][..., column("known_hand", "definition")],
                obs["decklists"][..., column("decklists", "definition")],
            )
        )
        if "known_library" in obs:
            ids.append(obs["known_library"][..., column("known_library", "definition")])
    if spec.space.one_token:
        ids.append(obs["stack"][..., STACK_DEFINITION_INDEX])
    if retained and prefixes is None:
        ids.append(
            obs["decision_references"][..., column("decision_references", "definition")]
        )
        details = obs["decision_reference_details"]
        # Field 0 with token kind 7 holds a definition id.
        definition = (details[..., _DETAIL_FIELD] == 0) & (
            details[..., _DETAIL_TOKEN_KIND] == 7
        )
        ids.append(record_word(details, _DETAIL_VALUE)[definition] + 1)
        validate_memory_vocabulary(obs, spec.tables)
    for values in ids:
        definition_ids(values, size, lookup)
    keys = (
        "agent_player",
        "opponent_player",
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
    )
    if spec.space.one_token:
        keys = (*keys, "stack")
    objects = sum(raw[k].shape[1] for k in keys)
    validity = np.concatenate([raw[k + "_valid"] for k in keys], 1)
    if spec.model.belief_count_buckets:
        beliefs(obs, spec.model)
        objects += obs["belief_card_def_ids"].shape[1] + 1
    permanent_links(obs, objects)
    if spec.space.one_token:
        sources = obs["stack"][..., STACK_SOURCE_INDEX]
        if not (
            (sources == np.floor(sources)) & (sources >= 0) & (sources <= objects)
        ).all():
            raise ValueError("invalid stack source object reference")
    catalog = None
    if spec.model.catalog_records():
        assert lookup is not None
        catalog = Catalog(
            lookup, spec.tables.record_name_keys, spec.tables.record_token_keys
        )
    sizes = bucketed.buckets.sizes if bucketed is not None else {}
    facts(
        raw["facts"],
        spec.space.config.max_facts,
        objects,
        catalog,
        prefix=sizes.get("facts"),
    )
    shared_effects(
        raw["effects"],
        raw["effect_details"],
        raw["effect_members"],
        objects,
        catalog,
        prefix=sizes.get("effects"),
    )
    definition_ids(obs["relations"][..., _RELATION_DEFINITIONS], size, lookup)
    relations(
        raw["relations"],
        spec.space.shapes["relations"][-1],
        objects,
        prefix=sizes.get("relations"),
    )
    if spec.model.recent_events:
        assert lookup is not None
        recent_events(obs, objects, lookup, spec.space.shapes["events"][-1])
    if (
        spec.model.value_aggregation != "historical_mean"
        and not spec.model.belief_count_buckets
        and (validity.sum(1) == 0).any()
    ):
        raise ValueError("value aggregation requires a valid object")
    valid_objects = validity.shape[1]
    if spec.model.belief_count_buckets:
        valid_objects += obs["belief_validity"].shape[1] + 1
    actions(obs, legal, valid_objects, spec.tables.name_definitions, lookup)


def definition_ids(
    ids: np.ndarray, size: int, lookup: np.ndarray | None = None
) -> np.ndarray:
    """Validate transport ids and return their optional catalog rows."""
    kind = "semantic" if lookup is not None else "card"
    message = f"policy received an unadmitted {kind} definition"
    if not ((ids == np.floor(ids)) & (ids >= 0) & (ids < size)).all():
        raise ValueError(message)
    rows = ids.astype(np.int64)
    if lookup is not None:
        rows = lookup[rows]
        if (rows < 0).any():
            raise ValueError(message)
    return rows


def _require_keys(keys: np.ndarray, query: np.ndarray, message: str) -> None:
    if query.size == 0:
        return
    indexes = np.searchsorted(keys, query).clip(max=len(keys) - 1)
    if not len(keys) or not np.array_equal(keys[indexes], query):
        missing = query if not len(keys) else query[keys[indexes] != query]
        detail = ", ".join(
            f"({int(key) >> 32}, 0x{int(key) & 0xFFFFFFFF:08x})"
            for key in np.unique(missing)[:8]
        )
        raise ValueError(f"{message}; missing keys: {detail}")


@dataclass(frozen=True)
class Catalog:
    """Only the immutable lookup buffers needed for admission, never weights."""

    definitions: np.ndarray
    names: np.ndarray
    tokens: np.ndarray

    def require_names(self, namespace: np.ndarray | int, words: np.ndarray) -> None:
        _require_keys(
            self.names,
            record_key(namespace, words).reshape(-1),
            "policy received an id outside the record vocabulary",
        )

    def require_tokens(self, kinds: np.ndarray, words: np.ndarray) -> None:
        kinds, words = kinds.astype(np.int64).reshape(-1), words.reshape(-1)
        number = np.isin(kinds, NUMBER_TOKEN_KINDS)
        if ((kinds < 1) | (kinds > 9)).any():
            raise ValueError("policy received an invalid program token kind")
        _require_keys(
            self.tokens,
            record_key(kinds[~number], words[~number]),
            "policy received a program token outside the vocabulary",
        )


def facts(
    rows: np.ndarray,
    max_facts: int,
    objects: int,
    catalog: Catalog | None,
    *,
    prefix: int | None = None,
) -> None:
    """Validate fact joins, kinds, catalog ids and consecutive program runs."""
    if rows.shape[1:] != (max_facts, FACT_DIM):
        raise ValueError(f"expected [batch, {max_facts}, {FACT_DIM}] facts")
    rows = rows[:, :prefix]
    valid = rows[..., FACT_VALID_INDEX]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("fact validity must be binary")
    rows = np.where(valid.astype(bool)[..., None], rows, 0)
    links, kinds = rows[..., FACT_OBJECT_INDEX], rows[..., FACT_KIND_INDEX]
    if not ((links == np.floor(links)) & (links >= 0) & (links <= objects)).all():
        raise ValueError("invalid fact object reference")
    if not ((kinds == np.floor(kinds)) & (kinds >= 0) & (kinds <= FACT_KINDS)).all():
        raise ValueError("invalid fact kind")
    program = np.isin(kinds, PROGRAM_KINDS)
    if catalog is not None:
        ident, detail = (
            record_word(rows, FACT_ID_INDEX),
            record_word(rows, FACT_DETAIL_INDEX),
        )
        named = (kinds > 0) & ~np.isin(kinds, ORDINAL_ID_KINDS)
        restriction = kinds == FACT_COMBAT_RESTRICTION
        named_detail = (
            np.isin(kinds, NAMED_DETAIL_KINDS) & (detail != 0)
        ) | restriction
        if (~(program | named_detail) & (detail != 0)).any():
            raise ValueError("fact kind carries no detail")
        spaces = RecordNamespace
        catalog.require_names(
            np.where(kinds == FACT_SET_NAME, spaces.NAME, spaces.TOKEN)[named],
            ident[named],
        )
        catalog.require_names(
            np.where(restriction, spaces.RESTRICTION, spaces.TOKEN)[named_detail],
            detail[named_detail],
        )
        catalog.require_tokens(rows[..., FACT_VALUE_INDEX][program], detail[program])
        definition_ids(
            rows[..., FACT_SOURCE_INDEX], len(catalog.definitions), catalog.definitions
        )
    if not program.any():
        return
    positions, selected = np.argwhere(program), rows[program]
    ordinals = selected[:, _FACT_ORDINAL]
    starts = np.flatnonzero(ordinals == 0)
    if not len(starts) or starts[0] != 0:
        raise ValueError("fact program must start at ordinal zero")
    lengths = np.diff(np.append(starts, len(selected)))
    if not np.array_equal(ordinals, np.concatenate([np.arange(n) for n in lengths])):
        raise ValueError("fact program ordinals must be consecutive")
    owners = np.concatenate(
        (
            positions[:, :1],
            selected[:, :FACT_DETAIL_INDEX],
            selected[:, FACT_DURATION_INDEX:FACT_VALID_INDEX],
        ),
        axis=-1,
    )
    if not np.array_equal(owners, np.repeat(owners[starts], lengths, axis=0)):
        raise ValueError("fact program changes owner or provenance")


def _indexes(values: np.ndarray, maximum: int, name: str) -> np.ndarray:
    if not ((values == np.floor(values)) & (values >= 0) & (values <= maximum)).all():
        raise ValueError(f"invalid {name} reference")
    return values.astype(np.int64)


def shared_effects(
    headers: np.ndarray,
    details: np.ndarray,
    members: np.ndarray,
    objects: int,
    catalog: Catalog | None,
    *,
    prefix: int | None = None,
) -> None:
    """Validate live headers and their complete CSR detail/member partitions."""
    if headers.ndim != 3 or headers.shape[1:] != (EFFECT_ROWS, EFFECT_DIM):
        raise ValueError(f"expected [batch, {EFFECT_ROWS}, {EFFECT_DIM}] state records")
    # Each live header selects its detail and member spans in the original buffers.
    valid = headers[:, :prefix, EFFECT_VALID]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("state record validity must be binary")
    positions = np.argwhere(valid)
    live = headers[positions[:, 0], positions[:, 1]]
    if not np.isfinite(live).all():
        raise ValueError("state records must be finite")
    batch = len(headers)
    if details.shape != (batch, DETAIL_ROWS, DETAIL_DIM) or members.shape != (
        batch,
        MEMBER_ROWS,
        MEMBER_DIM,
    ):
        raise ValueError("invalid shared effect buffer shape")
    if not len(positions):
        return
    halves = _indexes(live[:, EFFECT_MODIFICATION], 65535, "effect vocabulary half")
    source = _indexes(
        live[:, EFFECT_SOURCE_DEFINITION], 2**24, "effect source definition"
    )
    links = _indexes(live[:, EFFECT_LINKS], objects, "effect source/controller")
    if (links[:, 0] == 0).any():
        raise ValueError("effect controller is absent")
    offsets = _indexes(
        live[:, EFFECT_SPANS], max(MEMBER_ROWS, DETAIL_ROWS), "effect CSR offset/count"
    )
    detail_indexes: list[int] = []
    member_indexes: list[int] = []
    member_records: list[int] = []
    previous_batch = -1
    d_end = m_end = 0
    for i, ((b, _), (d, count, m, targets)) in enumerate(
        zip(positions.tolist(), offsets.tolist(), strict=True)
    ):
        if b != previous_batch:
            d_end = m_end = 0
            previous_batch = b
        if (
            d != d_end
            or m != m_end
            or d + count > DETAIL_ROWS
            or targets == 0
            or m + targets > MEMBER_ROWS
        ):
            raise ValueError("invalid effect CSR partition")
        d_end, m_end = d + count, m + targets
        detail_indexes.extend(range(d, d_end))
        member_indexes.extend(range(m, m_end))
        member_records.extend([i] * targets)
    detail_batches = np.repeat(positions[:, 0], offsets[:, 1])
    detail = details[detail_batches, detail_indexes]
    kinds = _indexes(detail[:, DETAIL_KIND], max(DETAIL_KINDS), "effect detail kind")
    ids = _indexes(detail[:, DETAIL_VALUE], 65535, "effect detail half")
    if not np.isin(kinds, DETAIL_KINDS).all():
        raise ValueError("live effect detail has no kind")
    member_batches = positions[member_records, 0]
    member = members[member_batches, member_indexes]
    _indexes(member[:, MEMBER_OBJECT], objects, "effect member")
    orders = _indexes(member[:, _MEMBER_ORDER], MEMBER_ROWS, "effect membership order")
    if (member[:, MEMBER_OBJECT] == 0).any():
        raise ValueError("effect membership is absent")
    if (orders == 0).any():
        raise ValueError("effect membership order is absent")
    global_orders = member_batches * (MEMBER_ROWS + 1) + orders
    if len(np.unique(global_orders)) != len(global_orders):
        raise ValueError("duplicate effect membership order")
    if catalog is not None:
        catalog.require_names(
            RecordNamespace.TOKEN, (halves[:, 0] << 16) | halves[:, 1]
        )
        definition_ids(source, len(catalog.definitions), catalog.definitions)
        unsigned = (ids[:, 0] << 16) | ids[:, 1]
        token_kinds = PROGRAM_TOKEN_KIND[kinds]
        program = token_kinds != 0
        catalog.require_names(RecordNamespace.TOKEN, unsigned[~program])
        catalog.require_tokens(token_kinds[program], unsigned[program])


def relations(
    rows: np.ndarray, width: int, objects: int, *, prefix: int | None = None
) -> None:
    """Validate continuation validity, finite payloads, object joins and frames."""
    if rows.shape[1:] != (RELATION_ROWS, width):
        raise ValueError(f"expected [batch, 64, {width}] relations")
    rows = rows[:, :prefix]
    valid = rows[..., RELATION_VALID_INDEX]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("relation validity must be binary")
    clean = np.where(valid.astype(bool)[..., None], rows, 0)
    if not np.isfinite(clean).all():
        raise ValueError("relations must be finite")
    for index in _RELATION_OBJECTS:
        values = clean[..., index]
        if not (
            (values == np.floor(values)) & (values >= 0) & (values <= objects)
        ).all():
            raise ValueError("invalid continuation object reference")
    ids = rows[..., _RELATION_FRAME]
    if not (
        ((ids == np.floor(ids)) & (ids >= 1) & (ids <= RELATION_ROWS)) | (valid == 0)
    ).all():
        raise ValueError("invalid continuation frame reference")


def names(candidates: np.ndarray, definitions: np.ndarray) -> None:
    """Validate name rows, optional definitions and balanced-group endpoints."""
    rows = definition_ids(candidates[..., _NAME_ROW], len(definitions))
    if not np.array_equal(
        candidates[..., _NAME_DEFINITION], definitions[rows].astype(candidates.dtype)
    ):
        raise ValueError("name candidate definition disagrees with vocabulary")
    if candidates.shape[-1] > _NAME_GROUP_LAST:
        first = definition_ids(candidates[..., _NAME_GROUP_FIRST], len(definitions))
        last = definition_ids(candidates[..., _NAME_GROUP_LAST], len(definitions))
        if not np.array_equal(first > 0, last > 0) or (first > last).any():
            raise ValueError("invalid naming group span")


def permanent_links(obs: ObservationBuffers, objects: int) -> None:
    """Validate one-based links into the policy's full object sequence."""
    for index in PERMANENT_LINK_INDEXES:
        links = np.concatenate(
            [
                obs[key][..., index]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            axis=1,
        )
        if not ((links == np.floor(links)) & (links >= 0) & (links <= objects)).all():
            raise ValueError("invalid permanent link object reference")


def actions(
    obs: ObservationBuffers,
    legal: np.ndarray,
    objects: int,
    name_definitions: np.ndarray,
    lookup: np.ndarray | None,
) -> None:
    """Validate only legal offers' continuation, naming and focus references."""
    batches, slots = np.nonzero(legal)
    indexes = obs["actions"][batches, slots, ACTION_CONTINUATION_INDEX]
    if not (
        (indexes == np.floor(indexes)) & (indexes >= 0) & (indexes <= RELATION_ROWS)
    ).all():
        raise ValueError("invalid action continuation reference")
    rows = obs["relations"]
    present = (
        (indexes[:, None] == rows[..., _RELATION_FRAME][batches])
        & (rows[..., RELATION_VALID_INDEX][batches] == 1)
    ).any(-1)
    if ((indexes != 0) & ~present).any():
        raise ValueError("action continuation reference has no frame")
    candidates = obs["name_candidates"][batches, slots]
    names(candidates, name_definitions)
    if lookup is not None:
        definition_ids(candidates[..., _NAME_DEFINITION], len(lookup), lookup)
    focus = obs["action_focus"][batches, slots]
    if not ((focus == np.floor(focus)) & (focus >= -1) & (focus < objects)).all():
        raise ValueError("invalid action focus reference")


def recent_events(
    obs: ObservationBuffers, object_count: int, definition_rows: np.ndarray, width: int
) -> None:
    """Reject malformed public history v1 before any forward; padding is ignored."""
    if "events" not in obs or "events_valid" not in obs:
        raise ValueError("recent-event ActorCritic requires events and events_valid")
    events, valid = obs["events"], obs["events_valid"]
    if events.ndim != 3 or events.shape[-1] != width or valid.shape != events.shape[:2]:
        raise ValueError(f"public history v1 requires [batch, events, {width}]")
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("event validity must be binary")
    rows = np.where(valid.astype(bool)[..., None], events, 0)
    if not np.isfinite(rows).all():
        raise ValueError("public history must be finite")

    def categorical(values: np.ndarray, size: int) -> None:
        if not ((values == np.floor(values)) & (values >= 0) & (values < size)).all():
            raise ValueError("invalid public history categorical field")

    categorical(rows[..., _EVENT_KIND], 6)
    # Source then target: definition, owner, zone, context, status.
    for start in _EVENT_SUBJECTS:
        ids, owner, zone, focus, status = (rows[..., start + i] for i in range(5))
        if not (
            (ids == np.floor(ids)) & (ids >= 0) & (ids < len(definition_rows))
        ).all():
            raise ValueError("unadmitted historical definition")
        if (definition_rows[ids.astype(np.int64)] < 0).any():
            raise ValueError("unadmitted historical definition")
        if not (
            (focus == np.floor(focus)) & (focus >= -1) & (focus < object_count)
        ).all():
            raise ValueError("invalid historical context reference")
        categorical(owner, 3)
        categorical(zone, 8)
        categorical(status, 3)


def beliefs(obs: ObservationBuffers, model: ModelConfig) -> None:
    """Admit explicit semantic belief rows, retaining their normalization contract."""
    keys = (
        "belief_card_def_ids",
        "belief_owner_role_ids",
        "belief_hidden_zone_ids",
        "belief_count_probabilities",
        "belief_validity",
        "belief_globals",
    )
    missing = sorted(set(keys).difference(obs))
    if missing:
        raise ValueError(f"belief-enabled ActorCritic is missing inputs: {missing}")
    card_ids = obs[keys[0]].astype(np.int64)
    owner_ids = obs[keys[1]].astype(np.int64)
    zone_ids = obs[keys[2]].astype(np.int64)
    probabilities = obs[keys[3]].astype(np.float32)
    validity = obs[keys[4]].astype(np.float32)
    globals_ = obs[keys[5]].astype(np.float32)
    if card_ids.ndim != 2:
        raise ValueError("belief ids must have shape [batch, rows]")
    expected_ids = card_ids.shape
    if owner_ids.shape != expected_ids or zone_ids.shape != expected_ids:
        raise ValueError("belief semantic ids must share shape [batch, rows]")
    if probabilities.shape != (*expected_ids, model.belief_count_buckets):
        raise ValueError(
            "belief count probabilities must have shape [batch, rows, buckets]"
        )
    if validity.shape != expected_ids:
        raise ValueError("belief validity must have shape [batch, rows]")
    if globals_.shape != (expected_ids[0], 2):
        raise ValueError("belief globals must have shape [batch, 2]")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("belief probabilities must be finite and non-negative")
    if not np.isfinite(globals_).all():
        raise ValueError("belief globals must be finite")
    valid_rows = validity > 0
    if valid_rows.any() and not np.allclose(
        probabilities.sum(axis=-1)[valid_rows], 1, atol=1e-5, rtol=0
    ):
        raise ValueError("valid belief rows must be normalized")
    for ids, size, label in (
        (card_ids, model.belief_card_vocab_size, "card definition"),
        (owner_ids, model.belief_owner_role_vocab_size, "owner role"),
        (zone_ids, model.belief_hidden_zone_vocab_size, "hidden zone"),
    ):
        if (ids < 0).any() or (ids >= size).any():
            raise ValueError(f"belief {label} id is outside the model vocabulary")
