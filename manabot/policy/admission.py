"""Host validation shared by policy backends before preparing observation indexes.

Native batches certify row boundaries and joins. Admission binds those rows to
the policy layout, saved vocabularies and optional memory and belief inputs.
"""

from dataclasses import dataclass

import numpy as np

from manabot.belief.encoding import BeliefRows
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import RetainedMemory
from manabot.env.observation import (
    FACT_COMBAT_RESTRICTION,
    FACT_DETAIL_INDEX,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_VALUE_INDEX,
    ObservationSpace,
    column,
)
from manabot.env.shared_effects import (
    NUMBER_TOKEN_KINDS,
    PROGRAM_TOKEN_KIND,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.buckets import (
    object_trimming,
)
from manabot.policy.decision_memory import (
    validate_retained_memory,
)
from manabot.policy.prepare import (
    NAMED_DETAIL_KINDS,
    ORDINAL_ID_KINDS,
    RecordNamespace,
    Tables,
    record_key,
)
from manabot.policy.readings import PROGRAM_KINDS, Readings
from manabot.policy.spans import (
    ObservationSelection,
    ObservationSpans,
    full_word,
    observation_spans,
)
import managym

# Native columns these checks read, by name.
_NAME_ROW = column("name_candidates", "name")
_NAME_DEFINITION = column("name_candidates", "definition")
_NAME_GROUP_FIRST = column("name_candidates", "group_first")
_NAME_GROUP_LAST = column("name_candidates", "group_last")
_DETAIL_FIELD = column("decision_reference_details", "field")
_DETAIL_TOKEN_KIND = column("decision_reference_details", "token_kind")
_DETAIL_VALUE = column("decision_reference_details", "value_high")


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


def history_config(config: ModelConfig, observation_space: ObservationSpace) -> None:
    """Check the optional history encoders against their native observation contract."""
    if (
        config.decision_memory != "none"
        and not observation_space.config.max_decision_records
    ):
        raise ValueError(
            "decision-memory model requires explicit observation capacities"
        )
    if config.decision_memory != "none" and not config.catalog_records():
        raise ValueError("decision memory requires a catalog world")
    if config.recent_events and (
        config.world is None or observation_space.config.policy_history_version != 1
    ):
        raise ValueError("recent events require world and policy_history_version=1")
    if (
        not config.recent_events
        and observation_space.config.policy_history_version != 0
    ):
        raise ValueError(
            "history observation contract requires recent_events ActorCritic"
        )


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


def admit(
    batch: managym.ObservationBatch | ObservationSelection,
    spec: Spec,
    *,
    bucketed: bool,
    belief: BeliefRows[np.ndarray] | None = None,
) -> ObservationSpans:
    """The encoder owns row boundaries and native joins. Admit model-dependent lookups."""
    rows = observation_spans(
        batch, spec.space, bucketed=bucketed, objects=object_trimming(spec.model)
    )
    if rows.memory is not None:
        check_memory_contract(rows.memory, spec)
    data = rows.tables
    beliefs(belief, spec.model, data["actions"].groups[0])
    lookup = spec.tables.definition_rows
    size = spec.definition_count if lookup is None else len(lookup)
    ids = [
        data["semantic_cards"].records,
        *(
            data[key].records["definition"]
            for key in ("agent_permanents", "opponent_permanents", "stack")
        ),
        *(
            data["relations"].records[name]
            for name in ("source_definition", "reference_definition")
        ),
    ]
    name_vocabulary(data["name_candidates"].integers, spec.tables.name_definitions)
    if lookup is not None:
        ids.extend(
            data[key].records["definition"]
            for key in ("known_hand", "known_library", "decklists", "name_candidates")
        )
    if spec.model.recent_events:
        ids.extend(
            data["events"].records[name]
            for name in ("source_definition", "target_definition")
        )
    if "decision_records" in data:
        refs = data["decision_references"].integers
        details = data["decision_reference_details"].integers
        ids.append(refs[:, column("decision_references", "definition")])
        definition = (details[:, _DETAIL_FIELD] == 0) & (
            details[:, _DETAIL_TOKEN_KIND] == 7
        )
        ids.append(full_word(details, _DETAIL_VALUE)[definition] + 1)
        headers = data["decision_records"].integers
        if (
            not spec.space.config.decision_memory_include_own
            and (headers[:, column("decision_records", "chooser_role")] == 1).any()
        ):
            raise ValueError("own-seat decision memory is disabled")
        named = data["decision_records"].records["name"].view(np.uint32)
        if (named >= len(spec.tables.name_definitions)).any():
            raise ValueError("invalid decision-memory name row")
    for values in ids:
        definition_ids(values, size, lookup)
    if spec.model.catalog_records():
        assert lookup is not None
        catalog = Catalog(
            lookup, spec.tables.record_name_keys, spec.tables.record_token_keys
        )
        if "decision_records" in data and spec.model.decision_memory != "none":
            headers = data["decision_records"].records
            chosen = headers["choice_kind_present"].astype(bool)
            for words in (headers["kind"], headers["choice_kind"][chosen]):
                catalog.require_names(RecordNamespace.TOKEN, words)
            details = data["decision_reference_details"].records
            program = details["field"] == 0
            catalog.require_tokens(
                details["token_kind"][program], details["value"][program]
            )
        live = data["facts"].integers
        kinds = live[:, FACT_KIND_INDEX]
        fact_vocabulary(live, kinds, np.isin(kinds, PROGRAM_KINDS), catalog)
        effects = data["effects"].records
        details = data["effect_details"].records
        effect_vocabulary(
            effects["modification"],
            effects["source_definition"],
            details["kind"],
            details["value"],
            catalog,
        )
    return rows


def check_memory_contract(memory: RetainedMemory, spec: Spec) -> None:
    if not all(
        key in spec.space.shapes and spec.space.shapes[key][0] == capacity
        for key, capacity in zip(MEMORY_FIELDS, memory.capacities, strict=True)
    ):
        raise ValueError("decision-memory capacity mismatch")
    validate_retained_memory(
        memory,
        spec.tables,
        spec.definition_count,
        spec.space.config.decision_memory_include_own,
    )


def definition_ids(
    ids: np.ndarray, size: int, lookup: np.ndarray | None = None
) -> np.ndarray:
    """Validate transport ids and return their optional catalog rows."""
    kind = "semantic" if lookup is not None else "card"
    message = f"policy received an unadmitted {kind} definition"
    if (ids.dtype.kind not in "iu" and not (ids == np.floor(ids)).all()) or not (
        (ids >= 0) & (ids < size)
    ).all():
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


def fact_vocabulary(
    rows: np.ndarray,
    kinds: np.ndarray,
    program: np.ndarray,
    catalog: Catalog,
) -> None:
    ident, detail = (
        full_word(rows, FACT_ID_INDEX),
        full_word(rows, FACT_DETAIL_INDEX),
    )
    named = (kinds > 0) & ~np.isin(kinds, ORDINAL_ID_KINDS)
    restriction = kinds == FACT_COMBAT_RESTRICTION
    named_detail = (np.isin(kinds, NAMED_DETAIL_KINDS) & (detail != 0)) | restriction
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


def effect_vocabulary(
    modification: np.ndarray,
    source: np.ndarray,
    kinds: np.ndarray,
    ids: np.ndarray,
    catalog: Catalog,
) -> None:
    catalog.require_names(RecordNamespace.TOKEN, modification)
    definition_ids(source, len(catalog.definitions), catalog.definitions)
    token_kinds = PROGRAM_TOKEN_KIND[kinds]
    program = token_kinds != 0
    catalog.require_names(RecordNamespace.TOKEN, ids[~program])
    catalog.require_tokens(token_kinds[program], ids[program])


def name_vocabulary(candidates: np.ndarray, definitions: np.ndarray) -> None:
    """Bind names, optional definitions and group endpoints to the checkpoint."""
    rows = definition_ids(candidates[..., _NAME_ROW], len(definitions))
    if not np.array_equal(
        candidates[..., _NAME_DEFINITION], definitions[rows].astype(candidates.dtype)
    ):
        raise ValueError("name candidate definition disagrees with vocabulary")
    if candidates.shape[-1] > _NAME_GROUP_LAST:
        definition_ids(
            candidates[..., _NAME_GROUP_FIRST : _NAME_GROUP_LAST + 1], len(definitions)
        )


def beliefs(
    rows: BeliefRows[np.ndarray] | None, model: ModelConfig, batch_size: int
) -> None:
    """Validate external marginals against the checkpoint and the game batch."""
    if rows is None:
        if model.belief_count_buckets:
            raise ValueError("belief-enabled ActorCritic requires belief rows")
        return
    if not model.belief_count_buckets:
        raise ValueError("belief rows supplied to a belief-free policy")
    card_ids, owner_ids, zone_ids = rows.cards, rows.owners, rows.zones
    probabilities, validity, globals_ = rows.probabilities, rows.validity, rows.globals
    if any(ids.dtype != np.int64 for ids in (card_ids, owner_ids, zone_ids)):
        raise ValueError("belief semantic ids must be int64")
    if any(
        values.dtype != np.float32 for values in (probabilities, validity, globals_)
    ):
        raise ValueError("belief probabilities, validity and globals must be float32")
    if card_ids.ndim != 2 or card_ids.shape[0] != batch_size:
        raise ValueError(
            "belief ids must match the observation batch size and have shape [batch, rows]"
        )
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
    if not ((validity == 0) | (validity == 1)).all():
        raise ValueError("belief validity must be binary")
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
