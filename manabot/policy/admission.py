"""Host validation shared by policy backends before preparing observation indexes.

The checks here own definition/vocabulary joins, fact programs, effect partitions,
continuation rows and legal-action references. Torch's component admission APIs
adapt tensors to these NumPy checks; MLX checks raw rollout rows before `prepare`
turns validity into booleans or performs lookups. Padding keeps the same meaning
in both paths: ignored where a validity column owns it, checked otherwise.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from manabot.env.observation import (
    FACT_COMBAT_RESTRICTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_COPY_EXCEPTION,
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_DURATION_INDEX,
    FACT_EFFECT,
    FACT_EFFECT_PROGRAM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS,
    FACT_OBJECT_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    PERMANENT_LINK_INDEXES,
    RELATION_ROWS,
    RELATION_VALID_INDEX,
)
from manabot.env.shared_effects import (
    DETAIL_DIM,
    DETAIL_ROWS,
    EFFECT_DIM,
    EFFECT_ROWS,
    MEMBER_DIM,
    MEMBER_ROWS,
)


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


def _key(high: np.ndarray | int, low: np.ndarray) -> np.ndarray:
    return (np.asarray(high, dtype=np.int64) << 32) | (
        low.astype(np.int64) & 0xFFFFFFFF
    )


def _require_keys(keys: np.ndarray, query: np.ndarray, message: str) -> None:
    if query.size == 0:
        return
    indexes = np.searchsorted(keys, query).clip(max=len(keys) - 1)
    if not len(keys) or not np.array_equal(keys[indexes], query):
        raise ValueError(message)


@dataclass(frozen=True)
class Catalog:
    """Only the immutable lookup buffers needed for admission, never weights."""

    definitions: np.ndarray
    names: np.ndarray
    tokens: np.ndarray

    def require_names(self, namespace: np.ndarray | int, words: np.ndarray) -> None:
        _require_keys(
            self.names,
            _key(namespace, words).reshape(-1),
            "policy received an id outside the record vocabulary",
        )

    def require_tokens(self, kinds: np.ndarray, words: np.ndarray) -> None:
        kinds, words = kinds.astype(np.int64).reshape(-1), words.reshape(-1)
        number = (kinds == 4) | (kinds == 5)
        if ((kinds < 1) | (kinds > 9)).any():
            raise ValueError("policy received an invalid program token kind")
        _require_keys(
            self.tokens,
            _key(kinds[~number], words[~number]),
            "policy received a program token outside the vocabulary",
        )


def _word(rows: np.ndarray, column: int) -> np.ndarray:
    return (rows[..., column].astype(np.int64) << 16) | rows[..., column + 1].astype(
        np.int64
    )


def facts(
    rows: np.ndarray, max_facts: int, objects: int, catalog: Catalog | None
) -> None:
    """Validate fact joins, kinds, catalog ids and consecutive program runs."""
    if rows.shape[1:] != (max_facts, FACT_DIM):
        raise ValueError(f"expected [batch, {max_facts}, {FACT_DIM}] facts")
    valid = rows[..., FACT_VALID_INDEX]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("fact validity must be binary")
    rows = np.where(valid.astype(bool)[..., None], rows, 0)
    links, kinds = rows[..., FACT_OBJECT_INDEX], rows[..., FACT_KIND_INDEX]
    if not ((links == np.floor(links)) & (links >= 0) & (links <= objects)).all():
        raise ValueError("invalid fact object reference")
    if not ((kinds == np.floor(kinds)) & (kinds >= 0) & (kinds <= FACT_KINDS)).all():
        raise ValueError("invalid fact kind")
    program = np.isin(kinds, (FACT_EFFECT_PROGRAM, FACT_CONTINUATION_PROGRAM))
    if catalog is not None:
        ident, detail = _word(rows, FACT_ID_INDEX), _word(rows, FACT_DETAIL_INDEX)
        named = (kinds > 0) & ~np.isin(
            kinds,
            (
                FACT_TRANSFORMS,
                FACT_USED_ABILITY,
                FACT_CONTINUATION_PROGRAM,
                FACT_COMBAT_RESTRICTION,
            ),
        )
        restriction = kinds == FACT_COMBAT_RESTRICTION
        named_detail = (
            np.isin(kinds, (FACT_EFFECT, FACT_COPY_EXCEPTION)) & (detail != 0)
        ) | restriction
        if (~(program | named_detail) & (detail != 0)).any():
            raise ValueError("fact kind carries no detail")
        catalog.require_names(
            np.where(kinds == FACT_SET_NAME, 2, 1)[named], ident[named]
        )
        catalog.require_names(
            np.where(restriction, 3, 1)[named_detail], detail[named_detail]
        )
        catalog.require_tokens(rows[..., FACT_VALUE_INDEX][program], detail[program])
        definition_ids(
            rows[..., FACT_SOURCE_INDEX], len(catalog.definitions), catalog.definitions
        )
    if not program.any():
        return
    positions, selected = np.argwhere(program), rows[program]
    ordinals = selected[:, FACT_VALUE_INDEX + 1]
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
) -> None:
    """Validate live headers and their complete CSR detail/member partitions."""
    if headers.ndim != 3 or headers.shape[1:] != (EFFECT_ROWS, EFFECT_DIM):
        raise ValueError(f"expected [batch, {EFFECT_ROWS}, {EFFECT_DIM}] state records")
    valid = headers[..., 17]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("state record validity must be binary")
    host = np.where(valid.astype(bool)[..., None], headers, 0)
    if not np.isfinite(host).all():
        raise ValueError("state records must be finite")
    batch = len(host)
    if details.shape != (batch, DETAIL_ROWS, DETAIL_DIM) or members.shape != (
        batch,
        MEMBER_ROWS,
        MEMBER_DIM,
    ):
        raise ValueError("invalid shared effect buffer shape")
    positions = np.argwhere(host[..., 17])
    if not len(positions):
        return
    live = host[positions[:, 0], positions[:, 1]]
    halves = _indexes(live[:, :2], 65535, "effect vocabulary half")
    source = _indexes(live[:, 11], 2**24, "effect source definition")
    links = _indexes(live[:, [10, 12]], objects, "effect source/controller")
    if (links[:, 0] == 0).any():
        raise ValueError("effect controller is absent")
    offsets = _indexes(
        live[:, 13:17], max(MEMBER_ROWS, DETAIL_ROWS), "effect CSR offset/count"
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
        detail_indexes.extend(range(b * DETAIL_ROWS + d, b * DETAIL_ROWS + d_end))
        member_indexes.extend(range(b * MEMBER_ROWS + m, b * MEMBER_ROWS + m_end))
        member_records.extend([i] * targets)
    detail = details.reshape(-1, DETAIL_DIM)[detail_indexes]
    kinds = _indexes(detail[:, 0], 19, "effect detail kind")
    ids = _indexes(detail[:, 1:], 65535, "effect detail half")
    if ((kinds == 0) | ((kinds > 3) & (kinds < 11))).any():
        raise ValueError("live effect detail has no kind")
    member = members.reshape(-1, MEMBER_DIM)[member_indexes]
    _indexes(member[:, 0], objects, "effect member")
    orders = _indexes(member[:, 1], MEMBER_ROWS, "effect membership order")
    if (member[:, 0] == 0).any():
        raise ValueError("effect membership is absent")
    if (orders == 0).any():
        raise ValueError("effect membership order is absent")
    global_orders = positions[:, 0][member_records] * (MEMBER_ROWS + 1) + orders
    if len(np.unique(global_orders)) != len(global_orders):
        raise ValueError("duplicate effect membership order")
    if catalog is not None:
        catalog.require_names(1, (halves[:, 0] << 16) | halves[:, 1])
        definition_ids(source, len(catalog.definitions), catalog.definitions)
        unsigned, program = (ids[:, 0] << 16) | ids[:, 1], kinds > 10
        catalog.require_names(1, unsigned[~program])
        catalog.require_tokens(kinds[program] - 10, unsigned[program])


def relations(rows: np.ndarray, width: int, objects: int) -> None:
    """Validate continuation validity, finite payloads, object joins and frames."""
    if rows.shape[1:] != (RELATION_ROWS, width):
        raise ValueError(f"expected [batch, 64, {width}] relations")
    valid = rows[..., RELATION_VALID_INDEX]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("relation validity must be binary")
    clean = np.where(valid.astype(bool)[..., None], rows, 0)
    if not np.isfinite(clean).all():
        raise ValueError("relations must be finite")
    for column in (6, 24):
        values = clean[..., column]
        if not (
            (values == np.floor(values)) & (values >= 0) & (values <= objects)
        ).all():
            raise ValueError("invalid continuation object reference")
    ids = rows[..., 1]
    if not (
        ((ids == np.floor(ids)) & (ids >= 1) & (ids <= RELATION_ROWS)) | (valid == 0)
    ).all():
        raise ValueError("invalid continuation frame reference")


def names(candidates: np.ndarray, definitions: np.ndarray) -> None:
    """Validate name rows, optional definitions and balanced-group endpoints."""
    rows = definition_ids(candidates[..., 0], len(definitions))
    if not np.array_equal(
        candidates[..., 1], definitions[rows].astype(candidates.dtype)
    ):
        raise ValueError("name candidate definition disagrees with vocabulary")
    if candidates.shape[-1] == 4:
        first = definition_ids(candidates[..., 2], len(definitions))
        last = definition_ids(candidates[..., 3], len(definitions))
        if not np.array_equal(first > 0, last > 0) or (first > last).any():
            raise ValueError("invalid naming group span")


def permanent_links(obs: Mapping[str, np.ndarray], objects: int) -> None:
    """Validate one-based links into the policy's full object sequence."""
    for column in PERMANENT_LINK_INDEXES:
        links = np.concatenate(
            [
                obs[key][..., column]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            axis=1,
        )
        if not ((links == np.floor(links)) & (links >= 0) & (links <= objects)).all():
            raise ValueError("invalid permanent link object reference")


def actions(
    obs: Mapping[str, np.ndarray],
    legal: np.ndarray,
    objects: int,
    name_definitions: np.ndarray,
    lookup: np.ndarray | None,
) -> None:
    """Validate only legal offers' continuation, naming and focus references."""
    batches, slots = np.nonzero(legal)
    indexes = obs["actions"][batches, slots][..., -2]
    if not (
        (indexes == np.floor(indexes)) & (indexes >= 0) & (indexes <= RELATION_ROWS)
    ).all():
        raise ValueError("invalid action continuation reference")
    rows = obs["relations"]
    present = (
        (indexes[:, None] == rows[..., 1][batches])
        & (rows[..., RELATION_VALID_INDEX][batches] == 1)
    ).any(-1)
    if ((indexes != 0) & ~present).any():
        raise ValueError("action continuation reference has no frame")
    candidates = obs["name_candidates"][batches, slots]
    names(candidates, name_definitions)
    if lookup is not None:
        definition_ids(candidates[..., 1], len(lookup), lookup)
    focus = obs["action_focus"][batches, slots]
    if not ((focus == np.floor(focus)) & (focus >= -1) & (focus < objects)).all():
        raise ValueError("invalid action focus reference")
