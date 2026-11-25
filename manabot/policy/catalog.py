"""Host lookup tables for encoding the running catalog and binding stored policies."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.env.shared_effects import NUMBER_TOKEN_KINDS
from manabot.infra.config import ModelConfig
from manabot.policy.prepare import (
    Layout,
    RecordNamespace,
    Tables,
    object_ownership,
    record_key,
)
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym


@dataclass(frozen=True)
class _RecordTables:
    """Token-embedding rows of record ids: `names` by (namespace, value),
    `tokens` by (kind, value) over card tokens and appended ones, `numbers` by
    numeric token kind. `size` is the embedding's row count."""

    names: dict[tuple[int, int], int]
    tokens: dict[tuple[int, int], int]
    numbers: dict[int, int]
    size: int


def _record_tables(
    vocabulary: dict[tuple[int, int], int], *, decision_memory: bool
) -> _RecordTables:
    """A name whose card token exists reads that token's row, so a granted
    keyword and the printed keyword share one embedding; every other name,
    token and numeric kind gets a row appended after the card tokens."""
    tokens = dict(vocabulary)
    names: dict[tuple[int, int], int] = {}
    size = len(vocabulary) + 1

    def append() -> int:
        nonlocal size
        size += 1
        return size - 1

    for space, name, value, kind, token in managym.record_vocabulary():
        if name.startswith("DecisionMemory::") and not decision_memory:
            continue
        key = (int(RecordNamespace[space.upper()]), value)
        if key in names:
            raise ValueError(f"record name {name!r} shares an id with another")
        if kind is None or token is None:
            names[key] = append()
        else:
            if (kind, token) not in tokens:
                tokens[(kind, token)] = append()
            names[key] = tokens[(kind, token)]
    numbers = {kind: append() for kind in NUMBER_TOKEN_KINDS}
    return _RecordTables(names, tokens, numbers, size)


def _sorted_keys(
    table: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.array(list(table), dtype=np.int64)
    keys = record_key(pairs[:, 0], pairs[:, 1])
    order = keys.argsort()
    return keys[order], np.asarray(list(table.values()))[order]


@dataclass(frozen=True)
class CatalogTables:
    buffers: dict[str, np.ndarray]
    vocabulary: tuple[tuple[int, int], ...]
    embedding_rows: int


def catalog_tables(
    pack_key: str, *, records: bool, decision_memory: bool
) -> CatalogTables:
    pack = bind_policy_catalog(pack_key)
    catalog = pack.catalog
    pairs = list(
        zip(map(int, catalog.token_kind), map(int, catalog.token_value), strict=True)
    )
    vocabulary = {pair: index + 1 for index, pair in enumerate(sorted(set(pairs)))}
    references = dict(
        zip(
            map(int, catalog.definition_ref_source_tokens),
            map(int, catalog.definition_ref_target_rows),
            strict=True,
        )
    )

    def definition_tokens(row: int, ancestors: tuple[int, ...] = ()) -> list[int]:
        start, stop = catalog.definition_offsets[row : row + 2]
        spans = [(int(start), int(stop))]
        for program in pack.program_rows(row):
            start, stop = catalog.program_offsets[program : program + 2]
            spans.append((int(start), int(stop)))
        tokens: list[int] = []
        for start, stop in spans:
            for offset in range(start, stop):
                tokens.append(vocabulary[pairs[offset]])
                # A referenced definition is inlined once per path. Meld
                # partners name each other, so a reference back to a definition
                # being written keeps only its reference token.
                target = references.get(offset)
                if target is not None and target not in (*ancestors, row):
                    tokens.extend(definition_tokens(target, (*ancestors, row)))
        return tokens

    # Structure, execution order, program boundaries and referenced token
    # definitions come from the checked catalog. Nothing is truncated.
    rows = [definition_tokens(row) for row in range(pack.definition_count)]
    width = max(map(len, rows))

    tables = (
        _record_tables(vocabulary, decision_memory=decision_memory) if records else None
    )
    # Appended record rows follow the card tokens, whose rows are unchanged.
    buffers = {
        "tokens": np.asarray(
            [row + [0] * (width - len(row)) for row in rows], dtype=np.int64
        ),
        "lengths": np.asarray(list(map(len, rows)), dtype=np.int64),
    }
    lookup = np.full(max(pack.definition_row_by_card_def_id) + 2, -1, dtype=np.int64)
    lookup[0] = 0
    for definition, row in pack.definition_row_by_card_def_id.items():
        lookup[definition + 1] = row + 1
    buffers["definition_rows"] = lookup
    if tables is not None:
        for prefix, table in (
            ("record_name", tables.names),
            ("record_token", tables.tokens),
        ):
            buffers[f"{prefix}_keys"], buffers[f"{prefix}_rows"] = _sorted_keys(table)
        buffers["record_number_rows"] = np.asarray(
            [tables.numbers.get(kind, 0) for kind in range(max(tables.numbers) + 1)]
        )
    return CatalogTables(
        buffers,
        tuple(sorted(vocabulary)),
        tables.size if tables else len(vocabulary) + 1,
    )


def name_definitions(candidates: Sequence[managym.NameCandidate]) -> np.ndarray:
    """One-based definition joins for the unique public names. Zero denotes absence."""
    if not candidates or len({candidate.name for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("naming vocabulary must be nonempty and unique")
    return np.asarray(
        [0]
        + [
            candidate.definition + 1 if candidate.definition is not None else 0
            for candidate in candidates
        ],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class PolicyBuffers:
    """Fresh lookup arrays before a backend registers them as model buffers."""

    catalog: CatalogTables | None
    definitions: np.ndarray
    program_rows: np.ndarray | None
    ownership: np.ndarray

    @property
    def token_rows(self) -> int:
        return self.catalog.embedding_rows if self.catalog is not None else 0

    def preparation_tables(self, layout: Layout) -> Tables:
        buffers = {
            "_is_agent_template": self.ownership,
            "name_candidates.definitions": self.definitions,
        }
        if self.catalog is not None:
            buffers.update(
                {
                    f"semantic_cards.{name}": value
                    for name, value in self.catalog.buffers.items()
                }
            )
        if self.program_rows is not None:
            buffers["name_candidates.program_rows"] = self.program_rows
        return Tables.from_buffers(buffers, layout)


def fresh_policy_tables(space: ObservationSpace, model: ModelConfig) -> PolicyBuffers:
    catalog = (
        catalog_tables(
            model.world,
            records=model.catalog_records(),
            decision_memory=bool(space.config.max_decision_records),
        )
        if model.world is not None
        else None
    )
    definitions = name_definitions(managym.naming_catalog())
    return PolicyBuffers(
        catalog,
        definitions,
        catalog.buffers["definition_rows"][definitions]
        if catalog is not None
        else None,
        object_ownership(
            space.config.max_cards_per_player, space.config.max_permanents_per_player
        )[None],
    )
