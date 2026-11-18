"""Host lookup tables for encoding the running catalog and binding stored policies."""

from dataclasses import dataclass

import numpy as np

from manabot.policy.prepare import NUMBER_TOKEN_KINDS, RecordNamespace, record_key
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


def _record_tables(vocabulary: dict[tuple[int, int], int]) -> _RecordTables:
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


def catalog_tables(pack_key: str, *, records: bool) -> CatalogTables:
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
        if row in ancestors:
            raise ValueError("cyclic semantic definition reference is not admitted")
        start, stop = catalog.definition_offsets[row : row + 2]
        spans = [(int(start), int(stop))]
        for program in pack.program_rows(row):
            start, stop = catalog.program_offsets[program : program + 2]
            spans.append((int(start), int(stop)))
        tokens: list[int] = []
        for start, stop in spans:
            for offset in range(start, stop):
                tokens.append(vocabulary[pairs[offset]])
                if offset in references:
                    tokens.extend(
                        definition_tokens(references[offset], (*ancestors, row))
                    )
        return tokens

    # Structure, execution order, program boundaries and referenced token
    # definitions come from the checked catalog. Nothing is truncated.
    rows = [definition_tokens(row) for row in range(pack.definition_count)]
    width = max(map(len, rows))

    tables = _record_tables(vocabulary) if records else None
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
            [tables.numbers.get(kind, 0) for kind in range(6)]
        )
    return CatalogTables(
        buffers,
        tuple(sorted(vocabulary)),
        tables.size if tables else len(vocabulary) + 1,
    )
