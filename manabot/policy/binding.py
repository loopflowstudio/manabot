"""Bind a writer's expanded card programs and learned tables to the running world.

Runtime joins use the running vocabulary. Retired rows follow that vocabulary
so that a stored program can still read every vector it was trained with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from manabot.model.writer_catalog import WriterCatalog
from manabot.policy.catalog import catalog_tables
from manabot.policy.prepare import RecordNamespace, record_key
import managym

if TYPE_CHECKING:
    from manabot.policy.checkpoint import PolicyCheckpoint


@dataclass(frozen=True)
class BindingReport:
    """What binding moved, for the load log; `changed_programs` counts stored
    programs that differ from the running catalog's spelling."""

    stored_programs: int
    changed_programs: int
    new_tokens: int
    new_names: int
    retired_keys: int
    renamed_keys: int

    def summary(self) -> str:
        return (
            f"{self.stored_programs} stored programs, {self.changed_programs} changed; "
            f"keys renamed {self.renamed_keys}, retired {self.retired_keys}; "
            f"{self.new_tokens} new token rows, {self.new_names} new name rows"
        )


def packed(kind: int, value: int) -> int:
    return int(record_key(kind, np.asarray(value)))


def successors() -> dict[str, dict[int, int | None]]:
    tokens, records = managym.catalog_token_renames()
    families = {
        "record_token": tokens,
        "record_name": [
            (int(RecordNamespace[space.upper()]), old, new)
            for space, old, new in records
        ],
    }
    return {
        family: {
            packed(kind, managym.catalog_token_value(old)): (
                None if new is None else packed(kind, managym.catalog_token_value(new))
            )
            for kind, old, new in entries
        }
        for family, entries in families.items()
    }


def named_rows(
    weights: np.ndarray, old: Sequence[str], current: Sequence[str]
) -> np.ndarray:
    if len(old) + 1 != len(weights) or len(set(old)) != len(old):
        raise ValueError("writer names do not identify every learned row")
    missing = set(old).difference(current)
    if missing:
        raise ValueError(f"missing learned naming rows: {sorted(missing)}")
    indices = {name: i for i, name in enumerate(old, 1)}
    mean = weights[1:].mean(axis=0)
    return np.stack(
        [weights[0], *(weights[indices[n]] if n in indices else mean for n in current)]
    )


def bind_catalog(
    checkpoint: PolicyCheckpoint, writer: WriterCatalog, current: WriterCatalog
) -> tuple[PolicyCheckpoint, BindingReport]:
    """Return the policy with current public joins and unchanged trained programs.

    `model_interface` has admitted the checkpoint’s catalog records."""
    catalog = catalog_tables(managym.WORLD_SETUP, records=True)
    buffers = catalog.buffers.copy()
    prefix = "semantic_cards."
    old = {
        k.removeprefix(prefix): v
        for k, v in checkpoint.buffers.items()
        if k.startswith(prefix)
    }
    params = dict(checkpoint.parameters)
    token_param = prefix + "token_embedding.weight"
    weights = params[token_param]
    definitions = writer["definition_keys"]
    now = {key: i for i, key in enumerate(current["definition_keys"])}
    if len(definitions) != len(old["tokens"]) or len(set(definitions)) != len(
        definitions
    ):
        raise ValueError("writer catalog does not identify each stored program")
    # Index by runtime embedding row. Distinct token and record keys can
    # deliberately address the same embedding; both joins must agree.
    move = np.full(catalog.embedding_rows, -1, dtype=np.int64)
    move[0] = 0

    def assign(target: int, source: int) -> None:
        if not 0 < source < len(weights) or move[target] not in (-1, source):
            raise ValueError(
                f"ambiguous or invalid learned token row {source} for runtime row {target}"
            )
        move[target] = source

    retired = renamed = 0
    renames = successors()
    for family, changes in renames.items():
        keys, rows = family + "_keys", family + "_rows"
        runtime = dict(zip(buffers[keys].tolist(), buffers[rows].tolist(), strict=True))
        for key, row in zip(old[keys].tolist(), old[rows].tolist(), strict=True):
            kind, value = divmod(key, 1 << 32)
            if family == "record_token" and kind == 7:
                if value >= len(definitions) or definitions[value] not in now:
                    raise ValueError(
                        f"missing referenced definition for writer row {value}"
                    )
                key = packed(kind, now[definitions[value]])
                if key not in runtime:
                    # The definition still exists but no running card references it.
                    continue
            if key not in runtime:
                if key not in changes:
                    raise ValueError(
                        f"unexplained {family} key {key}; classify it in managym/src/cards/token_renames.rs"
                    )
                key = changes[key]
                if key is None:
                    retired += 1
                    continue
                renamed += 1
                if key not in runtime:
                    raise ValueError(f"missing successor {family} key {key}")
            assign(runtime[key], row)
    for source, target in zip(
        old["record_number_rows"], buffers["record_number_rows"], strict=True
    ):
        if target:
            assign(target, source)
    unseen = move == -1
    count = int(unseen.sum())
    retained = np.flatnonzero(~unseen)
    writer_to_bound = np.full(len(weights), -1, dtype=np.int64)
    writer_to_bound[move[retained]] = retained
    retired_rows = np.flatnonzero(writer_to_bound < 0)
    writer_to_bound[retired_rows] = np.arange(len(move), len(move) + len(retired_rows))
    table = np.empty(
        (len(move) + len(retired_rows), weights.shape[1]), dtype=weights.dtype
    )
    table[retained] = weights[move[retained]]
    table[np.flatnonzero(unseen)] = weights[1:].mean(axis=0)
    table[len(move) :] = weights[retired_rows]
    params[token_param] = table
    lengths = buffers["lengths"].copy()
    programs = buffers["tokens"]
    width = max(programs.shape[1], old["tokens"].shape[1])
    programs = np.pad(programs, ((0, 0), (0, width - programs.shape[1])))
    changed = 0
    for index, key in enumerate(definitions):
        if key not in now:
            continue
        row = now[key]
        length = int(old["lengths"][index])
        stored = old["tokens"][index, :length]
        if not 0 < length <= old["tokens"].shape[1] or np.any(
            (stored < 1) | (stored >= len(weights))
        ):
            raise ValueError(f"invalid stored card program for {key}")
        stored = writer_to_bound[stored]
        if not np.array_equal(programs[row, : lengths[row]], stored):
            changed += 1
        programs[row].fill(0)
        programs[row, :length] = stored
        lengths[row] = length
    buffers.update(tokens=programs, lengths=lengths)
    stored_names = [item["name"] for item in writer["naming_catalog"]]
    names = [item["name"] for item in current["naming_catalog"]]
    name_param = "name_candidates.embedding.weight"
    params[name_param] = named_rows(params[name_param], stored_names, names)
    name_defs = np.array(
        [0]
        + [
            0 if item["definition"] is None else item["definition"] + 1
            for item in current["naming_catalog"]
        ],
        dtype=np.int64,
    )
    result = dict(checkpoint.buffers)
    result.update({prefix + name: value for name, value in buffers.items()})
    result["name_candidates.definitions"] = name_defs
    result["name_candidates.program_rows"] = buffers["definition_rows"][name_defs]
    return replace(checkpoint, parameters=params, buffers=result), BindingReport(
        stored_programs=len(now.keys() & definitions),
        changed_programs=changed,
        new_tokens=count,
        new_names=len(set(names) - set(stored_names)),
        retired_keys=retired,
        renamed_keys=renamed,
    )
