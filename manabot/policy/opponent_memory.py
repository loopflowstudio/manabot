"""Admission and host routes for whole-prefix decision memory.

Record slices partition each viewer's detail table. Preparation packs only real
records, preserving chronology, and creates the GRU schedule before transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Packed, Tables


@dataclass(frozen=True)
class MemoryRows[A]:
    numbers: A
    kinds: A
    detail_names: A
    detail_values: A
    detail_numbers: A
    detail_definitions: A
    detail_name_rows: A
    detail_records: A
    record_batches: A
    record_slots: A
    nonempty_batches: A
    schedule: Packed[A]
    lengths: tuple[int, ...]
    width: int


def admit_memory(obs: "ObservationBuffers") -> None:
    """Reject malformed partitions before any lookup or device transfer."""
    present = [key in obs for key in ("opponent_records", "opponent_details")]
    if not any(present):
        return
    if not all(present):
        raise ValueError("opponent memory requires both companion buffers")
    records, details = obs["opponent_records"], obs["opponent_details"]
    if (
        records.ndim != 3
        or details.ndim != 3
        or records.shape[-1] != 18
        or details.shape[-1] != 8
        or records.shape[0] != details.shape[0]
    ):
        raise ValueError("invalid opponent-memory shapes or integer values")
    if not np.isin(records[..., 17], (0, 1)).all():
        raise ValueError("invalid opponent-memory validity")
    for headers, rows in zip(records, details, strict=True):
        length = int(headers[:, 17].sum())
        if not (headers[:length, 17] == 1).all() or headers[length:].any():
            raise ValueError("opponent records must be a chronological prefix")
        # Zero padding also rejects NaN/Inf/fractional garbage. Validate numeric
        # values only on the owned prefix, avoiding repeated capacity-sized
        # scans and floor allocations for million-row detail buffers.
        live = headers[:length]
        if not np.isfinite(live).all() or not (live == np.floor(live)).all():
            raise ValueError("invalid opponent-memory integer values")
        cursor = 0
        for ordinal, header in enumerate(headers[:length]):
            if int(header[0]) * 65536 + int(header[1]) != ordinal:
                raise ValueError("invalid opponent-record ordinal")
            start, count = map(int, header[14:16])
            if start != cursor or count < 1 or start + count > len(rows):
                raise ValueError("invalid opponent-memory detail partition")
            cursor += count
            count_value = int(header[2]) * 65536 + int(header[3])
            first = int(header[6]) * 65536 + int(header[7])
            last = int(header[8]) * 65536 + int(header[9])
            if count_value < 1 or last - first + 1 != count_value or header[16] != 2:
                raise ValueError("invalid opponent-memory multiplicity or chooser")
        if rows[cursor:].any():
            raise ValueError("unowned opponent-memory detail rows")
        if length:
            words = headers[:length, [*range(10), 12, 13]]
            if not ((words >= 0) & (words <= 65535)).all():
                raise ValueError("invalid opponent-memory integer words")
            if (
                not np.isin(headers[:length, 10], range(5)).all()
                or not np.isin(headers[:length, 11], range(12)).all()
            ):
                raise ValueError("invalid opponent-memory phase or step")
        if cursor:
            live = rows[:cursor]
            if not np.isfinite(live).all() or not (live == np.floor(live)).all():
                raise ValueError("invalid opponent-memory detail integer values")
            words = rows[:cursor, [2, 3, 5, 6]]
            if not ((words >= 0) & (words <= 65535)).all():
                raise ValueError("invalid opponent-memory detail words")
            if not np.isin(
                rows[:cursor, 4], (-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9)
            ).all():
                raise ValueError("invalid opponent-memory detail token kind")
            if (
                not np.isin(rows[:cursor, 0], (0, 1, 2, 3, 4)).all()
                or (rows[:cursor, [1, 7]] < 0).any()
            ):
                raise ValueError("invalid opponent-memory detail coordinates")


def prepare_memory(
    obs: "ObservationBuffers", tables: "Tables"
) -> MemoryRows[np.ndarray] | None:
    from manabot.policy.prepare import NUMBER_TOKEN_KINDS, pack, record_word

    if "opponent_records" not in obs:
        return None
    headers = obs["opponent_records"]
    lengths = headers[..., 17].sum(1).astype(np.int64)
    batches, slots = np.nonzero(headers[..., 17])
    selected = headers[batches, slots]
    counts = np.stack(
        [record_word(selected[:, i : i + 2], 0) for i in (0, 2, 4, 6, 8)], -1
    )
    # Monotone signed-log scaling preserves exact transport while avoiding gate
    # saturation. Age/order are explicit, independent of allocated capacity.
    age = lengths[batches] - 1 - slots
    numbers = np.concatenate(
        (
            np.sign(counts) * np.log1p(np.abs(counts)) / 10,
            selected[:, 10:12] / np.array([4, 12]),
            age[:, None] / 128,
        ),
        -1,
    ).astype(np.float32)
    kinds = tables.names(1, record_word(selected[:, 12:14], 0))
    owned: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    for index, (batch, header) in enumerate(zip(batches, selected, strict=True)):
        start, count = map(int, header[14:16])
        owned.append(obs["opponent_details"][batch, start : start + count])
        routes.append(np.full(count, index, np.int64))
    details = np.concatenate(owned) if owned else np.zeros((0, 8), np.float32)
    detail_names = tables.names(1, record_word(details[:, 2:4], 0))
    payloads = record_word(details[:, 5:7], 0)
    token_kinds = details[:, 4].astype(np.int64)
    values = np.zeros(len(details), np.int64)
    names = token_kinds == -2
    values[names] = tables.names(1, payloads[names])
    programs = token_kinds >= 0
    values[programs] = tables.tokens(token_kinds[programs], payloads[programs])
    definitions = np.zeros(len(details), np.int64)
    definition_mask = token_kinds == -3
    definitions[definition_mask] = tables.rows(payloads[definition_mask])
    name_rows = np.where(token_kinds == -4, payloads, 0)
    if (name_rows < 0).any() or (name_rows >= len(tables.name_definitions)).any():
        raise ValueError("invalid opponent-memory name row")
    numeric = (token_kinds == -1) | np.isin(token_kinds, NUMBER_TOKEN_KINDS)
    signed = payloads.astype(np.uint32).view(np.int32).astype(np.float32)
    numeric_values = np.where(
        numeric, np.sign(signed) * np.log1p(np.abs(signed)) / 10, 0
    )
    return MemoryRows(
        numbers,
        kinds,
        detail_names,
        values,
        np.stack(
            (
                details[:, 0] / 4,
                details[:, 1] / 128,
                details[:, 7] / 128,
                numeric_values,
            ),
            -1,
        ).astype(np.float32),
        definitions,
        name_rows,
        np.concatenate(routes) if routes else np.zeros(0, np.int64),
        batches.astype(np.int64),
        slots.astype(np.int64),
        np.flatnonzero(lengths).astype(np.int64),
        pack(lengths[lengths > 0]),
        tuple(map(int, lengths)),
        max(1, int(lengths.max(initial=0))),
    )
