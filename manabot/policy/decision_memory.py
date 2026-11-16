"""Admission and host routes for shared historical reference snapshots.

Every local join is resolved here, before forward. Integer table identities
never become numeric features. Selected prefixes retain ordered repeated members.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS, MEMORY_WIDTHS, integer_words

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Packed, Tables

_FIELDS = MEMORY_FIELDS
_WIDTHS = tuple(MEMORY_WIDTHS[key] for key in _FIELDS)


@dataclass(frozen=True)
class MemoryRows[A]:
    numbers: A
    kinds: A
    choices: A
    choice_names: A
    reference_numbers: A
    reference_definitions: A
    detail_fields: A
    detail_values: A
    detail_numbers: A
    detail_references: A
    member_references: A
    member_groups: A
    member_positions: A
    record_batches: A
    record_slots: A
    nonempty_batches: A
    schedule: Packed[A]
    lengths: tuple[int, ...]
    width: int


def memory_lengths(obs: "ObservationBuffers", row: int) -> tuple[int, int, int, int]:
    headers, details, references = (obs[key][row] for key in _FIELDS[:3])
    return (
        int(headers[:, 19].sum()),
        int(details[:, 5].sum()),
        int(references[:, 12].sum()),
        int(headers[:, 15:18].sum()),
    )


def _integers(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError("invalid decision-memory integer values")


def _words(values: np.ndarray) -> None:
    if not ((values >= 0) & (values <= 65535)).all():
        raise ValueError("invalid decision-memory integer words")


def _live(rows: np.ndarray, valid: int) -> np.ndarray:
    if not np.isin(rows[:, valid], (0, 1)).all():
        raise ValueError("invalid decision-memory validity")
    count = int(rows[:, valid].sum())
    if not (rows[:count, valid] == 1).all() or rows[count:].any():
        raise ValueError(
            "decision-memory rows must be a chronological prefix with zero padding"
        )
    live = rows[:count]
    _integers(live)
    return live


def admit_memory(obs: "ObservationBuffers") -> None:
    present = [key in obs for key in _FIELDS]
    if not any(present):
        return
    if not all(present):
        raise ValueError("decision memory requires all companion buffers")
    arrays = [obs[key] for key in _FIELDS]
    if any(
        a.ndim != 3 or a.shape[-1] != w or a.shape[0] != arrays[0].shape[0]
        for a, w in zip(arrays, _WIDTHS, strict=True)
    ):
        raise ValueError("invalid decision-memory shapes")
    for headers, rows, snapshots, joins in zip(*arrays, strict=True):
        records, details, references = (
            _live(headers, 19),
            _live(rows, 5),
            _live(snapshots, 12),
        )
        _words(
            records[
                :, [*range(10), 12, 13, *range(20, 60), 61, 62, 64, 65, 67, 68, 70, 71]
            ]
        )
        _words(references[:, 2:6])
        _words(details[:, 2:4])
        if (
            not np.isin(records[:, 10], range(5)).all()
            or not np.isin(records[:, 11], range(12)).all()
            or not np.isin(records[:, 18], (1, 2)).all()
        ):
            raise ValueError("invalid decision-memory phase, step or chooser")
        if not np.isin(records[:, [60, 63, 66, 69, 72, 73]], (0, 1)).all():
            raise ValueError("invalid decision-memory selector presence")
        for offset in (60, 63, 66, 69):
            if records[records[:, offset] == 0, offset + 1 : offset + 3].any():
                raise ValueError("unavailable decision-memory selector has a value")
        if records[records[:, 72] == 0, 73].any():
            raise ValueError("unavailable decision-memory declaration has a value")
        cursor = 0
        last = -1
        seen: set[int] = set()
        for ordinal, header in enumerate(records):

            def word(offset: int) -> int:
                return int(header[offset]) * 65536 + int(header[offset + 1])

            if (
                word(0) != ordinal
                or word(2) < 1
                or word(6) != last + 1
                or word(8) - word(6) + 1 != word(2)
            ):
                raise ValueError(
                    "invalid decision-memory ordinal, span or multiplicity"
                )
            last = word(8)
            if header[14] != cursor or (header[15:18] < 0).any():
                raise ValueError("invalid decision-memory member partition")
            for count in header[15:18].astype(np.int64):
                group = joins[cursor : cursor + count]
                if len(group) != count:
                    raise ValueError("invalid decision-memory member partition")
                _integers(group)
                if (
                    not np.array_equal(group[:, 1], np.arange(count))
                    or ((group[:, 0] < 1) | (group[:, 0] > len(references))).any()
                ):
                    raise ValueError("invalid decision-memory member join or ordinal")
                for index in group[:, 0].astype(np.int64):
                    if int(index) not in seen:
                        if int(index) != len(seen) + 1:
                            raise ValueError(
                                "decision-memory snapshots must be ordered by first occurrence"
                            )
                        seen.add(int(index))
                cursor += int(count)
        if joins[cursor:].any() or len(seen) != len(references):
            raise ValueError("unowned decision-memory member or snapshot rows")
        if (
            not np.isin(references[:, 13], (0, 1)).all()
            or ((references[:, 13] == 1) & (references[:, 1] == 0)).any()
        ):
            raise ValueError("invalid decision-memory definition shortcut")
        cursor = 0
        for reference in references:
            start, count = map(int, reference[10:12])
            if start != cursor or count < 0 or start + count > len(details):
                raise ValueError("invalid decision-memory reference detail partition")
            if (
                reference[1] < 0
                or not np.isin(reference[0], (0, 1, 2))
                or not np.isin(reference[6:10], (0, 1)).all()
            ):
                raise ValueError("invalid decision-memory reference features")
            if reference[0] == 0 and (
                reference[:10].any() or reference[11] != 0 or reference[13] != 0
            ):
                raise ValueError("unavailable decision-memory reference has features")
            part = details[start : start + count]
            programs = int((part[:, 0] == 0).sum())
            if (
                not (part[:programs, 0] == 0).all()
                or not (part[programs:, 0] == 1).all()
                or not np.isin(part[:programs, 1], range(1, 10)).all()
                or not (part[programs:, 1] == -1).all()
            ):
                raise ValueError("invalid decision-memory program or mana kind")
            if (
                not np.array_equal(part[:programs, 4], np.arange(programs))
                or not np.array_equal(part[programs:, 4], np.arange(count - programs))
                or (count - programs) % 6
            ):
                raise ValueError("invalid decision-memory program or mana coordinates")
            if reference[13] and programs:
                raise ValueError("definition-only snapshot also owns a program")
            boolean = part[:, 1] == 5
            if ((part[boolean, 2] != 0) | ~np.isin(part[boolean, 3], (0, 1))).any():
                raise ValueError("invalid decision-memory boolean token")
            cursor += count
        if cursor != len(details):
            raise ValueError("unowned decision-memory reference details")


def validate_memory_vocabulary(obs: "ObservationBuffers", tables: "Tables") -> None:
    """Validate only owned values, including for a control without learned heads."""
    from manabot.policy.prepare import record_word
    from managym.decision_memory import decision_name_values, definition_programs

    if _FIELDS[0] not in obs:
        return
    for batch in range(obs[_FIELDS[0]].shape[0]):
        count, detail_count, reference_count, _ = memory_lengths(obs, batch)
        references = obs[_FIELDS[2]][batch, :reference_count]
        for definition in references[references[:, 13] == 1, 1].astype(np.int64):
            programs = definition_programs()
            if (
                not 0 < definition <= len(programs)
                or not programs[int(definition) - 1].eligible
            ):
                raise ValueError("ineligible decision-memory definition shortcut")
        headers = obs[_FIELDS[0]][batch, :count]
        details = obs[_FIELDS[1]][batch, :detail_count]
        if not np.isin(record_word(headers[:, 12:14], 0), decision_name_values()).all():
            raise ValueError("unadmitted decision-memory kind")
        chosen = headers[:, 60] == 1
        if not np.isin(
            record_word(headers[chosen, 61:63], 0), decision_name_values()
        ).all():
            raise ValueError("unadmitted decision-memory choice")
        program = details[:, 0] == 0
        tables.tokens(
            details[program, 1].astype(np.int64), record_word(details[program, 2:4], 0)
        )
        named = record_word(headers[:, 70:72], 0)
        if (named < 0).any() or (named >= len(tables.name_definitions)).any():
            raise ValueError("invalid decision-memory name row")


def _scaled(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (np.sign(values) * np.log1p(np.abs(values)) / 10).astype(np.float32)


def prepare_memory(
    obs: "ObservationBuffers", tables: "Tables"
) -> MemoryRows[np.ndarray] | None:
    from manabot.policy.prepare import NUMBER_TOKEN_KINDS, pack, record_word

    if _FIELDS[0] not in obs:
        return None
    headers = obs[_FIELDS[0]]
    lengths = headers[..., 19].sum(1).astype(np.int64)
    batches, slots = np.nonzero(headers[..., 19])
    selected = headers[batches, slots]
    temporal = np.stack(
        [record_word(selected[:, i : i + 2], 0) for i in (0, 2, 4, 6, 8)], -1
    )
    context = np.stack(
        [record_word(selected[:, i : i + 2], 0) for i in range(20, 60, 2)], -1
    )
    context[:, :2] = context[:, :2].astype(np.uint32).view(np.int32)
    selectors = np.zeros((len(selected), 10), np.float32)
    for index, offset in enumerate((60, 63, 66, 69)):
        selectors[:, index * 2] = selected[:, offset]
        if offset in (63, 66):
            selectors[:, index * 2 + 1] = _scaled(
                record_word(selected[:, offset + 1 : offset + 3], 0)
            )
    selectors[:, 8:] = selected[:, 72:74]
    numbers = np.concatenate(
        (
            _scaled(temporal),
            selected[:, 10:12] / [4, 12],
            (lengths[batches] - 1 - slots)[:, None] / 128,
            selected[:, 18:19] / 2,
            _scaled(context),
            selectors,
        ),
        -1,
    ).astype(np.float32)
    kinds = tables.names(1, record_word(selected[:, 12:14], 0))
    choices = np.zeros(len(selected), np.int64)
    present = selected[:, 60] == 1
    choices[present] = tables.names(1, record_word(selected[present, 61:63], 0))
    choice_names = record_word(selected[:, 70:72], 0)
    references_list: list[np.ndarray] = []
    details_list: list[np.ndarray] = []
    detail_routes: list[np.ndarray] = []
    member_routes: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    reference_base = record_base = 0
    for batch, length in enumerate(lengths):
        _, detail_count, reference_count, member_count = memory_lengths(obs, batch)
        references = obs[_FIELDS[2]][batch, :reference_count]
        references_list.append(references)
        from managym.decision_memory import definition_programs

        raw_details = obs[_FIELDS[1]][batch, :detail_count]
        for index, reference in enumerate(references):
            start, count = map(int, reference[10:12])
            part = raw_details[start : start + count]
            if reference[13]:
                program = definition_programs()[int(reference[1]) - 1].tokens
                expanded = np.array(
                    [
                        [0, kind, *integer_words(value), ordinal, 1]
                        for ordinal, (kind, value) in enumerate(program)
                    ],
                    np.float32,
                ).reshape(-1, 6)
                part = np.concatenate((expanded, part))
            details_list.append(part)
            detail_routes.append(np.full(len(part), reference_base + index, np.int64))
        members = obs[_FIELDS[3]][batch, :member_count]
        member_routes.append(members[:, 0].astype(np.int64) - 1 + reference_base)
        counts = headers[batch, :length, 15:18].astype(np.int64).ravel()
        groups.append(
            np.repeat(np.arange(length * 3, dtype=np.int64) + record_base * 3, counts)
        )
        positions.append((members[:, 1:2] / 128).astype(np.float32))
        reference_base += reference_count
        record_base += int(length)
    references = (
        np.concatenate(references_list)
        if references_list
        else np.zeros((0, 14), np.float32)
    )
    details = (
        np.concatenate(details_list) if details_list else np.zeros((0, 6), np.float32)
    )
    numeric = (
        np.stack([record_word(references[:, i : i + 2], 0) for i in (2, 4)], -1)
        .astype(np.uint32)
        .view(np.int32)
    )
    reference_numbers = np.concatenate(
        (
            references[:, :1] / 2,
            _scaled(numeric),
            references[:, 6:10],
            (references[:, 1:2] > 0).astype(np.float32),
        ),
        -1,
    ).astype(np.float32)
    payloads = record_word(details[:, 2:4], 0)
    token_kinds = details[:, 1].astype(np.int64)
    values = np.zeros(len(details), np.int64)
    program = details[:, 0] == 0
    values[program] = tables.tokens(token_kinds[program], payloads[program])
    signed = payloads.astype(np.uint32).view(np.int32)
    numeric_values = np.where(
        (token_kinds == -1) | np.isin(token_kinds, NUMBER_TOKEN_KINDS),
        _scaled(np.where(program, signed, payloads)),
        0,
    )
    # Field labels have one owner in the native vocabulary. No table indexes
    # or appearance/provenance identifiers enter these feature coordinates.
    # Native stable hashes of DecisionMemory::Program and ::Mana. The lookup
    # is against the supplied checkpoint vocabulary, not a current-world join.
    field_values = np.array([30649973, 1006277364], np.int64)
    fields = tables.names(1, field_values)[details[:, 0].astype(np.int64)]

    def cat(arrays: list[np.ndarray], width: int | None = None) -> np.ndarray:
        return (
            np.concatenate(arrays)
            if arrays
            else np.zeros((0,) if width is None else (0, width), np.int64)
        )

    return MemoryRows(
        numbers,
        kinds,
        choices,
        choice_names,
        reference_numbers,
        tables.rows(references[:, 1].astype(np.int64)),
        fields,
        values,
        np.stack(
            (
                token_kinds / 9,
                np.where(program, details[:, 4] / 128, 0),
                np.where(program, 0, np.floor_divide(details[:, 4], 6) / 128),
                np.where(program, 0, np.remainder(details[:, 4], 6) / 5),
                numeric_values,
            ),
            -1,
        ).astype(np.float32),
        cat(detail_routes),
        cat(member_routes),
        cat(groups),
        cat(positions, 1),
        batches.astype(np.int64),
        slots.astype(np.int64),
        np.flatnonzero(lengths).astype(np.int64),
        pack(lengths[lengths > 0]),
        tuple(map(int, lengths)),
        max(1, int(lengths.max(initial=0))),
    )
