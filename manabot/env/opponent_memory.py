"""Independent Python tensor projection of whole-game viewer decision records."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

import managym
from managym.opponent_memory import OpponentDecisionRecord

RECORD_DIM = 18
DETAIL_DIM = 8


def _words(value: int) -> tuple[int, int]:
    return (value & 0xFFFFFFFF) >> 16, value & 0xFFFF


def encode_records(
    records: Sequence[OpponentDecisionRecord],
    record_capacity: int,
    detail_capacity: int,
) -> dict[str, NDArray[np.float32]]:
    """Project every record or raise Capacity; no suffix or detail truncation."""
    headers: list[list[float]] = []
    details: list[list[float]] = []

    def detail(
        group: int, reference: int, field: str, kind: int, value: int, ordinal: int = 0
    ) -> None:
        details.append(
            [
                group,
                reference,
                *_words(managym.catalog_token_value(f"OpponentMemory::{field}")),
                kind,
                *_words(value),
                ordinal,
            ]
        )

    for record in records:
        start = len(details)
        for seat in range(2):
            detail(0, seat, "Life", -1, record.context.life[seat])
            detail(0, seat, "Hand", -1, record.context.hand[seat])
        for seat in range(2):
            for kind, count in enumerate(record.context.board_counts[seat]):
                detail(0, seat, "BoardCounts", -1, count, kind)
        for group, references in (
            (1, record.context.mana_sources),
            (2, record.response),
            (3, record.choice.references),
        ):
            for index, reference in enumerate(references):
                for field, value in (
                    ("Role", reference.role),
                    ("Power", reference.power),
                    ("Toughness", reference.toughness),
                    ("Tapped", int(reference.tapped)),
                    ("Ready", int(reference.ready)),
                    ("Attacking", int(reference.attacking)),
                    ("Blocking", int(reference.blocking)),
                ):
                    detail(group, index, field, -1, value)
                if reference.definition is not None:
                    detail(group, index, "Definition", -3, reference.definition + 1)
                for ordinal, (kind, value) in enumerate(reference.program):
                    detail(group, index, "Program", kind, value, ordinal)
                for ordinal, production in enumerate(reference.mana):
                    for color, amount in enumerate(production):
                        detail(group, index, "Mana", -1, amount, ordinal * 6 + color)
        if record.choice.kind is not None:
            detail(
                4,
                0,
                "Choice",
                -2,
                managym.catalog_token_value(f"OpponentMemory::{record.choice.kind}"),
            )
        if record.choice.mode is not None:
            detail(4, 0, "Mode", -1, record.choice.mode)
        if record.choice.ability is not None:
            detail(4, 0, "Ability", -1, record.choice.ability)
        if record.choice.name_row is not None:
            detail(4, 0, "Named", -4, record.choice.name_row + 1)
        if record.choice.declared is not None:
            detail(4, 0, "Declared", -1, int(record.choice.declared))
        headers.append(
            [
                *_words(record.ordinal),
                *_words(record.count),
                *_words(record.context.turn),
                *_words(record.first_position),
                *_words(record.last_position),
                record.context.phase,
                record.context.step,
                *_words(managym.catalog_token_value(f"OpponentMemory::{record.kind}")),
                start,
                len(details) - start,
                2,
                1,
            ]
        )
    output: dict[str, NDArray[np.float32]] = {}
    for key, rows, capacity, width in (
        ("opponent_records", headers, record_capacity, RECORD_DIM),
        ("opponent_details", details, detail_capacity, DETAIL_DIM),
    ):
        if len(rows) > capacity:
            raise managym.AgentError(
                f"Capacity: {key} requires {len(rows)} rows, capacity {capacity}"
            )
        array = np.zeros((capacity, width), np.float32)
        if rows:
            array[: len(rows)] = rows
        output[key] = array
    return output


def blank_memory(
    observation: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Remove every history/count/presence cue before host preparation."""
    return {
        key: np.zeros_like(value)
        if key in ("opponent_records", "opponent_details")
        else value
        for key, value in observation.items()
    }
