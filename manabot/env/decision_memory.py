"""Independent projection of fixed record context and shared historical snapshots.

Native canonical JSON remains expanded. Exact equality includes provenance and
all entitled fields; local table joins are transport, never learning features.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

import managym
from managym.decision_memory import DecisionRecord, definition_programs

RECORD_DIM = 74
REFERENCE_DIM = 14
DETAIL_DIM = 6
MEMBER_DIM = 2
MEMORY_WIDTHS = {
    "decision_records": RECORD_DIM,
    "decision_references": REFERENCE_DIM,
    "decision_reference_details": DETAIL_DIM,
    "decision_members": MEMBER_DIM,
}

MEMORY_FIELDS = (
    "decision_records",
    "decision_reference_details",
    "decision_references",
    "decision_members",
)


def integer_words(value: int) -> tuple[int, int]:
    return (value & 0xFFFFFFFF) >> 16, value & 0xFFFF


def encode_records(
    records: Sequence[DecisionRecord],
    record_capacity: int,
    detail_capacity: int,
    reference_capacity: int,
    member_capacity: int,
) -> dict[str, NDArray[np.float32]]:
    """Project complete prefixes or raise Capacity, without truncating lists."""
    headers: list[list[float]] = []
    references: list[list[float]] = []
    details: list[list[float]] = []
    members: list[list[float]] = []
    indexes: dict[str, int] = {}
    for record in records:
        start = len(members)
        groups = (
            record.context.mana_sources,
            record.response,
            record.choice.references,
        )
        for group in groups:
            for ordinal, reference in enumerate(group):
                # Pydantic's canonical value includes ordered programs and mana,
                # unknownness, description and appearance, not today's objects.
                key = reference.model_dump_json()
                if key not in indexes:
                    indexes[key] = len(references) + 1
                    detail_start = len(details)
                    exact = False
                    if reference.definition is not None:
                        definition = definition_programs()[reference.definition]
                        exact = (
                            reference.description == definition.description
                            and tuple(reference.program) == definition.tokens
                        )
                    for index, (kind, value) in enumerate(
                        () if exact else reference.program
                    ):
                        details.append([0, kind, *integer_words(value), index, 1])
                    for index, production in enumerate(reference.mana):
                        for color, amount in enumerate(production):
                            details.append(
                                [1, -1, *integer_words(amount), index * 6 + color, 1]
                            )
                    references.append(
                        [
                            reference.role,
                            0
                            if reference.definition is None
                            else reference.definition + 1,
                            *integer_words(reference.power),
                            *integer_words(reference.toughness),
                            int(reference.tapped),
                            int(reference.ready),
                            int(reference.attacking),
                            int(reference.blocking),
                            detail_start,
                            len(details) - detail_start,
                            1,
                            int(exact),
                        ]
                    )
                members.append([indexes[key], ordinal])
        context = [
            *record.context.life,
            *record.context.hand,
            *record.context.board_counts[0],
            *record.context.board_counts[1],
        ]
        header: list[float] = [
            *integer_words(record.ordinal),
            *integer_words(record.count),
            *integer_words(record.context.turn),
            *integer_words(record.first_position),
            *integer_words(record.last_position),
            record.context.phase,
            record.context.step,
            *integer_words(
                managym.catalog_token_value(f"DecisionMemory::{record.kind}")
            ),
            start,
            *(len(group) for group in groups),
            record.actor,
            1,
            *(word for value in context for word in integer_words(value)),
        ]
        for value in (
            None
            if record.choice.kind is None
            else managym.catalog_token_value(f"DecisionMemory::{record.choice.kind}"),
            record.choice.mode,
            record.choice.ability,
            None if record.choice.name_row is None else record.choice.name_row + 1,
        ):
            header.extend(
                [int(value is not None), *integer_words(0 if value is None else value)]
            )
        header.extend(
            [int(record.choice.declared is not None), int(bool(record.choice.declared))]
        )
        headers.append(header)
    output: dict[str, NDArray[np.float32]] = {}
    for key, rows, capacity in (
        ("decision_records", headers, record_capacity),
        ("decision_references", references, reference_capacity),
        ("decision_reference_details", details, detail_capacity),
        ("decision_members", members, member_capacity),
    ):
        if len(rows) > capacity:
            raise managym.AgentError(
                f"Capacity: {key} requires {len(rows)} rows, capacity {capacity}"
            )
        array = np.zeros((capacity, MEMORY_WIDTHS[key]), np.float32)
        if rows:
            array[: len(rows)] = rows
        output[key] = array
    return output


def blank_memory(
    observation: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Erase every history, presence and table cue before host preparation."""
    return {
        key: np.zeros_like(value) if key in MEMORY_WIDTHS else value
        for key, value in observation.items()
    }
