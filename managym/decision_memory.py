"""Read-only viewer decision records, reconstructed from native projections.

These values carry historical snapshots. They confer no execution authority and
must never be refreshed by looking up an appearance in the current game.
"""

from dataclasses import dataclass
from functools import cache
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import managym

UnsignedWord = Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
SignedWord = Annotated[int, Field(ge=-(1 << 31), le=(1 << 31) - 1)]
BoardCounts = tuple[
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
    UnsignedWord,
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DecisionReference(_Record):
    role: Annotated[int, Field(ge=0, le=2)]
    definition: UnsignedWord | None
    appearance: UnsignedWord | None
    name: str | None
    description: str
    power: SignedWord
    toughness: SignedWord
    tapped: bool
    ready: bool
    attacking: bool
    blocking: bool
    program: list[tuple[Annotated[int, Field(ge=1, le=9)], SignedWord]]
    mana: list[
        tuple[
            UnsignedWord,
            UnsignedWord,
            UnsignedWord,
            UnsignedWord,
            UnsignedWord,
            UnsignedWord,
        ]
    ]


class DecisionContext(_Record):
    turn: UnsignedWord
    phase: Annotated[int, Field(ge=0, le=4)]
    step: Annotated[int, Field(ge=0, le=11)]
    life: tuple[SignedWord, SignedWord]
    hand: tuple[UnsignedWord, UnsignedWord]
    board_counts: tuple[BoardCounts, BoardCounts]
    mana_sources: list[DecisionReference]


class PublicChoice(_Record):
    kind: str | None
    mode: UnsignedWord | None
    ability: UnsignedWord | None
    named: str | None
    name_row: UnsignedWord | None
    declared: bool | None
    references: list[DecisionReference]


class DecisionRecord(_Record):
    actor: Annotated[int, Field(ge=1, le=2)]
    ordinal: UnsignedWord
    first_position: UnsignedWord
    last_position: UnsignedWord
    count: UnsignedWord
    kind: str
    context: DecisionContext
    response: list[DecisionReference]
    choice: PublicChoice


_LOG = TypeAdapter(list[DecisionRecord])


def read_records(source: str) -> list[DecisionRecord]:
    """Validate serialized native or retained viewer records once at the boundary."""
    return _LOG.validate_json(source)


@dataclass(frozen=True)
class DefinitionProgram:
    description: str
    tokens: tuple[tuple[int, int], ...]
    eligible: bool


@cache
def definition_programs() -> tuple[DefinitionProgram, ...]:
    """Immutable registered faces, in native definition order; no live objects."""

    catalog = managym.catalog_tokens()
    definitions = json.loads(managym.card_definitions_json())
    result: list[DefinitionProgram] = []
    for row, definition in enumerate(definitions):
        if not isinstance(definition, dict) or not isinstance(
            definition.get("is_token"), bool
        ):
            raise ValueError("invalid native definition characteristics")
        # Native historical snapshots serialize an identity-free face, with
        # token/back-face metadata absent. Exact-description equality cannot
        # hold for a registered row that carries either kind of metadata.
        eligible = not definition["is_token"] and definition.get("back_face") is None
        start, stop = catalog["definition_offsets"][row : row + 2]
        spans = [(start, stop)]
        first, last = catalog["definition_program_offsets"][row : row + 2]
        for program in catalog["definition_program_rows"][first:last]:
            start, stop = catalog["program_offsets"][program : program + 2]
            spans.append((start, stop))
        tokens = tuple(
            (catalog["token_kind"][i], catalog["token_value"][i])
            for start, stop in spans
            for i in range(start, stop)
        )
        result.append(
            DefinitionProgram(
                json.dumps(definition, ensure_ascii=False, separators=(",", ":")),
                tokens,
                eligible,
            )
        )
    return tuple(result)


@cache
def decision_name_values() -> tuple[int, ...]:
    """Native memory labels for host admission, without optional model rows."""

    return tuple(
        value
        for _, name, value, _, _ in managym.record_vocabulary()
        if name.startswith("DecisionMemory::")
    )
