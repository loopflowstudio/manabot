"""Read-only viewer decision records, reconstructed from native projections.

These values carry historical snapshots. They confer no execution authority and
must never be refreshed by looking up an appearance in the current game.
"""

from pydantic import BaseModel, ConfigDict, TypeAdapter


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DecisionReference(_Record):
    role: int
    definition: int | None
    appearance: int | None
    name: str | None
    description: str
    power: int
    toughness: int
    tapped: bool
    ready: bool
    attacking: bool
    blocking: bool
    program: list[tuple[int, int]]
    mana: list[tuple[int, int, int, int, int, int]]


class DecisionContext(_Record):
    turn: int
    phase: int
    step: int
    life: tuple[int, int]
    hand: tuple[int, int]
    board_counts: tuple[tuple[int, ...], tuple[int, ...]]
    mana_sources: list[DecisionReference]


class PublicChoice(_Record):
    kind: str | None
    mode: int | None
    ability: int | None
    named: str | None
    name_row: int | None
    declared: bool | None
    references: list[DecisionReference]


class OpponentDecisionRecord(_Record):
    ordinal: int
    first_position: int
    last_position: int
    count: int
    kind: str
    context: DecisionContext
    response: list[DecisionReference]
    choice: PublicChoice


_LOG = TypeAdapter(list[OpponentDecisionRecord])


def read_records(source: str) -> list[OpponentDecisionRecord]:
    """Validate serialized native or retained viewer records once at the boundary."""
    return _LOG.validate_json(source)
