"""Typed optional game counters; native capture owns all counting formulas.

Live summaries are partial until terminal. Retained capture reduction validates
copy continuity, but says nothing about experiment eligibility or replay success.
"""

from pydantic import BaseModel, ConfigDict, TypeAdapter

import managym
from managym.card_capture import CapturedCards


class CardCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seat: int
    definition: int
    name: str
    deck_copies: int
    opening_copies: int
    drawn_copies: int
    opportunities: int | None
    selections: int | None
    resolutions: int | None


class GameStatistics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int
    generation: int
    stream: int
    terminal: bool
    winner: int | None
    cards: tuple[CardCounts, ...]


def summary(raw: str) -> GameStatistics | None:
    return TypeAdapter(GameStatistics | None).validate_json(raw)


def summaries(raw: str) -> list[GameStatistics | None]:
    return TypeAdapter(list[GameStatistics | None]).validate_json(raw)


def reduce_cards(cards: CapturedCards) -> list[CardCounts]:
    return TypeAdapter(list[CardCounts]).validate_json(
        managym._managym.reduce_card_capture_json(
            cards.capture.model_dump_json(by_alias=True)
        )
    )
