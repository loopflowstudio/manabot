"""Typed private native card facts shared by training and evaluation.

Copies and incarnations preserve native identity. Optional exposure records name
consumed nontrivial choices and completed effect frames; historical captures
without them say nothing about opportunities or resolutions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import managym


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


type Zone = Literal[
    "Library", "Hand", "Battlefield", "Graveyard", "Stack", "Exile", "Command"
]

type Seat = Literal[0, 1]


class CardSnapshot(StrictModel):
    copy_id: int = Field(alias="copy", ge=0)
    definition: int = Field(ge=0)
    name: str
    incarnation: int = Field(ge=0)
    owner: Seat
    zone: Zone | None
    origin: Literal["deck", "outside", "generated"]


class CardMove(StrictModel):
    sequence: int = Field(ge=0)
    card: CardSnapshot
    from_: Zone | None = Field(alias="from")
    controller: Seat
    cause: Literal["draw", "selection", "return", "outside", "generated", "move"]
    turn: int = Field(ge=1)


class ExposureOption(StrictModel):
    action_type: str
    objects: tuple[CardSnapshot, ...]


class ExposureChoice(StrictModel):
    revision: int = Field(ge=0)
    actor: Seat
    kind: str
    options: tuple[ExposureOption, ...]
    selected: int = Field(ge=0)

    @model_validator(mode="after")
    def legal(self) -> ExposureChoice:
        if len(self.options) < 2 or self.selected >= len(self.options):
            raise ValueError("exposure requires a nontrivial legal choice")
        return self


class ExposureObject(StrictModel):
    entity: int = Field(ge=0)
    incarnation: int = Field(ge=0)


class ExposureResolution(StrictModel):
    revision: int = Field(ge=0)
    controller: Seat
    source: ExposureObject | None
    definition: int | None


class ManaUse(StrictModel):
    priority_produced: int = Field(ge=0)
    regular_expired: int = Field(ge=0)
    combat_expired: int = Field(ge=0)
    priority_expired_min: int = Field(ge=0)
    priority_expired_max: int = Field(ge=0)


class TurnMana(StrictModel):
    available: int = Field(ge=0)
    paid: int = Field(ge=0)


class ExposureCapture(StrictModel):
    mana_turns: tuple[tuple[TurnMana, ...], tuple[TurnMana, ...]] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    attackers_lost_for_nothing: tuple[int, int] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    blockers_lost_for_nothing: tuple[int, int] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    mana: tuple[ManaUse, ManaUse] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    choices: tuple[ExposureChoice, ...]
    resolutions: tuple[ExposureResolution, ...]


class NativeCapture(StrictModel):
    version: Literal[1]
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial: tuple[CardSnapshot, ...]
    opening: tuple[tuple[int, ...], tuple[int, ...]]
    moves: tuple[CardMove, ...]
    exposure: ExposureCapture | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class CapturedCards(StrictModel):
    capture: NativeCapture
    final_cards: tuple[CardSnapshot, ...]
    turn: int = Field(ge=1)

    @model_validator(mode="after")
    def complete(self) -> CapturedCards:
        cards = {card.copy_id: card for card in self.capture.initial}
        if len(cards) != len(self.capture.initial):
            raise ValueError("duplicate initial copy")
        for seat in (0, 1):
            opening = self.capture.opening[seat]
            expected = {
                c.copy_id
                for c in cards.values()
                if c.owner == seat and c.zone == "Hand"
            }
            if len(opening) != len(set(opening)) or set(opening) != expected:
                raise ValueError("opening hand differs from setup")
        turn = 1
        for sequence, move in enumerate(self.capture.moves):
            card = move.card
            old = cards.get(card.copy_id)
            if move.sequence != sequence or not turn <= move.turn <= self.turn:
                raise ValueError("card facts out of sequence")
            turn = move.turn
            if old is None:
                if card.origin != "generated" or move.from_ is not None:
                    raise ValueError("unrecorded physical copy")
            elif (
                old.zone != move.from_
                or old.definition != card.definition
                or old.owner != card.owner
                or old.origin != card.origin
                or old.name != card.name
                or card.incarnation
                != old.incarnation + int(old.zone is not None and card.zone is not None)
            ):
                raise ValueError("card movement does not continue its physical copy")
            if move.cause == "draw" and (move.from_, card.zone) != ("Library", "Hand"):
                raise ValueError("actual draw must move library to hand")
            if card.zone == "Hand" and move.cause == "move":
                raise ValueError("hand acquisition needs a cause")
            cards[card.copy_id] = card
        if cards != {card.copy_id: card for card in self.final_cards} or len(
            cards
        ) != len(self.final_cards):
            raise ValueError("capture omitted a committed movement")
        return self


def captured_cards(env: managym.Env) -> CapturedCards:
    """Validate private native facts for live reducers and recorded-game replay."""
    return CapturedCards.model_validate_json(env.card_capture_json())
