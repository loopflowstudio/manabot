"""Copy-weighted GIH reduction and evaluation exclusion accounting."""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import GameEvidence, Seat

METRIC_VERSION = "gih-copies-v1"


class MatchExclusions(StrictModel):
    crash: int = Field(default=0, ge=0)
    timeout: int = Field(default=0, ge=0)
    truncated: int = Field(default=0, ge=0)
    replay_failed: int = Field(default=0, ge=0)
    failed_match: int = Field(default=0, ge=0)
    unobserved: int = Field(default=0, ge=0)


class CardMatchError(ValueError):
    """Small typed exclusion summary retained in Etude's failed-match record."""

    def __init__(self, outcomes: dict[str, int]) -> None:
        self.outcomes = MatchExclusions.model_validate(outcomes)
        super().__init__(outcomes)

    def __str__(self) -> str:
        return self.outcomes.model_dump_json(exclude_defaults=True)


class CardGame(StrictModel):
    player: Seat
    game: str
    deal: int
    leg: int
    card: str
    name: str
    definition: int
    content: str
    deck: str
    opponent_deck: str
    on_play: bool
    deck_copies: int
    opening_copies: int
    drawn_copies: int
    gih_copies: int
    eligible: bool
    outcome: str
    win: int | None
    metric_version: str = METRIC_VERSION


def reduce_cards(game: GameEvidence, participants: tuple[str, str]) -> list[CardGame]:
    """Opening copies + actual draws, including redraws; never hand presence.

    The outcome is per game, but each qualifying physical copy contributes one
    observation. Tutors, returns and outside/generated acquisitions contribute
    zero at acquisition. A later actual library draw still qualifies.
    """
    if set(participants) != set(game.seat_players) or len(set(participants)) != 2:
        raise ValueError("participants differ from engine seating")
    capture = game.cards.capture
    cards = {c.copy_id: c for c in capture.initial}
    cards.update({m.card.copy_id: m.card for m in capture.moves})
    rows: list[CardGame] = []
    for seat in (0, 1):
        deck = Counter(
            c.definition
            for c in capture.initial
            if c.owner == seat and c.origin == "deck"
        )
        opening = Counter(cards[c].definition for c in capture.opening[seat])
        drawn = Counter(
            m.card.definition
            for m in capture.moves
            if m.card.owner == seat and m.cause == "draw"
        )
        definitions = {c.definition: c for c in cards.values() if c.owner == seat}
        eligible = game.outcome == "win_loss" and game.replay.get("passed") is True
        for definition in sorted(deck.keys() | opening.keys() | drawn.keys()):
            rows.append(
                CardGame(
                    player=participants.index(game.seat_players[seat]),
                    game=game.game,
                    deal=game.deal,
                    leg=game.leg,
                    card=f"{capture.content_digest}:{definition}",
                    name=definitions[definition].name,
                    definition=definition,
                    content=capture.content_digest,
                    deck=game.seat_decks[seat],
                    opponent_deck=game.seat_decks[1 - seat],
                    on_play=seat == 0,
                    deck_copies=deck[definition],
                    opening_copies=opening[definition],
                    drawn_copies=drawn[definition],
                    gih_copies=opening[definition] + drawn[definition],
                    eligible=eligible,
                    outcome=game.outcome,
                    win=int(game.winner == seat) if eligible else None,
                )
            )
    return rows
