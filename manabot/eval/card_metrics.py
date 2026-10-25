"""Copy-weighted card counts per game, and their normalized table rows.

`reduce_cards` owns the counting rules. `normalize` splits its rows into the
three author tables an evaluation delivers: `card_plays` (one row per game and
player, with the game's facts), `card_counts` (per card in that game) and
`cards` (names for card keys)."""

from __future__ import annotations

from collections import Counter

from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import GameEvidence, Seat

METRIC_VERSION = "gih-copies-v1"


class CardGame(StrictModel):
    player: Seat
    game: str
    deal: int
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


PLAY_FIELDS = (
    "player",
    "game",
    "deal",
    "deck",
    "opponent_deck",
    "on_play",
    "eligible",
    "outcome",
    "win",
    "content",
    "metric_version",
)
COUNT_FIELDS = (
    "player",
    "game",
    "card",
    "deck_copies",
    "opening_copies",
    "drawn_copies",
    "gih_copies",
)
# Etude rows name a player, so a card's name row is keyed by whose deck held it.
CARD_FIELDS = ("player", "card", "content", "definition", "name")


class CardRows(StrictModel):
    plays: list[dict[str, int | float | str | bool | None]]
    counts: list[dict[str, int | float | str | bool | None]]
    cards: list[dict[str, int | float | str | bool | None]]


def normalize(rows: list[CardGame]) -> CardRows:
    """One `card_plays` row per (game, player); counts and names keyed by card."""
    plays: dict[tuple[str, int], dict[str, int | float | str | bool | None]] = {}
    cards: dict[tuple[str, int], dict[str, int | float | str | bool | None]] = {}
    counts = []
    for row in rows:
        data = row.model_dump()
        plays.setdefault((row.game, row.player), {f: data[f] for f in PLAY_FIELDS})
        cards.setdefault((row.card, row.player), {f: data[f] for f in CARD_FIELDS})
        counts.append({f: data[f] for f in COUNT_FIELDS})
    return CardRows(
        plays=list(plays.values()), counts=counts, cards=list(cards.values())
    )
