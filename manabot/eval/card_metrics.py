"""Copy-weighted card counts per game, and their normalized table rows.

`reduce_cards` owns the counting rules. `normalize` splits its rows into the
three author tables an evaluation delivers: `card_plays` (one row per game and
seat, with the game's facts), `card_counts` (per card in that game) and
`cards` (names for card keys). `CARD_ROLLUPS` are the per-deal totals of
those tables that card queries read: Etude sums them as each match is stored
(or once per block for runs stored without them), so a query never re-reads
card rows."""

from __future__ import annotations

from collections import Counter

from etude.eval.records import Rollup

from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import GameEvidence, Seat

METRIC_VERSION = "gih-copies-v1"


class CardGame(StrictModel):
    seat: Seat
    game_seat: Seat
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
    eligible = game.outcome == "win_loss" and game.replay.get("passed") is True
    for game_seat in (0, 1):
        seat = participants.index(game.seat_players[game_seat])
        deck = Counter(
            c.definition
            for c in capture.initial
            if c.owner == game_seat and c.origin == "deck"
        )
        opening = Counter(cards[c].definition for c in capture.opening[game_seat])
        drawn = Counter(
            m.card.definition
            for m in capture.moves
            if m.card.owner == game_seat and m.cause == "draw"
        )
        definitions = {c.definition: c for c in cards.values() if c.owner == game_seat}
        for definition in sorted(deck.keys() | opening.keys() | drawn.keys()):
            rows.append(
                CardGame(
                    seat=seat,
                    game_seat=game_seat,
                    game=game.game,
                    deal=game.deal,
                    card=f"{capture.content_digest}:{definition}",
                    name=definitions[definition].name,
                    definition=definition,
                    content=capture.content_digest,
                    deck=game.seat_decks[game_seat],
                    opponent_deck=game.seat_decks[1 - game_seat],
                    on_play=game_seat == 0,
                    deck_copies=deck[definition],
                    opening_copies=opening[definition],
                    drawn_copies=drawn[definition],
                    gih_copies=opening[definition] + drawn[definition],
                    eligible=eligible,
                    outcome=game.outcome,
                    win=int(game.winner == game_seat) if eligible else None,
                )
            )
    return rows


PLAY_FIELDS = (
    "seat",
    "game_seat",
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
    "seat",
    "game",
    "card",
    "deck_copies",
    "opening_copies",
    "drawn_copies",
    "gih_copies",
)
# Etude rows name a seat, so a card's name row is keyed by whose deck held it.
CARD_FIELDS = ("seat", "card", "content", "definition", "name")


class CardRows(StrictModel):
    plays: list[dict[str, int | float | str | bool | None]]
    counts: list[dict[str, int | float | str | bool | None]]
    cards: list[dict[str, int | float | str | bool | None]]


def normalize(rows: list[CardGame]) -> CardRows:
    """One `card_plays` row per (game, seat); counts and names keyed by card."""
    plays: dict[tuple[str, int], dict[str, int | float | str | bool | None]] = {}
    cards: dict[tuple[str, int], dict[str, int | float | str | bool | None]] = {}
    counts = []
    for row in rows:
        data = row.model_dump()
        plays.setdefault((row.game, row.seat), {f: data[f] for f in PLAY_FIELDS})
        cards.setdefault((row.card, row.seat), {f: data[f] for f in CARD_FIELDS})
        counts.append({f: data[f] for f in COUNT_FIELDS})
    return CardRows(
        plays=list(plays.values()), counts=counts, cards=list(cards.values())
    )


# ---- Rollups ------------------------------------------------------------------

# Every filter a card query offers, plus the deal (the bootstrap unit) and the
# Etude-reserved `player`/`opponents` keys of the row's seat and the other seat.
_SPLIT = ("deal", "on_play", "deck", "opponent_deck", "source_commit")
_IN_DECK = "deck_copies > 0"
_NOT_SEEN = "deck_copies > 0 and gih_copies = 0"

# Per card: copy-weighted opportunities and winning copies of each column. Only
# eligible games (a winner, a passed replay) count, so `win` is never null here.
CARD_DEALS = Rollup(
    name="card_deals",
    table="card_counts",
    join=("card_plays",),
    by=("card", *_SPLIT, "player", "opponents"),
    sums={
        "gp": _IN_DECK,
        "gp_won": f"({_IN_DECK})::INTEGER * win",
        "oh": "opening_copies",
        "oh_won": "opening_copies * win",
        "gd": "drawn_copies",
        "gd_won": "drawn_copies * win",
        "gih": "gih_copies",
        "gih_won": "gih_copies * win",
        "gns": _NOT_SEEN,
        "gns_won": f"({_NOT_SEEN})::INTEGER * win",
    },
    where="eligible",
)
# Per player-game, eligible or not: its `rows` count games, `wins` their wins,
# and ineligible games are reported by outcome.
GAME_DEALS = Rollup(
    name="game_deals",
    table="card_plays",
    by=(*_SPLIT, "eligible", "outcome", "player", "opponents"),
    sums={"wins": "win"},
)
CARD_ROLLUPS = (CARD_DEALS, GAME_DEALS)
