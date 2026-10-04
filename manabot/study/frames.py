"""
frames.py
Recorded games as pandas DataFrames, for notebooks and ad hoc questions

One row per game, per seat, or per decision. Use these to explore; once a
question is worth keeping, write it as a measure so the report can show it.
"""

from __future__ import annotations

# Standard library
from typing import Iterable, Mapping

# Third-party imports
import pandas as pd

# Local imports
from .measures import Rate, position
from .record import GameRecord


def games_frame(games: Iterable[GameRecord]) -> pd.DataFrame:
    """One row per game."""
    return pd.DataFrame(
        {
            "game_id": game.game_id,
            "seed": game.seed,
            "player_0": game.labels[0],
            "player_1": game.labels[1],
            "deck_0": game.decks[0],
            "deck_1": game.decks[1],
            "first_player": game.first_player,
            "winner": game.winner,
            "end": game.end,
            "turns": game.turns,
            "decisions": len(game.decisions),
            "seconds": game.seconds,
        }
        for game in games
    )


def seats_frame(games: Iterable[GameRecord]) -> pd.DataFrame:
    """One row per game and seat: who sat there, with which deck, and the result."""
    return pd.DataFrame(
        {
            "game_id": game.game_id,
            "seed": game.seed,
            "seat": seat,
            "player": game.labels[seat],
            "opponent": game.labels[1 - seat],
            "deck": game.decks[seat],
            "position": position(game, seat),
            "finished": game.completed,
            "score": None
            if not game.completed
            else 0.5
            if game.winner is None
            else float(game.winner == seat),
            "turns": game.turns,
        }
        for game in games
        for seat in (0, 1)
    )


def decisions_frame(games: Iterable[GameRecord]) -> pd.DataFrame:
    """One row per decision, with the board summarized as counts."""
    rows = []
    for game in games:
        for index, decision in enumerate(game.decisions):
            rows.append(
                {
                    "game_id": game.game_id,
                    "index": index,
                    "player": game.labels[decision.actor],
                    "deck": game.decks[decision.actor],
                    "seat": decision.actor,
                    "turn": decision.turn,
                    "own_turn": decision.own_turn,
                    "phase": decision.phase,
                    "step": decision.step,
                    "kind": decision.kind,
                    "chosen": decision.chosen,
                    "label": decision.label,
                    "declared": decision.declared,
                    "subject": decision.subject,
                    "offers": len(decision.offer_labels),
                    "life": decision.life[0],
                    "opponent_life": decision.life[1],
                    "hand": decision.hand,
                    "lands": sum(p.is_land for p in decision.mine),
                    "untapped_lands": sum(
                        p.is_land and not p.tapped for p in decision.mine
                    ),
                    "creatures": sum(p.is_creature for p in decision.mine),
                    "opponent_creatures": sum(p.is_creature for p in decision.theirs),
                    "opponent_untapped_creatures": sum(
                        p.is_creature and not p.tapped for p in decision.theirs
                    ),
                    "latency": decision.latency,
                }
            )
    return pd.DataFrame(rows)


def rates_frame(rates: Mapping[object, Rate], name: str = "group") -> pd.DataFrame:
    """Rates as a table: value, count and 95% interval per group."""
    rows = []
    for key, rate in rates.items():
        low, high = rate.interval() or (None, None)
        rows.append(
            {
                name: " / ".join(map(str, key)) if isinstance(key, tuple) else key,
                "rate": rate.value,
                "hits": rate.hits,
                "total": rate.total,
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)
