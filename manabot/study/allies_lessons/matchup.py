"""
matchup.py
The Allies versus Lessons setup, seatings and matchup-specific measures

A pairing is scheduled in every seating of decks and
players so that deck and play/draw effects are not confused with the player.
"""

from __future__ import annotations

# Standard library
from collections import Counter
from typing import Iterable

# First-party imports
from manabot.env import Match
from manabot.infra.hypers import MatchHypers

# Local imports
from ..measures import choices
from ..record import GameRecord, GameSpec

PACK = "ur-lessons-vs-gw-allies"
DECKS = {"ur_lessons": "UR Lessons", "gw_allies": "GW Allies"}

PLAYERS = {
    "search": {
        "kind": "demo_search",
        "sims": 64,
        "rollouts_per_world": 4,
        "max_steps": 2000,
    },
    "random": {"kind": "random"},
    "greedy": {"kind": "scripted_greedy"},
}

LEARN = {
    "LEARN_TAKE_LESSON": "Took a Lesson from the sideboard",
    "LEARN_DISCARD": "Discarded, then drew",
    "DECLINE_CHOICE": "Declined",
}


def setup(first_deck: str) -> tuple[MatchHypers, tuple[str, str]]:
    """The authored match with `first_deck` in seat 0."""
    match = MatchHypers.authored(
        PACK, "ur_lessons", "gw_allies", hero="seat-0", villain="seat-1"
    )
    if first_deck == "ur_lessons":
        return match, ("ur_lessons", "gw_allies")
    return Match(match).swapped().hypers, ("gw_allies", "ur_lessons")


def schedule(
    first: str, second: str, *, deals: int, seed: int, max_commands: int = 10_000
) -> list[GameSpec]:
    """Every seating of two players over `deals` deal seeds.

    A mirror (`first == second`) plays two games per deal, one per deck order.
    Two different players play four: each deck order with each player first.
    """
    orders = (
        [(first, second)] if first == second else [(first, second), (second, first)]
    )
    specs = []
    for deal in range(deals):
        for first_deck in DECKS:
            match, decks = setup(first_deck)
            for labels in orders:
                specs.append(
                    GameSpec(
                        game_id=f"{labels[0]}-{labels[1]}-{first_deck}-{seed + deal}",
                        match=match.model_dump(),
                        players=(PLAYERS[labels[0]], PLAYERS[labels[1]]),
                        labels=labels,
                        decks=decks,
                        seed=seed + deal,
                        max_commands=max_commands,
                    )
                )
    return specs


def learn_choices(games: Iterable[GameRecord], label: str) -> Counter:
    """What the player did each time it resolved Learn."""
    return choices(games, label, "LEARN")
