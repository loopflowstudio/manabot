"""Completed self-play games, counted by deck matchup and starting seat.

Each game contributes two seat outcomes, but one play and one draw observation.
Counts are drained once per successful learner update and logged with that
update's Etude metrics; missing rows are not zero-game observations. No gradients or
policy sampling depend on this accounting.
"""

from dataclasses import dataclass, field
from typing import Literal, TypedDict


class SelfPlayOutcome(TypedDict):
    deck: str
    opponent_deck: str
    position: Literal["play", "draw"]
    wins: int
    losses: int
    draws: int


@dataclass
class OutcomeCounts:
    wins: int = 0
    losses: int = 0
    draws: int = 0


@dataclass
class SelfPlayCounter:
    """Pending counts survive collector snapshots and reset only when exported."""

    counts: dict[tuple[str, str, Literal["play", "draw"]], OutcomeCounts] = field(
        default_factory=dict
    )

    def record(self, decks: tuple[str, str], winner: int | None) -> None:
        if winner not in (None, 0, 1):
            raise ValueError("winner must be a seat or a drawn game")
        for seat, position in ((0, "play"), (1, "draw")):
            key = (decks[seat], decks[1 - seat], position)
            counts = self.counts.setdefault(key, OutcomeCounts())
            if winner is None:
                counts.draws += 1
            elif winner == seat:
                counts.wins += 1
            else:
                counts.losses += 1

    def drain(self) -> list[SelfPlayOutcome]:
        rows: list[SelfPlayOutcome] = [
            SelfPlayOutcome(
                deck=deck,
                opponent_deck=opponent,
                position=position,
                wins=c.wins,
                losses=c.losses,
                draws=c.draws,
            )
            for (deck, opponent, position), c in sorted(self.counts.items())
        ]
        self.counts.clear()
        return rows
