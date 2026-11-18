"""Completed self-play games, counted by deck matchup and starting seat.

Each game contributes two seat outcomes, but one play and one draw observation.
Its settings (conditions that can differ between two games of one run, such as
decklist disclosure) are counted in the same call, so an update's setting
counts and outcomes always describe the same games. Counts are drained once per
successful learner update and logged with that update's immutable tables;
missing rows are not zero-game observations.
No gradients or policy sampling depend on this accounting.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypedDict


class SelfPlayOutcome(TypedDict):
    deck: str
    opponent_deck: str
    position: Literal["play", "draw"]
    wins: int
    losses: int
    draws: int


class GameSetting(NamedTuple):
    """One condition a game was played under.

    `seat` is None for a setting of the whole game and the native seat (0 is on
    the play) for a per-seat one. `value` is text, spelled as the native setup
    spells it; booleans are `true` and `false`.
    """

    seat: int | None
    setting: str
    value: str


class GameSettingCount(TypedDict):
    seat: int | None
    setting: str
    value: str
    games: int


class CompletedGames(NamedTuple):
    outcomes: list[SelfPlayOutcome]
    settings: list[GameSettingCount]


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
    settings: Counter[GameSetting] = field(default_factory=Counter)

    def record(
        self,
        decks: tuple[str, str],
        winner: int | None,
        settings: Iterable[GameSetting] = (),
    ) -> None:
        if winner not in (None, 0, 1):
            raise ValueError("winner must be a seat or a drawn game")
        self.settings.update(settings)
        for seat, position in ((0, "play"), (1, "draw")):
            key = (decks[seat], decks[1 - seat], position)
            counts = self.counts.setdefault(key, OutcomeCounts())
            if winner is None:
                counts.draws += 1
            elif winner == seat:
                counts.wins += 1
            else:
                counts.losses += 1

    def drain(self) -> CompletedGames:
        outcomes: list[SelfPlayOutcome] = [
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
        settings = [
            GameSettingCount(**key._asdict(), games=games)
            # Whole-game settings first: a None seat cannot be ordered against a seat.
            for key, games in sorted(
                self.settings.items(),
                key=lambda item: (item[0].seat is not None, item[0]),
            )
        ]
        self.counts.clear()
        self.settings.clear()
        return CompletedGames(outcomes, settings)
