"""Observed self-play outcomes of one run, separate from playing strength.

Aggregate counts within update windows, never average per-update percentages.
Stratify by actual deck matchup and keep seeds separate. Both seats use
the current self-play policy: these points measure matchup and first-turn balance,
not performance against a fixed opponent.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from manabot.training.update_metrics import MetricRow


@dataclass(frozen=True)
class SelfPlayPoint:
    update: int
    deck: str
    opponent_deck: str
    position: Literal["play", "draw"]
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws


def self_play_points(
    rows: Sequence[MetricRow], window_updates: int = 100
) -> list[SelfPlayPoint]:
    """Joint counts per window of one run's rows; unrecorded history emits nothing."""
    if window_updates < 1:
        raise ValueError("window_updates must be positive")
    groups: dict[tuple[int, str, str, Literal["play", "draw"]], list[int]] = {}
    for row in rows:
        update = row.get("coordinates/updates")
        if not isinstance(update, int):
            continue
        for name, count in row.items():
            parts = name.split("/")
            if len(parts) != 5 or parts[0] != "self_play_outcomes":
                continue
            _, deck, opponent, seat, outcome = parts
            if seat not in ("play", "draw") or not isinstance(count, int):
                raise ValueError(f"invalid self-play outcome {name!r}")
            position: Literal["play", "draw"] = "play" if seat == "play" else "draw"
            key = ((max(update, 1) - 1) // window_updates, deck, opponent, position)
            totals = groups.setdefault(key, [0, 0, 0, update])
            totals[("wins", "losses", "draws").index(outcome)] += count
            totals[3] = max(totals[3], update)
    return [
        SelfPlayPoint(update, deck, opponent, position, wins, losses, draws)
        for (_, deck, opponent, position), (wins, losses, draws, update) in sorted(
            groups.items()
        )
        if wins + losses + draws
    ]
