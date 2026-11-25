"""Observed self-play outcomes of one run, separate from playing strength.

Aggregate counts within update windows, never average per-update percentages.
Stratify by actual deck matchup and keep seeds separate. Both seats use
the current self-play policy: these points measure matchup and first-turn balance,
not performance against a fixed opponent.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
import warnings

from manabot.training.update_metrics import (
    OUTCOME_COUNTS,
    SEAT_LABELS,
    MetricRow,
    parse_outcome_scalar,
)


@dataclass(frozen=True)
class SelfPlayPoint:
    """One deck's pooled results against another from one native seat (0 is on the play)."""

    update: int
    deck: str
    opponent_deck: str
    seat: int
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def position(self) -> Literal["play", "draw"]:
        warnings.warn(
            "SelfPlayPoint.position is deprecated and will be removed in 0.3.0; "
            "use seat (0 is on the play)",
            DeprecationWarning,
            stacklevel=2,
        )
        return SEAT_LABELS[self.seat]


def self_play_points(
    rows: Sequence[MetricRow], window_updates: int = 100
) -> list[SelfPlayPoint]:
    """Joint counts per window of one run's rows; unrecorded history emits nothing."""
    if window_updates < 1:
        raise ValueError("window_updates must be positive")
    groups: dict[tuple[int, str, str, int], list[int]] = {}
    for row in rows:
        update = row.get("coordinates/updates")
        if not isinstance(update, int):
            continue
        for name, count in row.items():
            parsed = parse_outcome_scalar(name)
            if parsed is None:
                continue
            deck, opponent, seat, outcome = parsed
            if not isinstance(count, int):
                raise ValueError(f"invalid self-play outcome {name!r}")
            key = ((max(update, 1) - 1) // window_updates, deck, opponent, seat)
            totals = groups.setdefault(key, [0, 0, 0, update])
            totals[OUTCOME_COUNTS.index(outcome)] += count
            totals[3] = max(totals[3], update)
    return [
        SelfPlayPoint(update, deck, opponent, seat, wins, losses, draws)
        for (_, deck, opponent, seat), (wins, losses, draws, update) in sorted(
            groups.items()
        )
        if wins + losses + draws
    ]
