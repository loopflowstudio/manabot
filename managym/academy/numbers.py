"""The game in numbers: what each seat of a played game did, and a record.

`game_numbers` re-executes a played game (`record_of`) once and reduces it
with the engine's own reducers (`managym.statistics.Behaviour` and `Totals`),
checking each step against the stored track as an analysis does. The result,
a `GameNumbers`, is saved beside the game (`attempts.numbers_json`) once the
game is no longer being played, so lists never re-execute; an older game gets
its numbers the first time they are read (`GET /api/traces/{id}/numbers`).

`wilson_interval` bounds a win rate for the My games record table.
"""

import math

from pydantic import BaseModel, ConfigDict

import managym
from managym.academy.attempts import RetainedAttempt
from managym.academy.recording import record_of, step_matches
from managym.replay import walk
from managym.statistics import Behaviour, BehaviourRow, Reduction, Totals, TotalsRow


class SeatNumbers(BaseModel):
    """One native seat's counts (definitions in `managym.statistics`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    seat: int
    lands_played: int
    spells_cast: int
    attacks: int
    cards_drawn: int
    damage_dealt: int | None
    damage_taken: int | None


class GameNumbers(BaseModel):
    """A played game's numbers. `complete` is False for a game that stopped
    before game over; its counts run to where it stopped. `seats` are in
    native seat order (seat 0 is on the play)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    complete: bool
    turns: int
    seats: tuple[SeatNumbers, SeatNumbers]


def game_numbers(attempt: RetainedAttempt) -> GameNumbers | None:
    """The game's numbers, or None where it does not re-execute as stored
    (another build or card set, or a divergence)."""
    trace = attempt.trace
    if trace["end_reason"] == "active":
        raise ValueError("A game still being played has no numbers yet.")
    human = trace["config"].get("human_seat", 0)
    try:
        walker = walk(record_of(attempt), capture_cards=True)
    except (KeyError, TypeError, ValueError, managym.AgentError):
        return None
    reduction = Reduction([Behaviour(), Totals()])
    reduction.start(walker.game.players)
    events = trace["events"]
    for index, decision in enumerate(walker):
        if index >= len(events) or not step_matches(
            decision.observation, events[index], human
        ):
            return None
        reduction.decision(decision)
    stop = walker.stop
    if stop is None or stop.status == "diverged" or stop.index != len(events):
        return None
    reduction.end(stop, walker.cards())
    behaviour: list[BehaviourRow]
    totals: list[TotalsRow]
    behaviour, totals = reduction.tables()
    return GameNumbers(
        complete=stop.status == "complete",
        turns=totals[0].turns,
        seats=(_seat(behaviour[0], totals[0]), _seat(behaviour[1], totals[1])),
    )


def _seat(behaviour: BehaviourRow, totals: TotalsRow) -> SeatNumbers:
    return SeatNumbers(
        seat=totals.seat,
        lands_played=behaviour.lands_played,
        spells_cast=totals.spells_cast,
        attacks=behaviour.attacks,
        cards_drawn=totals.cards_drawn,
        damage_dealt=totals.damage_dealt,
        damage_taken=totals.damage_taken,
    )


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a win rate (95% by default); draws count
    as games that are not wins."""
    if games <= 0:
        return (0.0, 1.0)
    rate = wins / games
    denominator = 1 + z * z / games
    centre = (rate + z * z / (2 * games)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    half /= denominator
    return (max(0.0, centre - half), min(1.0, centre + half))
