"""The game in numbers: what each seat of a played game did.

`game_numbers` re-executes a played game (`record_of`) once and reduces it
with the engine's own reducers (`managym.statistics.Behaviour` and `Totals`),
checking each step against the stored track as an analysis does. The result,
a `GameNumbers`, is saved beside the game (`attempts.numbers_json`) when it
ends, so lists never re-execute; an unfinished or older game gets its numbers
the first time they are read (`GET /api/traces/{id}/numbers`), and a step added
to the game clears them.
"""

from pydantic import BaseModel, ConfigDict

from managym.academy.attempts import RetainedAttempt
from managym.academy.recording import REEXECUTION_ERRORS, record_of, step_matches
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
    native seat order (seat 0 is on the play). `analysis_shown` counts the
    person's decisions committed with a live analysis shown (stored on each
    event, not re-executed; numbers saved before it existed read 0).
    `played_from` is the step a game played on from another starts at: the
    steps before it are that game's, so My record leaves the game out."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    complete: bool
    turns: int
    seats: tuple[SeatNumbers, SeatNumbers]
    analysis_shown: int = 0
    played_from: int | None = None


def game_numbers(attempt: RetainedAttempt) -> GameNumbers | None:
    """The game's numbers, or None where it does not re-execute as stored
    (another build or card set, or a divergence)."""
    trace = attempt.trace
    human = trace["config"].get("human_seat", 0)
    try:
        walker = walk(record_of(attempt), capture_cards=True)
    except REEXECUTION_ERRORS:
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
        seats=(
            seat_numbers(behaviour[0], totals[0]),
            seat_numbers(behaviour[1], totals[1]),
        ),
        analysis_shown=sum(1 for event in events if event.get("analysis_shown")),
        played_from=(trace.get("played_from") or {}).get("event"),
    )


def seat_numbers(behaviour: BehaviourRow, totals: TotalsRow) -> SeatNumbers:
    return SeatNumbers(
        seat=totals.seat,
        lands_played=behaviour.lands_played,
        spells_cast=totals.spells_cast,
        attacks=behaviour.attacks,
        cards_drawn=totals.cards_drawn,
        damage_dealt=totals.damage_dealt,
        damage_taken=totals.damage_taken,
    )
