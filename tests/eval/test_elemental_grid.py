"""MTG-181's frozen grid assigns every scheduled deck and seat through real arena legs."""

from collections import Counter
from pathlib import Path

import pytest

from experiments.elemental_grid import (
    DECKS,
    GAMES_PER_CELL,
    SCHEDULE_SHA256,
    GameRow,
    Group,
    _legs,
    groups,
    schedule,
    schedule_sha256,
    summarize,
)
from manabot.sim.matchup import AuthoredMatchup


def test_schedule_is_frozen_and_complete() -> None:
    games = schedule()
    assert schedule_sha256() == SCHEDULE_SHA256
    assert len(games) == len(DECKS) ** 2 * GAMES_PER_CELL
    assert len({g.deal for g in games}) == len(games)
    cells = Counter((g.model_deck, g.heuristic_deck) for g in games)
    assert set(cells.values()) == {GAMES_PER_CELL}
    assert sum(len(g.deals) for g in groups(games)) == len(games)


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize(
    ("model_deck", "heuristic_deck"),
    [("elemental_ur", "elemental_bg"), ("elemental_wr", "elemental_wr")],
)
def test_leg_puts_model_deck_in_scheduled_seat(
    model_deck: str, heuristic_deck: str, seat: int
) -> None:
    group = Group(
        model_deck=model_deck,
        heuristic_deck=heuristic_deck,
        model_seat=seat,  # type: ignore[arg-type]
        deals=(1,),
    )
    role_a, role_b, leg = _legs(group)
    seat_decks = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a=role_a, role_b=role_b
    ).decks(leg)
    # play_cell seats player A (the model) at leg % 2.
    assert leg % 2 == seat
    assert seat_decks[seat] == model_deck
    assert seat_decks[1 - seat] == heuristic_deck


def test_invalid_games_are_reported_not_scored(tmp_path: Path) -> None:
    def row(score: float | None, valid: bool) -> GameRow:
        return GameRow(
            model_deck=DECKS[0],
            heuristic_deck=DECKS[1],
            model_seat=0,
            deal=1,
            score=score,
            valid=valid,
            terminated=valid,
            truncated=False,
            failure=None if valid else "crash",
            decisions=1,
            model_decisions=1,
            seconds=1.0,
        )

    summary = summarize([row(1.0, True), row(0.5, True), row(None, False)])
    cell = summary.cells[DECKS[0]][DECKS[1]]
    assert (cell.score, cell.valid, cell.invalid) == (0.75, 2, 1)
    assert cell.low is not None and cell.high is not None
    assert cell.low < 0.75 < cell.high
    assert summary.cells[DECKS[1]][DECKS[0]].score is None
