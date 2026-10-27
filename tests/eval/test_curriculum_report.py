"""Small grids verify equal cells, paired seats and visible missing outcomes."""

import pytest

from experiments.curriculum_report import ARMS, Arm, Bounds, compare
from experiments.elemental_grid import Game, GameRow


def _schedule() -> list[Game]:
    return [
        Game(deck, "opponent", 2 * deal + seat, deal, seat)
        for deck, deals in (("a", (10, 11)), ("b", (20,)))
        for deal in deals
        for seat in (0, 1)
    ]


def _results(schedule: list[Game]) -> dict[Arm, list[GameRow]]:
    return {
        arm: [
            GameRow(
                model_deck=g.model_deck,
                heuristic_deck=g.heuristic_deck,
                model_seat=g.model_seat,
                deal=g.deal,
                score=float(
                    (g.model_deck == "a" and (g.model_seat == 0 or arm in {"M", "ME"}))
                    or (g.model_deck == "b" and arm in {"E", "ME"})
                ),
                valid=True,
                terminated=True,
                truncated=False,
                failure=None,
                decisions=1,
                model_decisions=1,
                seconds=0.0,
            )
            for g in schedule
        ]
        for arm in ARMS
    }


def test_equal_cell_and_seat_weights_and_paired_interaction() -> None:
    schedule = _schedule()
    result = compare(schedule, _results(schedule), seed=1, replicates=8)
    assert result.complete
    assert result.scores == {"U": 0.25, "M": 0.5, "E": 0.75, "ME": 1.0}
    expected = {"M-U": 0.25, "E-U": 0.5, "ME-U": 0.75, "ME-M-E+U": 0.0}
    for name, value in expected.items():
        estimate = result.contrasts[name]
        # The seats disagree, but complete-deal averages are constant per cell.
        assert (estimate.value, estimate.low, estimate.high) == (value, value, value)


def test_missing_and_invalid_games_do_not_become_a_changed_population() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    rows["U"][0] = rows["U"][0].model_copy(update={"valid": False, "score": None})
    rows["ME"].pop()
    result = compare(schedule, rows, seed=1, replicates=8)
    assert not result.complete
    assert result.scores == result.contrasts == {}
    assert result.coverage["U"].invalid == 1
    assert result.coverage["U"].missing == 0
    assert result.coverage["ME"].missing == 1
    assert result.coverage["ME"].valid == len(schedule) - 1
    assert result.score_bounds["U"] == Bounds(0.125, 0.25)
    assert result.score_bounds["ME"] == Bounds(0.75, 1.0)
    assert result.contrast_bounds["ME-U"] == Bounds(0.5, 0.875)


def test_duplicates_and_off_schedule_rows_are_rejected() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    rows["M"].append(rows["M"][0])
    with pytest.raises(ValueError, match="duplicate"):
        compare(schedule, rows, seed=1, replicates=8)
    rows["M"] = rows["M"][:-1]
    rows["M"][0] = rows["M"][0].model_copy(update={"deal": 999})
    with pytest.raises(ValueError, match="off-schedule"):
        compare(schedule, rows, seed=1, replicates=8)


def test_incomplete_seat_schedule_and_invalid_valid_scores_are_rejected() -> None:
    schedule = _schedule()
    with pytest.raises(ValueError, match="both seats"):
        compare(schedule[:-1], _results(schedule[:-1]), seed=1, replicates=8)
    rows = _results(schedule)
    rows["E"][0] = rows["E"][0].model_copy(update={"score": float("nan")})
    with pytest.raises(ValueError, match="finite score"):
        compare(schedule, rows, seed=1, replicates=8)


def test_shared_deal_noise_cancels_from_paired_contrasts() -> None:
    schedule = _schedule()
    rows = _results(schedule)
    shared = [
        row.model_copy(update={"score": float(row.deal == 10)}) for row in rows["U"]
    ]
    result = compare(schedule, {arm: shared for arm in ARMS}, seed=9, replicates=16)
    assert result.scores == dict.fromkeys(ARMS, 0.25)
    for estimate in result.contrasts.values():
        assert (estimate.value, estimate.low, estimate.high) == (0.0, 0.0, 0.0)
