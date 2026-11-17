"""Outcome routing and count-weighted reporting, including unrecorded history."""

from copy import deepcopy

import pytest

from manabot.reports.self_play import self_play_points
from manabot.training.rollout.self_play import SelfPlayCounter
from manabot.training.update_metrics import OUTCOME_TABLE, metric_row
import managym


def test_both_seats_deck_swaps_and_drawn_games() -> None:
    counter = SelfPlayCounter()
    counter.record(("lessons", "allies"), 1)
    counter.record(("allies", "lessons"), 0)
    counter.record(("lessons", "allies"), None)
    rows = counter.drain()
    lessons_play = next(
        r for r in rows if r["deck"] == "lessons" and r["position"] == "play"
    )
    assert (lessons_play["wins"], lessons_play["losses"], lessons_play["draws"]) == (
        0,
        1,
        1,
    )
    assert sum(r["wins"] for r in rows) == 2
    assert sum(r["losses"] for r in rows) == 2
    assert sum(r["draws"] for r in rows) == 2
    assert counter.drain() == []


def test_mirror_and_recovery_keep_seat_counts() -> None:
    counter = SelfPlayCounter()
    counter.record(("allies", "allies"), 0)
    snapshot = deepcopy(counter)
    counter.drain()
    rows = snapshot.drain()
    assert len(rows) == 2
    assert sum(r["wins"] for r in rows) == 1
    assert sum(r["losses"] for r in rows) == 1
    assert next(r for r in rows if r["position"] == "play")["wins"] == 1


def test_identity_includes_sideboard_and_ignores_dictionary_order() -> None:
    assert (
        managym.Deck({"a": 2, "b": 1}).identity
        == managym.Deck({"b": 1, "a": 2}).identity
    )
    assert managym.Deck({"a": 2}).identity != managym.Deck({"a": 2}, {"b": 1}).identity


def test_windows_pool_counts_and_unrecorded_history_emits_nothing() -> None:
    assert self_play_points([{"coordinates/updates": 1, "loss": 0.5}]) == []
    counter = SelfPlayCounter()
    counter.record(("a", "b"), 0)
    rows = [metric_row({"coordinates": {"updates": 1}, OUTCOME_TABLE: counter.drain()})]
    for _ in range(9):
        counter.record(("a", "b"), 1)
    rows.append(
        metric_row({"coordinates": {"updates": 2}, OUTCOME_TABLE: counter.drain()})
    )
    points = self_play_points(rows, 100)
    first = next(p for p in points if p.deck == "a")
    assert (first.wins, first.games, first.update) == (1, 10, 2)
    assert len(self_play_points(rows, 1)) > len(points)
    with pytest.raises(ValueError, match="positive"):
        self_play_points(rows, 0)
