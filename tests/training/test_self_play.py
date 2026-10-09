"""Outcome routing and count-weighted reporting, including historical absence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from manabot.training.experiment_report import load_evidence
from manabot.training.self_play import SelfPlayCounter, deck_identity
from manabot.training.self_play_report import self_play_figures, self_play_points


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
    assert deck_identity({"a": 2, "b": 1}, {}) == deck_identity({"b": 1, "a": 2}, {})
    assert deck_identity({"a": 2}, {}) != deck_identity({"a": 2}, {"b": 1})


def test_windows_pool_counts_and_old_reports_remain_unavailable() -> None:
    evidence = load_evidence(Path("experiments/study/experiment-demo"))
    run = evidence.runs[0].model_copy(deep=True)
    stage = run.stages[0]
    stage.diagnostics = []
    assert self_play_points(run) == []
    counter = SelfPlayCounter()
    counter.record(("a", "b"), 0)
    stage.diagnostics.append(
        {"coordinates": {"updates": 1}, "self_play_outcomes": counter.drain()}
    )
    for _ in range(9):
        counter.record(("a", "b"), 1)
    stage.diagnostics.append(
        {"coordinates": {"updates": 2}, "self_play_outcomes": counter.drain()}
    )
    points = self_play_points(run, 100)
    first = next(p for p in points if p.deck == "a")
    assert (first.wins, first.games, first.update) == (1, 10, 2)
    figures = self_play_figures(replace(evidence, runs=(run,)))
    assert len(figures) == 1
    assert figures[0].axes[0].lines[1].get_ydata()[0] in (0.1, 0.9)
    stage.diagnostics = []
    missing = self_play_figures(replace(evidence, runs=(run,)))
    assert "unavailable" in missing[0].axes[0].texts[0].get_text()
    with pytest.raises(ValueError, match="positive"):
        self_play_points(run, 0)
