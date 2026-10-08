"""Deck routing, paired uncertainty and honest missing-cohort presentation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manabot.training.experiment_report import (
    RetainedRun,
    load_evidence,
    write_dashboard,
)
from manabot.training.learning_dashboard import (
    candidate_deck,
    checkpoint_panels,
    deck_results,
)
from manabot.training.monitor_evaluation import MonitorResult

DEMO = Path(__file__).resolve().parents[2] / "experiments/study/experiment-demo"


def _monitor() -> MonitorResult:
    return load_evidence(DEMO).monitors[0]


def test_deck_follows_actual_candidate_seat_not_leg_number() -> None:
    monitor = _monitor()
    rows = []
    for row in monitor.rows:
        copy = row.model_copy(deep=True)
        copy.score_a = float(candidate_deck(row) == "gw_allies")
        rows.append(copy)
    result = monitor.model_copy(update={"rows": rows, "win": None})
    decks = {d.deck: d for d in deck_results(result)}
    assert decks["gw_allies"].wins == 2
    assert decks["ur_lessons"].wins == 0
    for row in rows:
        extra = row.model_extra
        assert extra is not None
        extra["player_a_seat"] = 1 - extra["player_a_seat"]
        extra["seat_decks"] = list(reversed(extra["seat_decks"]))
    assert deck_results(result) == tuple(decks.values())


def test_draw_is_not_a_win_and_deals_keep_both_seats() -> None:
    monitor = _monitor()
    rows = []
    seeds = (100, 101, 102)
    for seed, scores in zip(
        seeds,
        ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)),
        strict=True,
    ):
        rows.extend(
            r.model_copy(update={"deal_seed": seed, "score_a": score})
            for r, score in zip(monitor.rows, scores, strict=True)
        )
    protocol = monitor.protocol.model_copy(update={"deal_seeds": seeds})
    result = monitor.model_copy(
        update={"rows": rows, "protocol": protocol, "expected_games": 12, "win": None}
    )
    for deck in deck_results(result):
        assert (deck.wins, deck.draws, deck.games) == (2, 2, 6)
        indices = np.random.default_rng(protocol.bootstrap_seed).integers(
            0, 3, (protocol.bootstrap_replicates, 3)
        )
        expected = np.quantile(
            np.array([1.0, 0.0, 0.0])[indices].mean(axis=1), [0.025, 0.975]
        )
        assert (deck.interval.lower, deck.interval.upper) == pytest.approx(expected)


def test_missing_and_incomplete_decks_do_not_become_zero() -> None:
    monitor = _monitor()
    assert deck_results(monitor.model_copy(update={"status": "incomplete"})) == ()
    changed = monitor.model_copy(deep=True)
    assert changed.rows[0].model_extra is not None
    changed.rows[0].model_extra.pop("player_a_seat")
    assert deck_results(changed) == ()
    with pytest.raises(ValueError, match="missing or duplicate"):
        deck_results(monitor.model_copy(update={"rows": monitor.rows[:-1]}))
    with pytest.raises(ValueError, match="win mean"):
        deck_results(
            monitor.model_copy(
                update={"win": monitor.win.model_copy(update={"mean": 0.123})}
            )
        )


def test_seeds_and_opponent_identities_never_share_curves() -> None:
    evidence = load_evidence(DEMO)
    first = evidence.monitors[0]
    foreign = first.model_copy(
        update={
            "opponent": first.opponent.model_copy(
                update={"display_name": "Different opponent"}
            )
        }
    )
    assert (
        len(
            checkpoint_panels(replace(evidence, monitors=(*evidence.monitors, foreign)))
        )
        == len(evidence.monitors) + 1
    )
    assert all(
        len({r.training_seed for r in p.results}) == 1
        for p in checkpoint_panels(evidence)
    )


def test_newer_metadata_is_preserved_only_in_read_only_projection() -> None:
    run = load_evidence(DEMO).runs[0].model_dump(mode="json")
    run["calendar_seconds"] = 99
    run["regime"]["recovery"] = {"checkpoint_updates": 128}
    projected = RetainedRun.model_validate(run)
    assert projected.model_dump()["regime"]["recovery"] == {"checkpoint_updates": 128}
    assert projected.model_dump()["calendar_seconds"] == 99
    run["seed"] = "not a number"
    with pytest.raises(ValueError):
        RetainedRun.model_validate(run)


def test_dashboard_uses_portable_guidance_and_keeps_missing_runs_visible(
    tmp_path: Path,
) -> None:
    evidence = load_evidence(DEMO)
    html = write_dashboard(
        replace(evidence, monitors=()),
        tmp_path / "report.html",
        question="Fixture",
        docs="missing.md",
        sections=[],
    ).read_text()
    assert "No monitoring evaluations retained" in html
    assert "pending / unavailable" in html
    assert "report-metrics.html#evaluation" in html
    assert 'id="evaluation"' in (tmp_path / "report-metrics.html").read_text()
    assert (tmp_path / "report-monitoring.json").exists()
    assert "cdn." not in html


def test_snapshot_reads_wal_consistently_and_preserves_payloads(tmp_path: Path) -> None:
    import json
    import sqlite3

    from manabot.training.report_snapshot import snapshot_report

    source = tmp_path / "source"
    source.mkdir()
    run = load_evidence(DEMO).runs[0]
    con = sqlite3.connect(source / "experiment.sqlite")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        "CREATE TABLE training_runs (id TEXT, payload TEXT); CREATE TABLE training_stages (run_id TEXT, stage_id TEXT, payload TEXT); CREATE TABLE experiment_runs (id TEXT, payload TEXT);"
    )
    con.execute(
        "INSERT INTO training_runs VALUES (?, ?)",
        (run.id, run.model_dump_json(exclude={"stages"})),
    )
    for stage in run.stages:
        con.execute(
            "INSERT INTO training_stages VALUES (?, ?, ?)",
            (run.id, stage.id, stage.model_dump_json()),
        )
    con.commit()
    output = snapshot_report(source, tmp_path / "snapshot")
    exported = load_evidence(output).runs[0]
    assert exported.model_dump() == run.model_dump()
    receipt = json.loads((output / "snapshot.json").read_text())
    assert receipt["database_sha256"]
    assert receipt["files"]
    with pytest.raises(FileExistsError):
        snapshot_report(source, output)
    con.close()


def test_supervised_epoch_plans_are_not_reported_as_zero_updates(
    tmp_path: Path,
) -> None:
    from manabot.training.models import TrainSupervised

    evidence = load_evidence(DEMO)
    run = evidence.runs[0]
    stage = TrainSupervised(
        id=run.stages[0].id,
        operation="train_supervised",
        datasets=["teacher"],
        epochs=7,
    )
    run = run.model_copy(
        update={"regime": run.regime.model_copy(update={"stages": [stage]})}
    )
    html = write_dashboard(
        replace(evidence, runs=(run,), monitors=()),
        tmp_path / "report.html",
        question="Epoch plan",
        docs="missing.md",
        sections=[],
    ).read_text()
    assert "7 (includes epochs)" in html
