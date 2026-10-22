"""Etude retains complete arena deal blocks and stable checkpoint-independent protocols."""

from pathlib import Path
from typing import Any

import pytest
from test_monitor_evaluation import _manifest, _rows

pytest.importorskip("etude.eval.match")

from etude.eval.query import connect  # noqa: E402
from etude.eval.tournament import curves  # noqa: E402

from manabot.arena.match import selected_decks, selected_match
from manabot.arena.models import file_sha256
from manabot.remote.live_evaluation import Ledger, _checkpoints, report
from manabot.training import etude_evaluation as evaluation
from manabot.training.models import (
    MonitoringCheckpoint,
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult
import managym


def _admission(tmp_path: Path) -> MonitorResult:
    manifest = _manifest()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    manifest.artifact = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    manifest.protocol = MonitorProtocol(
        deal_seeds=(101, 102), opponent="heuristic", include_mirrors=True
    )
    manifest.candidate = manifest.candidate.model_copy(
        update={
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
        }
    )
    return manifest


def _play_rows(manifest: MonitorResult, deal: int, out: Path) -> list[dict[str, Any]]:
    rows = _rows(manifest, (deal,))
    return [
        dict(
            row,
            leg=leg,
            score_a=float(deal == 102),
            seat_decks=list(selected_decks(leg)),
            player_a_seat=leg % 2,
        )
        for leg, row in enumerate(rows * 2)
    ]


def test_checkpoint_progress_does_not_change_benchmark(tmp_path: Path) -> None:
    first = _admission(tmp_path)
    second = first.model_copy(deep=True)
    second.coordinates.updates = 500
    second.artifact = {"path": "/other/500.pt", "bytes": 200, "sha256": "a" * 64}
    second.candidate = second.candidate.model_copy(
        update={
            "checkpoint_sha256": "a" * 64,
            "checkpoint_bytes": 200,
            "artifact_id": "monitor/second",
        }
    )
    assert evaluation.benchmark(first) == evaluation.benchmark(second)


def test_etude_resumes_complete_deals_and_report_counts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _admission(tmp_path)
    played: list[int] = []

    def play_cell(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        deal = kwargs["deal_seeds"][0]
        played.append(deal)
        return _play_rows(manifest, deal, kwargs["out_dir"]), {}, {}

    monkeypatch.setattr(evaluation, "play_cell", play_cell)
    target = tmp_path / "evaluations" / "one"
    evaluation.evaluate(manifest, target)
    evaluation.evaluate(manifest, target)
    assert sorted(played) == [101, 102]
    db = connect(tmp_path / "track")
    try:
        points = curves(db, evaluation.BENCHMARK, table="plays", value="score")[
            0
        ].points
        assert points[0].rows == 16
        assert points[0].matches == 2
        assert points[0].value == 0.5
    finally:
        db.close()
    ledger = Ledger(cohort_sha256="a" * 64, observed_at=1)
    report(tmp_path, ledger)
    assert "heuristic" in (tmp_path / "public/eval.html").read_text()
    assert "whole deals" in (tmp_path / "public/index.html").read_text()


def test_failed_deal_retains_evidence_and_etude_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _admission(tmp_path)

    def play_cell(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        rows = _play_rows(manifest, 101, kwargs["out_dir"])
        rows[0]["replay_passed"] = False
        return rows, {}, {}

    monkeypatch.setattr(evaluation, "play_cell", play_cell)
    with pytest.raises(Exception, match="match_failed"):
        evaluation.evaluate(manifest, tmp_path / "evaluations/one")
    assert list((tmp_path / "track").rglob("*.failed.json"))
    assert list((tmp_path / "evaluations/one/arena").rglob("rows.json"))
    assert not (tmp_path / "evaluations/one/result.json").exists()

    # A live run can have its declaration and failed match records before any
    # score table exists. Rendering that state must not stop the observer.
    report(tmp_path, Ledger(cohort_sha256="a" * 64, observed_at=1))
    assert (tmp_path / "public/status.json").is_file()


def test_changed_checkpoint_bytes_are_refused_before_declaring_eval(
    tmp_path: Path,
) -> None:
    manifest = _admission(tmp_path)
    Path(manifest.artifact["path"]).write_bytes(b"different weights")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        evaluation.evaluate(manifest, tmp_path / "evaluations/one")
    assert not (tmp_path / "track").exists()


@pytest.mark.parametrize(
    "status", ["paused", "completed", "running", "failed", "interrupted"]
)
def test_budget_endpoint_between_milestones_is_selected_only_after_export(
    status: str,
) -> None:
    regime = TrainingRegime(
        id="endpoint",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        stages=[
            TrainSelfPlay(
                id="self",
                operation="train_self_play",
                updates=1000,
                streams=2,
                transitions=8,
            )
        ],
    )
    milestone = MonitoringCheckpoint(
        stage_id="self",
        ordinal=0,
        updates=250,
        training_seconds=90,
        artifact={"path": "monitor.pt", "sha256": "a" * 64, "bytes": 1},
    )
    stage = StageRecord.model_validate(
        {
            "id": "self",
            "status": status,
            "cumulative_seconds": 100,
            "optimizer_updates": 253,
            "diagnostics": [{} for _ in range(253)],
            "artifacts": {"raw": {"path": "final.pt", "sha256": "b" * 64, "bytes": 2}},
        }
    )
    run = TrainingRun(
        id="endpoint",
        regime_digest="c" * 64,
        regime=regime,
        seed=1,
        seed_streams={},
        identities={},
        stages=[stage],
        monitoring_checkpoints=[milestone],
    )
    selected = _checkpoints(run)
    assert [item.coordinates.updates for item in selected] == (
        [250, 253] if status in {"paused", "completed"} else [250]
    )
    if status in {"paused", "completed"}:
        assert selected[-1].artifact["sha256"] == "b" * 64
        assert selected[-1].coordinates.training_seconds == 100
        stage.artifacts["raw"] = milestone.artifact
        assert len(_checkpoints(run)) == 1


def test_deck_labels_follow_candidate_seat_when_cross_decks_swap(
    tmp_path: Path,
) -> None:
    manifest = _admission(tmp_path)
    rows = [
        evaluation.ArenaRow.model_validate(row)
        for row in _play_rows(manifest, 101, tmp_path)
    ]
    projected = evaluation.arena_tables(rows)["plays"]
    expected = {"LvA": {0, 3}, "AvL": {1, 2}, "LvL": {4, 5}, "AvA": {6, 7}}
    for matchup, legs in expected.items():
        assert {
            row["game"]
            for row in projected
            if row["player"] == 0 and row["matchup"] == matchup
        } == legs
    assert [row["matchup"] for row in projected if row["player"] == 1][:4] == [
        "AvL",
        "LvA",
        "LvA",
        "AvL",
    ]
    corrupted = rows[0].model_dump(mode="json")
    corrupted["player_a_seat"] = 1
    rows[0] = evaluation.ArenaRow.model_validate(corrupted)
    with pytest.raises(ValueError, match="deck/seat assignment"):
        evaluation.arena_tables(rows)
