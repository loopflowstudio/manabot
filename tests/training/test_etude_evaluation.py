"""Etude retains complete arena deal blocks and stable checkpoint-independent protocols."""

from pathlib import Path
from typing import Any

import pytest
from test_monitor_evaluation import _manifest, _rows

pytest.importorskip("etude.eval.match")

from etude.eval.query import connect  # noqa: E402
from etude.eval.tournament import curves  # noqa: E402

from manabot.arena.models import file_sha256
from manabot.remote.live_evaluation import Ledger, report
from manabot.training import etude_evaluation as evaluation
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult


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
        dict(row, leg=leg, score_a=float(deal == 102))
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
