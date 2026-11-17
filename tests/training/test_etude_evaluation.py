"""Etude retains complete arena deal blocks and stable checkpoint-independent protocols."""

from pathlib import Path
from typing import Any

import pytest

from tests.training.test_monitor_evaluation import _manifest, _rows

pytest.importorskip("etude.eval.match")


from manabot.eval import benchmark as evaluation
from manabot.eval.monitor import MonitorProtocol, MonitorResult
from manabot.infra.identity import file_sha256
from manabot.sim.matchup import selected_decks


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


def test_changed_checkpoint_bytes_are_refused_before_declaring_eval(
    tmp_path: Path,
) -> None:
    manifest = _admission(tmp_path)
    Path(manifest.artifact["path"]).write_bytes(b"different weights")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        evaluation.evaluate(manifest, tmp_path / "evaluations/one")
    assert not (tmp_path / "track").exists()


def test_deck_labels_follow_candidate_seat_when_cross_decks_swap(
    tmp_path: Path,
) -> None:
    manifest = _admission(tmp_path)
    rows = [
        evaluation.EvaluationRow.model_validate(row)
        for row in _play_rows(manifest, 101, tmp_path)
    ]
    projected = evaluation.evaluation_tables(rows)["plays"]
    expected = {"LvA": {0, 3}, "AvL": {1, 2}, "LvL": {4, 5}, "AvA": {6, 7}}
    for matchup, legs in expected.items():
        assert {
            row["game"]
            for row in projected
            if row["seat"] == 0 and row["matchup"] == matchup
        } == legs
    assert [row["matchup"] for row in projected if row["seat"] == 1][:4] == [
        "AvL",
        "LvA",
        "LvA",
        "AvL",
    ]
    # The candidate plays Lessons in legs 0 and 3, so a deck matrix puts those
    # games in its Lessons column against the heuristic's Allies row.
    candidate = [row for row in projected if row["seat"] == 0]
    assert {row["game"] for row in candidate if row["deck"] == "ur_lessons"} == {
        0,
        3,
        4,
        5,
    }
    named = {"L": "ur_lessons", "A": "gw_allies"}
    for row in projected:
        assert (row["deck"], row["opponent_deck"]) == (
            named[row["matchup"][0]],
            named[row["matchup"][-1]],
        )
    corrupted = rows[0].model_dump(mode="json")
    corrupted["player_a_seat"] = 1
    rows[0] = evaluation.EvaluationRow.model_validate(corrupted)
    with pytest.raises(ValueError, match="deck/seat assignment"):
        evaluation.evaluation_tables(rows)


@pytest.mark.parametrize(
    "changed", [{"failure": "native error"}, {"replay_passed": False}]
)
def test_invalid_game_keeps_null_scores_for_both_etude_seats(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    manifest = _admission(tmp_path)
    rows = [
        evaluation.EvaluationRow.model_validate(row)
        for row in _play_rows(manifest, 101, tmp_path)
    ]
    rows[0] = rows[0].model_copy(update=changed)
    projected = evaluation.evaluation_tables(rows)["plays"]
    first = [row for row in projected if row["game"] == 0]
    assert len(first) == 2
    assert all(row["score"] is None and row["valid"] is False for row in first)
