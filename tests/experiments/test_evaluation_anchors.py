"""Cohort analysis refuses missing deals and keeps correlated seat legs together."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.runners import evaluation_anchors as anchors
from experiments.runners.evaluation_anchors import cell_scores, interval
from experiments.runners.heuristic_check import _relocate
from manabot.arena.match import selected_decks
from manabot.arena.models import file_sha256
from manabot.training.monitor_evaluation import ArenaRow
from tests.training.test_monitor_evaluation import _manifest, _rows


def test_complete_cells_and_deal_bootstrap() -> None:
    manifest = _manifest()
    raw = _rows(manifest)
    raw += [{**r, "leg": r["leg"] + 4} for r in raw]
    for row in raw:
        row.update(
            seat_decks=list(selected_decks(row["leg"])), player_a_seat=row["leg"] % 2
        )
    rows = [ArenaRow.model_validate(r) for r in raw]
    cells = cell_scores(rows, (101, 102))
    assert tuple(cells) == ("LvA", "AvL", "LvL", "AvA")
    for scores in cells.values():
        np.testing.assert_array_equal(scores, [0, 1])
        measured = interval(scores)
        assert (measured.mean, measured.lower, measured.upper) == (0.5, 0, 1)
    with pytest.raises(ValueError, match="incomplete"):
        cell_scores(rows[:-1], (101, 102))
    with pytest.raises(ValueError, match="duplicate"):
        cell_scores(rows + rows[:1], (101, 102))
    wrong = rows[0].model_copy(deep=True)
    wrong.__pydantic_extra__["player_a_seat"] = 1
    with pytest.raises(ValueError, match="seat/deck"):
        cell_scores([wrong, *rows[1:]], (101, 102))


def test_identical_retained_copies_are_one_artifact(tmp_path: Path) -> None:
    for folder in ("a", "b"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "weights.pt").write_bytes(b"frozen")
    artifact = {
        "path": "old/weights.pt",
        "bytes": 6,
        "sha256": file_sha256(tmp_path / "a/weights.pt"),
    }
    assert _relocate(artifact, tmp_path)["path"] == str(tmp_path / "a/weights.pt")


def _fixture_cohort(tmp_path: Path) -> anchors.Cohort:
    manifest = _manifest()
    names = ("init", "mini-12261", "mini-12262", "mini-12263", "gpu-12261", "gpu-12262")
    members = [
        anchors.Member(
            name=name,
            group="init"
            if name == "init"
            else "25k"
            if name.startswith("gpu")
            else "10k",
            producer=tmp_path / "run.json",
            point="endpoint",
            baseline=tmp_path / "baseline.json",
            baseline_sha256="a" * 64,
            baseline_rows=[],
            manifest=manifest,
        )
        for name in names
    ]
    pairings = [
        anchors.Pairing(
            name=f"{name}-vs-{anchor}", member=name, anchor=anchor, manifest=manifest
        )
        for name in names
        for anchor in ("heuristic", "checkpoint")
    ]
    pairings += [
        anchors.Pairing(name=name, member="mini-12261", anchor=name, manifest=manifest)
        for name in (
            "greedy-v-heuristic",
            "greedy-v-checkpoint",
            "heuristic-v-checkpoint",
        )
    ]
    return anchors.Cohort(
        source_sha256="a" * 64,
        native_sha256="b" * 64,
        analysis_sha256="c" * 64,
        members=members,
        pairings=pairings,
    )


def test_failed_attempt_is_retained_and_cannot_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _fixture_cohort(tmp_path)
    path = tmp_path / "cohort.json"
    path.write_text(cohort.model_dump_json())
    monkeypatch.setattr(anchors, "source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="source changed"):
        anchors.execute(path, tmp_path / "attempt", smoke=False)
    state = json.loads((tmp_path / "attempt/status.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_games"] == 0
    assert (tmp_path / "scientific-attempt.json").is_file()
    with pytest.raises(FileExistsError):
        anchors.execute(path, tmp_path / "second-attempt", smoke=False)
    found = anchors.report(path, tmp_path / "attempt")
    assert found["strength"] is None
    assert found["status"] == state
    with pytest.raises(ValueError, match="six-checkpoint"):
        anchors.Cohort.model_validate(
            {**cohort.model_dump(), "members": cohort.members[:-1]}
        )


def test_failed_game_stops_cohort_and_retains_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _fixture_cohort(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"frozen")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    for member in cohort.members:
        member.manifest.artifact = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        }
        member.baseline_sha256 = file_sha256(baseline)
    cohort.native_sha256 = file_sha256(anchors._managym.__file__)
    path = tmp_path / "cohort.json"
    path.write_text(cohort.model_dump_json())
    monkeypatch.setattr(anchors, "source_digest", lambda: cohort.source_sha256)
    calls = 0

    def failed_game(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        nonlocal calls
        calls += 1
        row = _rows(cohort.pairings[0].manifest)[0]
        row.update(failure="command cap", terminated=False)
        return [row], {}, {"passed": False}

    monkeypatch.setattr(anchors, "play_cell", failed_game)
    attempt = tmp_path / "failed-game"
    with pytest.raises(RuntimeError, match="failed game"):
        anchors.execute(path, attempt, smoke=False)
    assert calls == 1
    state = json.loads((attempt / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_games"] == 0
    rows = json.loads(next(attempt.glob("*/*/rows.json")).read_text())
    assert rows[0]["failure"] == "command cap"


def test_complete_report_orders_cells_and_reports_headroom(tmp_path: Path) -> None:
    cohort = _fixture_cohort(tmp_path)
    baseline = _rows(cohort.members[0].manifest)
    baseline += [{**row, "leg": row["leg"] + 4} for row in baseline]
    for row in baseline:
        row.update(
            seat_decks=list(selected_decks(row["leg"])), player_a_seat=row["leg"] % 2
        )
    for member in cohort.members:
        member.baseline_rows = [ArenaRow.model_validate(row) for row in baseline]
    path = tmp_path / "cohort.json"
    path.write_text(cohort.model_dump_json())
    attempt = tmp_path / "complete"
    attempt.mkdir()
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "cohort_sha256": file_sha256(path),
                "status": "completed",
                "smoke": False,
            }
        )
    )
    groups = {m.name: m.group for m in cohort.members}
    for pairing in cohort.pairings:
        score = {"init": 0.0, "10k": 0.5, "25k": 1.0}[groups[pairing.member]]
        directory = attempt / pairing.name / "all-rows"
        directory.mkdir(parents=True)
        (directory / "targeting.json").write_text(
            json.dumps(
                {
                    pairing.manifest.candidate.player_id: {
                        "cells": [
                            {
                                "cell": "LvA",
                                "counts": [
                                    {"seat": 0, "source": "removal", "count": 3},
                                    {"seat": 0, "source": "counter", "count": 1},
                                    {"seat": 1, "source": "removal", "count": 2},
                                ],
                            }
                        ],
                    },
                    pairing.manifest.opponent.player_id: {"cells": []},
                }
            )
        )
        (directory / "rows.json").write_text(
            json.dumps([{**row, "score_a": score} for row in baseline])
        )
    found = anchors.report(path, attempt)
    summary = found["anchor_summary"]["heuristic"]["LvA"]
    assert summary["orders_group_means"]
    assert summary["seed_spread"] == {"init": 0, "10k": 0, "25k": 0}
    assert summary["headroom_per_checkpoint"]["gpu-12261"] == 0
    assert found["recommendation"]["daily_regression"] == "heuristic"
    assert (
        found["paired_delta_from_greedy"]["gpu-12261-vs-heuristic"]["LvA"].mean == 0.5
    )
    assert found["scores"]["greedy-v-checkpoint"]["AvL"].mean == 0.5
    counts = found["targeting"][cohort.pairings[0].name]
    assert {
        (entry["seat"], entry["source"]): entry[
            "fraction_of_target_prompts_in_seat_cell"
        ]
        for entry in counts
    } == {(0, "removal"): 0.75, (0, "counter"): 0.25, (1, "removal"): 1.0}
    # A missing leg invalidates the report even if status claims completion.
    broken = attempt / cohort.pairings[0].name / "all-rows/rows.json"
    broken.write_text(json.dumps(baseline[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        anchors.report(path, attempt)

    broken.write_text(json.dumps(baseline))
    (broken.parent / "targeting.json").unlink()
    with pytest.raises(ValueError, match="missing targeting"):
        anchors.report(path, attempt)
