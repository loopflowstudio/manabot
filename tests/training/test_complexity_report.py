"""Offline rendering preserves notebook edits and exposes missing evidence."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.runners import complexity_report as report
from experiments.runners.environment_complexity_analysis import paired_improvement
from tests.training.test_complexity_analysis import cohort
from tests.training.test_complexity_campaign import gpu_plan, plan


def test_empty_report_does_not_invent_learning(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(plan().model_dump_json())
    report.write_report(tmp_path, tmp_path / "out.html")
    data = json.loads((tmp_path / "out.analysis.json").read_text())
    assert all(not panel["endpoint"]["available"] for panel in data["panels"])
    assert len(data["unavailable"]) == 9
    assert all(p["status"] == "unavailable" for p in data["progress"].values())
    path = report.notebook(tmp_path, tmp_path / "editable.ipynb")
    path.write_text("deliberate notebook edit")
    assert report.notebook(tmp_path, path).read_text() == "deliberate notebook edit"


def test_nonempty_summary_serializes_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = cohort()
    candidate = control.model_copy(deep=True)
    candidate.coordinates.updates = 1000
    candidate.validate_cohort()
    effect = paired_improvement(candidate, control, initialization=control.artifact)
    effects = {
        ("l1", anchor, seed, 1000): replace(effect, seed=seed)
        for anchor in ("scripted_greedy", "heuristic")
        for seed in (12251, 12252, 12253)
    }
    anchors = ("scripted_greedy", "random", "heuristic")
    evidence = report.Evidence(tmp_path, plan(), anchors, {}, effects, (), {}, 0, 48)
    monkeypatch.setattr(report, "load", lambda root: evidence)
    report.write_report(tmp_path, tmp_path / "complete.html")
    data = json.loads((tmp_path / "complete.analysis.json").read_text())
    assert data["panels"][0]["endpoint"]["available"]
    assert (
        data["panels"][0]["endpoint"]["seed_effects"][0]["coordinates"]["updates"]
        == 1000
    )
    # The supplementary anchor gets its own labelled panel, never a frozen one.
    assert [p["anchor"] for p in data["panels"][:3]] == list(anchors)
    assert data["panels"][2]["endpoint"]["available"]
    html = (tmp_path / "complete.html").read_text()
    assert html.count("Supplementary anchor") == 3
    assert "48/54000 attempted" in html


def test_gpu_report_retains_plan_and_marks_unrun_cohort_unavailable(
    tmp_path: Path,
) -> None:
    frozen = gpu_plan()
    (tmp_path / "plan.json").write_text(frozen.model_dump_json())
    evidence = report.load(tmp_path)
    assert evidence.plan.digest == frozen.digest
    assert evidence.plan.recipes()["l1"].stages[0].execution.device == "cuda"
    assert evidence.plan.recipes()["l1"].world == "w6"
    assert len(evidence.unavailable) == 9
    report.write_report(tmp_path, tmp_path / "gpu.html")
    data = json.loads((tmp_path / "gpu.analysis.json").read_text())
    assert all(not panel["endpoint"]["available"] for panel in data["panels"])


def test_unknown_plan_version_fails_closed(tmp_path: Path) -> None:
    payload = gpu_plan().model_dump(mode="json")
    payload["schema_version"] = 99
    (tmp_path / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        report.load(tmp_path)
