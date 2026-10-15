"""Offline rendering preserves notebook edits and exposes missing evidence."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.runners import complexity_report as report
from experiments.runners.environment_complexity_analysis import paired_improvement
from tests.training.test_complexity_analysis import cohort
from tests.training.test_complexity_campaign import plan


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
        ("l1", "scripted_greedy", seed, 1000): replace(effect, seed=seed)
        for seed in (12251, 12252, 12253)
    }
    evidence = report.Evidence(tmp_path, plan(), {}, effects, (), {}, 0, 48)
    monkeypatch.setattr(report, "load", lambda root: evidence)
    report.write_report(tmp_path, tmp_path / "complete.html")
    data = json.loads((tmp_path / "complete.analysis.json").read_text())
    assert data["panels"][0]["endpoint"]["available"]
    assert (
        data["panels"][0]["endpoint"]["seed_effects"][0]["coordinates"]["updates"]
        == 1000
    )
