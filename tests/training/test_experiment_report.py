"""Offline reporting preserves user edits and never promotes unmatched evidence."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manabot.training.report_study import ScientificStudy

from nbclient import NotebookClient
import nbformat

from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.experiment_report import (
    load_evidence,
    matched_milestones,
    write_dashboard,
)

DEMO = Path(__file__).resolve().parents[2] / "experiments/study/experiment-demo"


def test_saved_notebook_generates_html_without_inline_plots(tmp_path: Path) -> None:
    path = write_comparison_notebook(DEMO, tmp_path / "report.ipynb")
    notebook = nbformat.read(path, as_version=4)
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "comparison.html", "chosen.html"
    )
    notebook.cells.append(nbformat.v4.new_code_cell("personal_note = 'preserve me'"))
    nbformat.write(notebook, path)
    before = path.read_bytes()
    write_comparison_notebook(DEMO, path)
    assert path.read_bytes() == before
    NotebookClient(
        notebook, timeout=60, resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    html = (tmp_path / "chosen.html").read_text()
    assert html.count("<svg") == 5
    assert "Download all raw scalar diagnostics" in html
    assert "Initialization unavailable" not in html  # no scientific study
    assert "Monitoring initialization: unavailable" in html
    assert html.index("Monitoring strength versus time") < html.index(
        "Progress, throughput"
    )
    assert "Final experiment report" in html
    assert "#evaluation" in html and "#costs" in html
    assert not (tmp_path / "comparison.html").exists()
    assert all(
        "image/png" not in output.get("data", {})
        and "image/svg+xml" not in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )


def test_unequal_latest_updates_and_missing_runs_are_not_comparisons() -> None:
    evidence = load_evidence(DEMO)
    assert len(next(iter(matched_milestones(evidence).values()))) == 4
    first = evidence.monitors[0]
    later = first.model_copy(
        update={"coordinates": first.coordinates.model_copy(update={"updates": 2})}
    )
    assert not matched_milestones(
        replace(evidence, monitors=(later, *evidence.monitors[1:]))
    )
    execution = evidence.executions[0]
    pending = execution.attempts[0].model_copy(
        update={"run_id": None, "status": "pending"}
    )
    execution = execution.model_copy(
        update={"attempts": [*execution.attempts, pending]}
    )
    assert not matched_milestones(replace(evidence, executions=(execution,)))


def test_failed_evaluation_suppresses_rates_and_remains_visible(tmp_path: Path) -> None:
    shutil.copytree(
        DEMO,
        tmp_path / "data",
        ignore=shutil.ignore_patterns("*.ipynb", "*.html", ".ipynb_checkpoints"),
    )
    root = tmp_path / "data"
    attempt = next(root.rglob("attempt.json"))
    record = json.loads(attempt.read_text())
    record["status"] = "failed"
    record["error"] = "retained timeout"
    attempt.write_text(json.dumps(record))
    evidence = load_evidence(root)
    assert len(evidence.monitors) == 3
    assert not matched_milestones(evidence)
    execution = evidence.executions[0].model_copy(update={"status": "running"})
    evidence = replace(evidence, executions=(execution,))
    html = write_dashboard(
        evidence,
        tmp_path / "report.html",
        question="<script>unsafe</script>",
        docs="metrics.md",
        sections=[],
    ).read_text()
    assert "In-flight experiment report" in html
    assert "retained timeout" in html
    assert "<script>unsafe" not in html
    assert "pending / unavailable" in html


def test_missing_observation_times_and_resources_are_unavailable(
    tmp_path: Path,
) -> None:
    evidence = load_evidence(DEMO)
    runs = tuple(
        r.model_copy(
            update={
                "stages": [
                    s.model_copy(
                        update={"diagnostics": [], "sampled_peak_rss_bytes": None}
                    )
                    for s in r.stages
                ]
            }
        )
        for r in evidence.runs
    )
    report = write_dashboard(
        replace(evidence, runs=runs),
        tmp_path / "missing.html",
        question="Missing diagnostics",
        docs="metrics.md",
        sections=[],
    ).read_text()
    assert "lag unavailable training s" in report
    assert "RSS unavailable MiB" in report


def test_diagnostic_smoothing_preserves_missing_records_and_raw_data() -> None:
    import math

    from manabot.training.experiment_report import diagnostic_figures

    evidence = load_evidence(DEMO)
    run = evidence.runs[0]
    stage = run.stages[0]
    prototype = stage.diagnostics[0]
    rows = []
    for update, value in enumerate((2.0, 4.0, None, 8.0), 1):
        row = dict(prototype)
        row["coordinates"] = dict(
            prototype["coordinates"], updates=update, training_seconds=update
        )
        if value is None:
            row.pop("policy_loss", None)
        else:
            row["policy_loss"] = value
        rows.append(row)
    changed = run.model_copy(
        update={"stages": [stage.model_copy(update={"diagnostics": rows})]}
    )
    fig = diagnostic_figures(replace(evidence, runs=(changed,)), window=2)[0]
    smoothed, raw = fig.axes[0].lines[:2]
    assert list(smoothed.get_ydata())[:2] == [2, 3]
    assert math.isnan(smoothed.get_ydata()[2])
    assert smoothed.get_ydata()[3] == 8
    assert raw.get_ydata()[1] == 4


def _scientific_fixture() -> "ScientificStudy":
    from manabot.training.report_study import ScientificStudy

    evidence = load_evidence(DEMO)
    monitor = evidence.monitors[0]
    measurements = []
    cells = []
    for regime in ("a", "b"):
        for seed in (1, 2, 3):
            for cutoff in (0, 1):
                score = sum(
                    r.score_a for r in monitor.rows if r.score_a is not None
                ) / len(monitor.rows)
                measurements.append(
                    dict(
                        regime=regime,
                        seed=seed,
                        cutoff=cutoff,
                        opponent="anchor",
                        complete=True,
                        score=score,
                        training_seconds=(cutoff + 1) * seed,
                        learner_transitions=cutoff * 256,
                        decisions=cutoff * 512,
                    )
                )
                cells.append(
                    dict(
                        a=regime,
                        b="anchor",
                        training_seed=seed,
                        cutoff=cutoff,
                        scheduled_games=len(monitor.rows),
                        rows=[r.model_dump() for r in monitor.rows],
                        replay={"passed": True},
                    )
                )
    return ScientificStudy.model_validate(
        dict(
            study="fixture",
            status="completed",
            seconds=1,
            seeds=[1, 2, 3],
            runs=[{"regime": "a"}, {"regime": "b"}],
            measurements=measurements,
            comparisons=cells,
        )
    )


def test_scientific_cohort_matching_and_initialization() -> None:
    from manabot.training.report_study import study_strength_figures

    study = _scientific_fixture()
    figure = study_strength_figures(study)[0]
    assert len(figure.axes) == 2
    assert list(figure.axes[0].lines[0].get_xdata()) == [0, 256]
    assert len(figure.axes[0].containers) == 2  # independent-seed/deal intervals
    study.comparisons.pop()
    figure = study_strength_figures(study)[0]
    assert list(figure.axes[0].lines[0].get_xdata()) == [0]
    study.status = "failed"
    assert not study_strength_figures(study)


def test_scientific_mismatched_rows_and_worlds_are_not_silently_pooled() -> None:
    import pytest

    from manabot.training.report_study import study_strength_figures

    study = _scientific_fixture()
    study.measurements[0].score = 0.123
    with pytest.raises(ValueError, match="differs"):
        study_strength_figures(study)
    study = _scientific_fixture()
    for cell in study.comparisons:
        if cell.a == "b":
            for row in cell.rows:
                row.arena_key = row.arena_key.model_copy(
                    update={"world": "different-world"}
                )
    assert not study_strength_figures(study)


def test_missing_retention_is_not_zero_and_run_histories_stay_separate() -> None:
    from manabot.training.experiment_report import strength_figures

    evidence = load_evidence(DEMO)
    run = evidence.runs[0]
    stage = run.stages[0]
    diagnostic = dict(stage.diagnostics[0])
    diagnostic.pop("retained", None)
    diagnostic["rows"] = 256
    run = run.model_copy(
        update={"stages": [stage.model_copy(update={"diagnostics": [diagnostic]})]}
    )
    assert "rl/retained_fraction" not in evidence.metrics(run)[0]
    figures = strength_figures(evidence, "training_seconds", per_run=True)
    assert len(figures) == len(evidence.runs)
    assert all("Within-run" in f.axes[0].get_title() for f in figures)
