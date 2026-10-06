"""Offline reporting preserves user edits and never promotes unmatched evidence."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

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
    assert html.count("<svg") == 4
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
