"""Report refresh preserves authored notebooks and rejects changed evidence."""

from pathlib import Path

import nbformat
import pytest

from experiments.runners.report_history_wandb import (
    ReportPlan,
    publish_plan,
    write_notebook,
)
from manabot.arena.models import file_sha256


def test_generator_preserves_existing_edits(tmp_path: Path) -> None:
    path = tmp_path / "report-generator.ipynb"
    write_notebook(tmp_path, path)
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_markdown_cell("Personal analysis stays here.")
    )
    nbformat.write(notebook, path)
    original = path.read_bytes()
    write_notebook(tmp_path, path)
    assert path.read_bytes() == original
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "publish_plan" not in code
    assert "report-plan.json" in code


def test_changed_evidence_rejected_before_publication(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    source.write_text("{}")
    plan = ReportPlan(
        title="fixture",
        introduction="fixture",
        sources={str(source): file_sha256(source)},
        projections=[],
        sections=[],
        evidence_notes=[],
    )
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="source changed"):
        publish_plan(plan, tmp_path)
