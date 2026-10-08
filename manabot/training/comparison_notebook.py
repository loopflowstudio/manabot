"""Create the editable report generator once; preserve all existing notebook bytes.

Cells only read retained artifacts and write a read-only HTML dashboard. Plot
selection stays editable here; exhaustive analysis lives in experiment_report.
"""

import os
from pathlib import Path


def write_comparison_notebook(data_root: Path, output: Path) -> Path:
    """Create once, including under races. Existing notebooks require explicit migration."""
    import nbformat

    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(data_root.resolve(), output.parent.resolve())
    docs = os.path.relpath(
        Path(__file__).resolve().parents[2] / "docs/experiment-metrics.md",
        output.parent.resolve(),
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Experiment report generator\n\nRun All reads saved evidence and writes `comparison.html`, "
            "the default viewing surface. No training, evaluation games or tracker access. "
            "Edit discovery, question and plotting code below; report refresh never overwrites this notebook. "
            "Figures render only in HTML. A reproducible descriptive reading is derived from saved rows; research decisions remain separate."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "from dataclasses import replace\n"
            "from manabot.training.experiment_report import (\n"
            "    load_evidence, metric_figure, diagnostic_figures, strength_figures, write_dashboard,\n"
            ")\n\n"
            "from manabot.training.report_study import load_study, study_strength_figures\n\n"
            f"DATA_ROOT = Path({relative!r}).resolve()\n"
            "OUTPUT = Path('comparison.html')\n"
            f"NOTEBOOK = Path({output.name!r})\n"
            f"METRIC_DOCS = {docs!r}\n"
            "QUESTION = 'How do the declared variants progress at shared training milestones?'\n"
            "# Narrow DATA_ROOT to a retained execution; never combine incompatible studies.\n"
            "evidence = load_evidence(DATA_ROOT)\n"
            "STUDY_PATH = DATA_ROOT / 'study' / 'study.json'\n"
            "study = load_study(STUDY_PATH) if STUDY_PATH.exists() else None\n"
            "if study is not None:\n"
            "    evidence = replace(evidence, paths=(*evidence.paths, STUDY_PATH))\n"
            "NOTES = 'Scientific study: unavailable.' if study is None else (\n"
            "    f'Scientific study: {study.status}; {len(study.seeds)} independent seeds. '\n"
            "    + ('Initialization retained.' if any(m.learner_transitions == 0 for m in study.measurements)\n"
            "       else 'Initialization unavailable: no zero-transition evaluation retained.')\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "## Choose the report\n\nChange metrics or axes here. `evidence.metrics(run)` exposes every "
            "saved scalar; `metric_figure` plots any of them. The default dashboard shows each seed separately; cross-run comparisons require matched milestones. "
            "No notebook cell displays or launches a figure inline."
        ),
        nbformat.v4.new_code_cell(
            "# Editable plotting code. Keep the default report concise.\n"
            "sections = [\n"
            "    ('Matched training steps', 'comparisons', strength_figures(evidence)),\n"
            "    ('Matched milestones: learner sample exposure', 'comparisons', strength_figures(evidence, 'learner_transitions')),\n"
            "    ('Matched milestones: active training cost', 'costs', strength_figures(evidence, 'active_training_seconds')),\n"
            "]\n"
            "if study is not None:\n"
            "    sections += [\n"
            "        ('Scientific checkpoint strength', 'comparisons', study_strength_figures(study)),\n"
            "        ('Scientific checkpoints at recorded cost', 'comparisons', study_strength_figures(study, 'training_seconds')),\n"
            "    ]\n"
            "sections += [\n"
            "    ('Policy, value, regularization and sample retention', 'sampling', diagnostic_figures(evidence)),\n"
            "    ('Sampled process memory', 'costs', [metric_figure(evidence, 'progress/rss_bytes')]),\n"
            "]\n"
            "# For deeper analysis, add e.g. metric_figure(evidence, 'rl/collection_kl')."
        ),
        nbformat.v4.new_code_cell(
            "report = write_dashboard(evidence, OUTPUT, question=QUESTION, docs=METRIC_DOCS, sections=sections, notes=NOTES, study=study, notebook=NOTEBOOK)\n"
            "print(f'Read-only report: {report.resolve()}')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation (editable)\n\nRecord observations with evidence paths and hashes from HTML. "
            "Keep hypotheses and conclusions separate. Editing this cell does not change saved measurements."
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    try:
        with output.open("x") as stream:
            nbformat.write(notebook, stream)
    except FileExistsError:
        pass
    return output
