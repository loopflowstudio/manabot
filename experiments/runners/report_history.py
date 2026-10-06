"""Copy retained history evidence and execute a fresh editable report notebook.

No trainer, evaluator or tracker is invoked. Existing copies must have identical
bytes; existing notebooks are executed without rewriting their source or edits.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

from nbclient import NotebookClient
import nbformat
from pydantic import BaseModel

from manabot.arena.models import file_sha256
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorResult


class _CopiedEvidence(TypedDict):
    source: str
    copy: str
    sha256: str


class _Effect(BaseModel):
    mean_effect: float
    paired_seed_common_deal_95_percentile: tuple[float, float]


class _Contrast(BaseModel):
    disposition: str
    checkpoints: list[_Effect]
    limit: str


class _Supervisor(BaseModel):
    cumulative_seconds: float
    prior_seconds: float


def _history_notes(root: Path) -> str:
    contrast = _Contrast.model_validate_json(
        (root / "study/history-contrast.json").read_text()
    )
    supervisor = _Supervisor.model_validate_json((root / "supervisor.json").read_text())
    effects = "; ".join(
        f"cutoff {i}: {e.mean_effect:+.2%} [{e.paired_seed_common_deal_95_percentile[0]:+.2%}, "
        f"{e.paired_seed_common_deal_95_percentile[1]:+.2%}]"
        for i, e in enumerate(contrast.checkpoints)
    )
    return (
        f" History disposition: {contrast.disposition}. On−off score effects: {effects}; "
        "saved paired seed/common-deal intervals. "
        + contrast.limit
        + f" Supervisor charged {supervisor.cumulative_seconds:,.2f} s including "
        f"the prior {supervisor.prior_seconds:.2f} s failed calibration; evaluation is included, "
        "not additional. See study/history-contrast.json and study/report.md. "
        "This saved study is not a positive learning control."
    )


def _copy(source: Path, target: Path) -> _CopiedEvidence:
    payload = source.read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise ValueError(
            f"Retained copy differs: {target}; choose a fresh output directory"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(payload)
    return {"source": str(source), "copy": str(target), "sha256": file_sha256(source)}


def _execute(root: Path, notes: str) -> None:
    path = root / "report-generator.ipynb"
    fresh = not path.exists()
    write_comparison_notebook(root, path)
    notebook = nbformat.read(path, as_version=4)
    if fresh:
        notebook.cells[1].source += f"\nNOTES += {notes!r}"
        if (root / "supervisor.json").exists():
            notebook.cells[1].source += (
                "\nevidence = replace(evidence, paths=(*evidence.paths, "
                "DATA_ROOT / 'supervisor.json', DATA_ROOT / 'study/history-contrast.json'))"
            )
        if not (root / "study/study.json").exists():
            notebook.cells[3].source = (
                notebook.cells[3]
                .source.replace(
                    "strength_figures(evidence, 'training_seconds')",
                    "strength_figures(evidence, 'training_seconds', per_run=True)",
                )
                .replace(
                    "strength_figures(evidence, 'environment_decisions')",
                    "strength_figures(evidence, 'environment_decisions', per_run=True)",
                )
            )
        nbformat.write(notebook, path)
    before = path.read_bytes()
    NotebookClient(
        notebook, timeout=120, resources={"metadata": {"path": str(root)}}
    ).execute()
    assert path.read_bytes() == before
    print(root / "comparison.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--monitoring", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from retained source")
    scientific = output / "scientific"
    manifest: list[_CopiedEvidence] = []
    for name in (
        "study/study.json",
        "study/history-contrast.json",
        "study/report.md",
        "study/protocol.json",
        "study/recipes.json",
        "supervisor.json",
    ):
        manifest.append(_copy(source / name, scientific / name))
    runs = sorted((source / "training").rglob("run.json"))
    for path in runs:
        manifest.append(_copy(path, scientific / path.relative_to(source)))
    _execute(scientific, _history_notes(scientific))
    if args.monitoring:
        monitoring = args.monitoring.resolve()
        if (
            output == monitoring
            or monitoring in output.parents
            or output in monitoring.parents
        ):
            raise ValueError("Output must be separate from monitoring source")
        target = output / "monitoring"
        monitors = sorted(monitoring.rglob("monitor.json"))
        run_ids = {
            MonitorResult.model_validate_json(p.read_text()).run_id for p in monitors
        }
        for path in monitors:
            manifest.append(
                _copy(path, target / "monitoring" / path.relative_to(monitoring))
            )
        for path in monitoring.rglob("attempt.json"):
            manifest.append(
                _copy(path, target / "monitoring" / path.relative_to(monitoring))
            )
        for path in runs:
            if TrainingRun.model_validate_json(path.read_text()).id in run_ids:
                manifest.append(_copy(path, target / path.relative_to(source)))
        _execute(
            target,
            " Earlier single-seed monitoring only; independent deals and protocol. "
            "Do not combine with the three-seed scientific study. No live execution heartbeat retained.",
        )
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in manifest:
        assert file_sha256(Path(row["source"])) == row["sha256"]
        assert file_sha256(Path(row["copy"])) == row["sha256"]


if __name__ == "__main__":
    main()
