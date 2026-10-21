"""Offline MTG-122 paired evidence and create-once notebook/HTML projection.

Only retained TrainingRun/arena JSON is read. Missing cells and fits remain
unavailable. No cell executes games, launches jobs or selects a checkpoint.
The dashboard uses the shared report style and numerical analysis owner.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from io import StringIO
import json
import os
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import TypeAdapter

from experiments.runners.complexity_campaign import (
    FROZEN_ANCHORS,
    POOLS,
    SEEDS,
    Anchor,
    AnyComplexityPlan,
    read_complexity_plan,
)
from experiments.runners.environment_complexity_analysis import (
    CohortEffect,
    SeedEffect,
    observed_progress,
    paired_improvement,
    three_seed_summary,
)
from manabot.arena.models import file_sha256
from manabot.training.learning_dashboard import DASHBOARD_CSS
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorResult


@dataclass(frozen=True)
class Evidence:
    root: Path
    plan: AnyComplexityPlan
    # Frozen anchors, then `heuristic` when any of its cells were scored.
    anchors: tuple[Anchor, ...]
    runs: dict[tuple[str, int], TrainingRun]
    effects: dict[tuple[str, str, int, int], SeedEffect]
    unavailable: tuple[str, ...]
    hashes: dict[str, str]
    evaluation_seconds: float
    evaluated_games: int


def load(root: Path) -> Evidence:
    root = root.resolve()
    plan = read_complexity_plan((root / "plan.json").read_text())
    hashes = {"plan.json": file_sha256(root / "plan.json")}
    runs: dict[tuple[str, int], TrainingRun] = {}
    effects: dict[tuple[str, str, int, int], SeedEffect] = {}
    missing: list[str] = []
    evaluation_seconds, games = 0.0, 0
    supplementary: tuple[Anchor, ...] = (
        ("heuristic",) if any(root.glob("*/*-heuristic-*/monitor.json")) else ()
    )
    anchors = FROZEN_ANCHORS + supplementary
    for pool in POOLS:
        for seed in SEEDS:
            directory = root / f"{pool}-{seed}"
            path = directory / "training/run.json"
            if not path.exists():
                missing.append(f"{pool}/{seed}: training not present")
                continue
            run = TrainingRun.model_validate_json(path.read_text())
            runs[pool, seed] = run
            hashes[str(path.relative_to(root))] = file_sha256(path)
            if run.seed != seed or run.regime != plan.recipes()[pool]:
                raise ValueError("retained training identity differs from frozen plan")
            if not run.stages or "initial_raw" not in run.stages[0].artifacts:
                missing.append(f"{pool}/{seed}: initialization unavailable")
                continue
            for anchor in anchors:
                for update in (0, plan.updates // 2, plan.updates):
                    retained: list[MonitorResult] = []
                    variants = (
                        ("candidate",) if update == 0 else ("candidate", "control")
                    )
                    for variant in variants:
                        path = directory / f"{update}-{anchor}-{variant}/monitor.json"
                        if not path.exists():
                            missing.append(
                                f"{pool}/{seed}/{update}/{anchor}/{variant}: not present"
                            )
                            continue
                        result = MonitorResult.model_validate_json(path.read_text())
                        hashes[str(path.relative_to(root))] = file_sha256(path)
                        evaluation_seconds += result.evaluation_seconds
                        games += len(result.rows)
                        if (
                            result.protocol.study_protocol_sha256 != plan.digest
                            or result.run_id != run.id
                        ):
                            raise ValueError(
                                "retained arena identity differs from frozen plan/producer"
                            )
                        if result.protocol != plan.protocol(pool, anchor, update):
                            raise ValueError(
                                "evaluation cohort differs from planned deals/roles/inference"
                            )
                        if result.coordinates.updates != (
                            0 if variant == "control" else update
                        ):
                            raise ValueError(
                                "evaluation milestone differs from planned candidate/control"
                            )
                        checked = result.model_copy(deep=True)
                        checked.validate_cohort()
                        if (
                            result.status != "completed"
                            or checked.status != "completed"
                        ):
                            missing.append(
                                f"{pool}/{seed}/{update}/{anchor}/{variant}: {result.status}"
                            )
                            continue
                        retained.append(result)
                    if len(retained) == len(variants):
                        effects[pool, anchor, seed, update] = paired_improvement(
                            retained[0],
                            retained[-1],
                            initialization=run.stages[0].artifacts["initial_raw"],
                        )
    if len({r.identities.get("source_commit") for r in runs.values()}) > 1:
        raise ValueError("report mixes scientific source freezes")
    return Evidence(
        root,
        plan,
        anchors,
        runs,
        effects,
        tuple(missing),
        hashes,
        evaluation_seconds,
        games,
    )


def curves(evidence: Evidence, clock: str) -> Figure:
    columns = len(evidence.anchors)
    figure = Figure(figsize=(6.5 * columns, 9), layout="constrained")
    axes = figure.subplots(3, columns, squeeze=False)
    for row, pool in enumerate(POOLS):
        for col, anchor in enumerate(evidence.anchors):
            axis = axes[row, col]
            for seed in SEEDS:
                points = [
                    evidence.effects[pool, anchor, seed, n]
                    for n in (0, evidence.plan.updates // 2, evidence.plan.updates)
                    if (pool, anchor, seed, n) in evidence.effects
                ]
                points = [
                    p for p in points if getattr(p.coordinates, clock) is not None
                ]
                if points:
                    axis.errorbar(
                        [getattr(p.coordinates, clock) for p in points],
                        [100 * p.improvement.mean for p in points],
                        yerr=[
                            [
                                100 * (p.improvement.mean - p.improvement.lower)
                                for p in points
                            ],
                            [
                                100 * (p.improvement.upper - p.improvement.mean)
                                for p in points
                            ],
                        ],
                        marker="o",
                        label=str(seed),
                        linewidth=1,
                    )
            axis.axhline(0, color="#777", linewidth=0.6)
            axis.set(
                title=f"{pool.upper()} · {anchor}",
                xlabel=clock,
                ylabel="gain vs same-deal initialization (points)",
            )
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No paired results yet",
                    transform=axis.transAxes,
                    ha="center",
                )
    return figure


def _svg(figure: Figure) -> str:
    stream = StringIO()
    figure.savefig(stream, format="svg")
    return stream.getvalue()[stream.getvalue().index("<svg") :]


def write_report(root: Path, output: Path, *, notes: str = "") -> Path:
    evidence = load(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, object]] = []
    lines = [
        "<h1>Learning across environment complexity</h1>",
        f"<p>Saved evidence only · rendered {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>",
        f"<p>{sum(r.status == 'completed' for r in evidence.runs.values())}/9 fits complete; "
        f"{evidence.evaluated_games}/{18000 * len(evidence.anchors)} attempted evaluation games; "
        f"{len(evidence.unavailable)} unavailable components. Incomplete cohorts are not zero strength.</p>",
        f"<p>{escape(notes)}</p>",
        "<p>Each pool has its own controls. Raw scores across pools are not a difficulty ranking. "
        "Curves show conditional whole-deal intervals; method uncertainty uses three training seeds separately.</p>",
    ]
    for pool in POOLS:
        for anchor in evidence.anchors:
            endpoint = tuple(
                evidence.effects[pool, anchor, seed, evidence.plan.updates]
                for seed in SEEDS
                if (pool, anchor, seed, evidence.plan.updates) in evidence.effects
            )
            summary = three_seed_summary(endpoint, SEEDS)
            panels.append(
                {
                    "pool": pool,
                    "anchor": anchor,
                    "endpoint": TypeAdapter(CohortEffect).dump_python(
                        summary, mode="json"
                    ),
                }
            )
            interval = summary.improvement
            text = (
                f"{100 * interval.mean:+.2f} [{100 * interval.lower:+.2f}, {100 * interval.upper:+.2f}] points"
                if interval
                else "unavailable: not all three seed cohorts complete"
            )
            lines.append(
                f"<h2>{pool.upper()} · {anchor}</h2>"
                + (
                    "<p>Supplementary anchor, scored after the frozen results on "
                    "the same deals, legs and frozen controls.</p>"
                    if anchor not in FROZEN_ANCHORS
                    else ""
                )
                + f"<p>Endpoint gain: {text}. "
                "95% t interval, df=2; assumption-sensitive.</p>"
            )
            lines.append(
                "<table><tr><th>Seed</th><th>Model role / anchor role</th><th>Candidate</th><th>Control</th><th>Gain [conditional 95% CI]</th></tr>"
            )
            for point in endpoint:
                for cell in point.cells:
                    lines.append(
                        f"<tr><td>{point.seed}</td><td>{escape(cell.candidate_role)} / {escape(cell.opponent_role)}</td>"
                        f"<td>{100 * cell.candidate_score:.1f}%</td><td>{100 * cell.control_score:.1f}%</td>"
                        f"<td>{100 * cell.improvement.mean:+.1f} [{100 * cell.improvement.lower:+.1f}, {100 * cell.improvement.upper:+.1f}]</td></tr>"
                    )
            lines.append("</table>")
    for clock in (
        "updates",
        "learner_transitions",
        "optimizer_exposures",
        "active_training_seconds",
        "training_seconds",
    ):
        lines.extend([f"<h2>{clock}</h2>", _svg(curves(evidence, clock))])
    progress = {}
    for pool in POOLS:
        for seed in SEEDS:
            points = tuple(
                evidence.effects[pool, "scripted_greedy", seed, n]
                for n in (0, evidence.plan.updates // 2, evidence.plan.updates)
                if (pool, "scripted_greedy", seed, n) in evidence.effects
            )
            progress[f"{pool}-{seed}"] = asdict(
                observed_progress(
                    points, (0, evidence.plan.updates // 2, evidence.plan.updates)
                )
            )
    lines.append(
        "<h2>Ten-point greedy progress, confirmed at endpoint</h2><pre>"
        + escape(json.dumps(progress, indent=2))
        + "</pre>"
    )
    costs = {
        "retained_training_executor_seconds": sum(
            r.seconds for r in evidence.runs.values()
        ),
        "retained_evaluation_seconds": evidence.evaluation_seconds,
        "inclusive_all_attempts_cost": "pending launcher settlement including failed setup, calibration, transfer and reporting",
    }
    lines.append(
        "<h2>Costs and missing evidence</h2><pre>"
        + escape(json.dumps(costs, indent=2))
        + "</pre>"
    )
    lines.append(
        "<details><summary>Unavailable components</summary><pre>"
        + escape("\n".join(evidence.unavailable))
        + "</pre></details>"
    )
    lines.append(
        "<details><summary>Evidence hashes</summary><pre>"
        + escape(json.dumps(evidence.hashes, indent=2))
        + "</pre></details>"
    )
    output.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + "<title>MTG-122 learning comparison</title><style>"
        + DASHBOARD_CSS
        + "</style><main>"
        + "\n".join(lines)
        + "</main></html>"
    )
    output.with_suffix(".analysis.json").write_text(
        json.dumps(
            {
                "panels": panels,
                "progress": progress,
                "costs": costs,
                "unavailable": evidence.unavailable,
                "hashes": evidence.hashes,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def notebook(root: Path, output: Path) -> Path:
    """Create once: edits survive every report refresh, including failed runs."""
    import nbformat

    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(root.resolve(), output.parent.resolve())
    book = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# MTG-122 report generator\nRun All reads retained evidence only. Edit notes/plots here; refreshing never overwrites this notebook."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nfrom experiments.runners.complexity_report import write_report\n"
                + f"ROOT = Path({relative!r}).resolve()\nNOTES = 'No strength conclusion until all declared cohorts complete.'\n"
            ),
            nbformat.v4.new_code_cell(
                "write_report(ROOT, Path('comparison.html'), notes=NOTES)"
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    try:
        with output.open("x") as stream:
            nbformat.write(book, stream)
    except FileExistsError:
        pass
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = notebook(args.root, args.out / "comparison.ipynb")
    # Execute a copy: preserve authored notebook bytes and all edits.
    from nbclient import NotebookClient
    import nbformat

    NotebookClient(
        nbformat.read(path, as_version=4),
        timeout=300,
        resources={"metadata": {"path": str(path.parent.resolve())}},
    ).execute()


if __name__ == "__main__":
    main()
