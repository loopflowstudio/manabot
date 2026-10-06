"""Generate an editable saved-history report plan and publish native W&B graphs.

The notebook only reads retained evidence and writes a plan/HTML viewer. Publishing
is an explicit CLI step. Scientific score streams have content-bound identities;
existing training and monitoring histories are referenced, never rewritten.
"""

import argparse
import html
import math
from pathlib import Path
from typing import Literal

import nbformat
from pydantic import BaseModel, ConfigDict
import wandb_workspaces.reports.v2 as wr

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.execution import atomic_json
from manabot.training.monitoring import Dashboard, publish_dashboard
from manabot.training.recovery import attempt_lock
from manabot.training.report_study import load_study, study_strength_panels


class Plot(BaseModel):
    title: str
    x: str
    y: list[str]
    smoothing: float = 0
    bounds: tuple[float | None, float | None] = (None, None)


class Section(BaseModel):
    title: str
    description: str
    run_ids: list[str]
    plots: list[Plot]
    collapsed: bool = False


class ReportPlan(BaseModel):
    schema_version: Literal[1] = 1
    title: str
    introduction: str
    entity: str = "loopflow-studio"
    project: str = "etude"
    sources: dict[str, str]
    projections: list[Dashboard]
    sections: list[Section]
    evidence_notes: list[str]


class ContrastPoint(BaseModel):
    arm_seed_scores: list[list[float]]
    mean_effect: float
    paired_seed_common_deal_95_percentile: tuple[float, float]


class Contrast(BaseModel):
    model_config = ConfigDict(extra="ignore")
    arms: list[str]
    seeds: list[int]
    checkpoints: list[ContrastPoint]


class SupervisorCosts(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    cumulative_seconds: float
    prior_seconds: float


def build_plan(root: Path) -> ReportPlan:
    study_path = root / "scientific/study/study.json"
    contrast_path = study_path.with_name("history-contrast.json")
    study = load_study(study_path)
    panels = study_strength_panels(study)
    if len(panels) != 1:
        raise ValueError("history report requires one complete scientific cohort")
    panel = panels[0]
    if (
        panel.phase != "development"
        or panel.variant != "raw"
        or len(study.seeds) != 3
        or any(c.scheduled_games != 100 for c in panel.cells)
    ):
        raise ValueError("expected the retained three-seed raw development cohort")
    contrast = Contrast.model_validate_json(contrast_path.read_text())
    cutoffs = sorted({m.cutoff for m in panel.measurements})
    if len(cutoffs) != 2 or any(m.learner_transitions == 0 for m in panel.measurements):
        raise ValueError("expected two retained non-initial checkpoints")
    if contrast.seeds != study.seeds or len(cutoffs) != len(contrast.checkpoints):
        raise ValueError("contrast does not match the admitted cohort")
    if contrast.arms != ["history-off", "history-on"]:
        raise ValueError("expected the retained off/on history contrast")
    if {m.regime for m in panel.measurements} != set(contrast.arms):
        raise ValueError("scientific cohort has different arms from the contrast")
    sources = {str(p.resolve()): file_sha256(p) for p in (study_path, contrast_path)}
    identity = canonical_sha256(sources)[:16]
    projections: list[Dashboard] = []
    sections: list[Section] = []
    for arm in contrast.arms:
        ids: list[str] = []
        for seed in study.seeds:
            points = sorted(
                (m for m in panel.measurements if (m.regime, m.seed) == (arm, seed)),
                key=lambda m: m.cutoff,
            )
            run_id = f"history-score-{identity}-{arm}-{seed}"
            ids.append(run_id)
            projections.append(
                Dashboard(
                    run_id=run_id,
                    config={
                        "training_run_id": f"{arm} · seed {seed} · scientific scores",
                        "regime_digest": identity,
                        "seed": seed,
                        "arm": arm,
                        "purpose": "saved-scientific-evaluation",
                        "sources": sources,
                    },
                    summary={
                        "phase": panel.phase,
                        "variant": panel.variant,
                        "opponent": panel.opponent,
                        "initialization": "unavailable",
                        "metric_limits": "Retained development cohort; three training seeds, one anchor. No new evaluation.",
                    },
                    rows=[
                        {
                            "progress/learner_transitions": m.learner_transitions,
                            "progress/observation": i,
                            "progress/training_seconds": m.training_seconds,
                            "scientific/score": float(m.score),
                            "scientific/cutoff": m.cutoff,
                        }
                        for i, m in enumerate(points)
                        if m.score is not None
                    ],
                )
            )
        sections.append(
            Section(
                title=f"{arm}: checkpoint playing strength",
                description="Each line is one independent training seed. Score = win + half draw; 100 games per point. Both retained checkpoints are development evaluations. Hover to read exact values.",
                run_ids=ids,
                plots=[
                    Plot(
                        title=f"{arm} · three seeds",
                        x="progress/learner_transitions",
                        y=["scientific/score"],
                        bounds=(0, 1),
                    ),
                    Plot(
                        title=f"{arm} · recorded cost (not cost matched)",
                        x="progress/training_seconds",
                        y=["scientific/score"],
                        bounds=(0, 1),
                    ),
                ],
            )
        )
    effect_rows: list[dict[str, int | float | str | bool]] = []
    for cutoff, point in zip(cutoffs, contrast.checkpoints, strict=True):
        expected_effect = sum(
            b - a for a, b in zip(*point.arm_seed_scores, strict=True)
        ) / len(study.seeds)
        if not math.isclose(point.mean_effect, expected_effect, abs_tol=1e-12):
            raise ValueError("contrast mean differs from paired seed scores")
        coordinates = {
            m.learner_transitions for m in panel.measurements if m.cutoff == cutoff
        }
        if len(coordinates) != 1:
            raise ValueError("scientific cutoff has unequal transition coordinates")
        for arm_index, arm in enumerate(contrast.arms):
            observed = [
                next(
                    m.score
                    for m in panel.measurements
                    if (m.regime, m.seed, m.cutoff) == (arm, seed, cutoff)
                )
                for seed in study.seeds
            ]
            if observed != point.arm_seed_scores[arm_index]:
                raise ValueError("contrast scores differ from retained games")
        effect_rows.append(
            {
                "progress/observation": len(effect_rows),
                "progress/learner_transitions": coordinates.pop(),
                "effect/mean_points": 100 * point.mean_effect,
                "effect/lower95_points": 100
                * point.paired_seed_common_deal_95_percentile[0],
                "effect/upper95_points": 100
                * point.paired_seed_common_deal_95_percentile[1],
                "effect/zero": 0,
            }
        )
    effect_id = f"history-effect-{identity}"
    projections.append(
        Dashboard(
            run_id=effect_id,
            config={
                "training_run_id": "History on − off · paired scientific effect",
                "regime_digest": identity,
                "purpose": "saved-scientific-evaluation",
                "sources": sources,
            },
            summary={
                "metric_limits": "Saved paired training-seed/common-deal percentile intervals; not W&B standard error."
            },
            rows=effect_rows,
        )
    )
    sections.append(
        Section(
            title="Does history help?",
            description="Mean on−off effect and explicit lower/upper 95% bounds, copied from the saved paired seed/common-deal analysis. These are exploratory intervals from three seeds, not W&B-generated uncertainty. Zero lies inside both intervals; neither superiority nor equivalence is established.",
            run_ids=[effect_id],
            plots=[
                Plot(
                    title="History on − off · percentage points",
                    x="progress/learner_transitions",
                    y=[
                        "effect/mean_points",
                        "effect/lower95_points",
                        "effect/upper95_points",
                        "effect/zero",
                    ],
                )
            ],
        )
    )
    entropy: list[float] = []
    empty = 0
    for arm in contrast.arms:
        ids = []
        for seed in study.seeds:
            path = root / "wandb" / f"{arm}-seed-{seed}" / "dashboard.json"
            saved = Dashboard.model_validate_json(path.read_text())
            entropy.extend(
                float(r["rl/entropy"]) for r in saved.rows if "rl/entropy" in r
            )
            empty += sum(r.get("rl/retained") == 0 for r in saved.rows)
            ids.append(saved.run_id)
            sources[str(path.resolve())] = file_sha256(path)
        sections.append(
            Section(
                title=f"{arm}: learning diagnostics",
                collapsed=True,
                description="EMA smoothing 0.8 with original curves available. Loss/entropy/KL sample the last optimized timestep minibatch, not the whole update or fixed positions. Lines can bridge missing samples; zero retention marks empty-filter updates. No historical averages are reconstructed. Entropy is not competence.",
                run_ids=ids,
                plots=[
                    Plot(
                        title=title + " · EMA 0.8",
                        x="progress/updates",
                        y=[metric],
                        smoothing=0.8,
                    )
                    for title, metric in (
                        ("Policy objective (includes KL)", "rl/policy_loss"),
                        ("Value loss", "rl/value_loss"),
                        ("Entropy (nats)", "rl/entropy"),
                        ("Collection KL (unweighted)", "rl/collection_kl"),
                        ("Reference KL (unweighted)", "rl/reference_kl"),
                    )
                ]
                + [
                    Plot(
                        title="Selected / collected rows · raw",
                        x="progress/updates",
                        y=["rl/retained_fraction"],
                        bounds=(0, 1),
                    )
                ],
            )
        )
        monitor_path = root / "wandb" / f"{arm}-monitoring" / "dashboard.json"
        monitor = Dashboard.model_validate_json(monitor_path.read_text())
        sources[str(monitor_path.resolve())] = file_sha256(monitor_path)
        sections.append(
            Section(
                title=f"{arm}: earlier monitoring",
                collapsed=True,
                description="One training seed, separate deals/protocol and observation ABI. Bounds resample deals for a checkpoint; they are not method uncertainty. Do not pool with the scientific cohort or compare these arms as matched runs.",
                run_ids=[monitor.run_id],
                plots=[
                    Plot(
                        title="Monitoring score and deal interval",
                        x="progress/training_seconds",
                        y=[
                            "monitor/score/mean",
                            "monitor/score/lower",
                            "monitor/score/upper",
                        ],
                        bounds=(0, 1),
                    )
                ],
            )
        )
    endpoint = effect_rows[-1]
    supervisor_path = root / "scientific/supervisor.json"
    supervisor = SupervisorCosts.model_validate_json(supervisor_path.read_text())
    sources[str(supervisor_path.resolve())] = file_sha256(supervisor_path)
    near_binary = sum(abs(value - math.log(2)) < 0.001 for value in entropy)
    return ReportPlan(
        title="History input — learning and evidence",
        introduction=f"Completed saved study · 3 training seeds · 1,200 replayed scientific games. Endpoint history effect: {endpoint['effect/mean_points']:+.2f} points [{endpoint['effect/lower95_points']:+.2f}, {endpoint['effect/upper95_points']:+.2f}]. Unresolved. Initialization was not evaluated; these two checkpoints do not demonstrate improvement from initialization.",
        sources=sources,
        projections=projections,
        sections=sections,
        evidence_notes=[
            f"Saved-data snapshot; no live execution heartbeat. Supervisor cost: {supervisor.cumulative_seconds:,.2f} seconds, including {supervisor.prior_seconds:,.2f} prior failed-calibration seconds. Evaluation is included, not an additional charge. Fixed-update endpoints are not matched-cost effects. Models and artifacts live in S3; W&B training-run summaries contain their versioned references.",
            f"{near_binary:,} of {len(entropy):,} saved entropy values lie within .001 nat of ln(2); {empty} updates had empty filters. Nearly uniform binary choices remain a hypothesis: minibatch support sizes and probabilities were not retained. Pipeline/replay smoke is not a positive learning control.",
        ],
    )


def write_viewer(plan: ReportPlan, output: Path, url: str | None = None) -> None:
    content = "<p>Publish the saved report plan to load W&B graphs here.</p>"
    if url:
        safe = html.escape(url, quote=True)
        content = f'<p><a href="{safe}" target="_blank" rel="noopener">Open the interactive W&B report</a> · Sign in to W&B if prompted.</p><iframe title="W&B experiment report" src="{safe}?jupyter=true" style="width:100%;height:1600px;border:0"></iframe>'
    output.write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(plan.title)}</title><style>body{{font:16px/1.5 system-ui;margin:30px auto;max-width:1400px;padding:0 24px;color:#23343b}}a{{color:#146a76}}</style><h1>{html.escape(plan.title)}</h1>{content}</html>'
    )


def write_notebook(root: Path, output: Path) -> None:
    """Create once. Notebook edits and all previous reports remain untouched."""
    if output.exists():
        return
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# W&B history report generator\nRun All reads saved evidence and writes a report plan and HTML viewer. It never trains, evaluates or uploads. Edit the plan below; publish explicitly with the runner's --publish option."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nimport json\nfrom experiments.runners.report_history_wandb import build_plan, write_viewer\n"
                + f"ROOT = Path({str(root.resolve())!r})\nplan = build_plan(ROOT)\n"
            ),
            nbformat.v4.new_code_cell(
                "# Editable report choices: titles, sections, metrics, smoothing and axes.\n# Example: plan.sections[0].plots[0].bounds = (0, 0.6)\nPath('report-plan.json').write_text(plan.model_dump_json(indent=2))\nreceipt = Path('report-url.json')\nurl = json.loads(receipt.read_text())['url'] if receipt.exists() else None\nwrite_viewer(plan, Path('comparison.html'), url)\nprint('Read-only viewer: comparison.html')"
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
    with output.open("x") as stream:
        nbformat.write(notebook, stream)


def publish_plan(plan: ReportPlan, output: Path) -> str:
    for source, digest in plan.sources.items():
        if file_sha256(Path(source)) != digest:
            raise ValueError(f"report source changed: {source}")
    for dashboard in plan.projections:
        directory = output / dashboard.run_id
        directory.mkdir(exist_ok=True)
        publish_dashboard(
            dashboard,
            directory,
            entity=plan.entity,
            project=plan.project,
            job_type="scientific-evaluation",
        )
    blocks: list[wr.P | wr.H2 | wr.PanelGrid | wr.MarkdownBlock] = [
        wr.P(plan.introduction)
    ]
    for section in plan.sections:
        plots = [
            wr.LinePlot(
                title=p.title,
                x=p.x,
                y=p.y,
                range_y=p.bounds,
                smoothing_type="exponential" if p.smoothing else "none",
                smoothing_factor=p.smoothing,
                smoothing_show_original=True,
                aggregate=False,
                groupby_rangefunc="none",
                ignore_outliers=False,
                max_runs_to_show=len(section.run_ids),
                line_marks={
                    metric: "dashed"
                    if "95" in metric
                    else "dotted"
                    if metric.endswith("/zero")
                    else "solid"
                    for metric in p.y
                },
                legend_position="south",
                layout=wr.Layout(
                    x=(i % 2) * 12,
                    y=(i // 2) * 8,
                    w=12 if len(section.plots) > 1 else 24,
                    h=8,
                ),
            )
            for i, p in enumerate(section.plots)
        ]
        grid = wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=plan.entity,
                    project=plan.project,
                    name=section.title,
                    filters=f"ID in {section.run_ids!r}",
                )
            ],
            panels=plots,
        )
        body = [wr.P(section.description), grid]
        blocks.extend(
            [wr.H2(section.title, collapsed_blocks=body)]
            if section.collapsed
            else [wr.H2(section.title), *body]
        )
    blocks.extend(
        [
            wr.H2("Evidence and costs"),
            *[wr.P(note) for note in plan.evidence_notes],
            wr.MarkdownBlock(
                "[Metric definitions and sampling limits](https://github.com/loopflowstudio/etude/blob/jack/etu-117-make-experiment-reports-reveal-learning-trends-and-evidence/docs/experiment-metrics.md) · [Project and raw run histories](https://wandb.ai/loopflow-studio/etude)"
            ),
        ]
    )
    report = wr.Report(
        entity=plan.entity,
        project=plan.project,
        title=plan.title,
        description="Saved history experiment; scientific scores, diagnostics and monitoring kept separate.",
        blocks=blocks,
        width="fluid",
    )
    report.save()
    atomic_json(
        output / "report-url.json",
        {
            "url": report.url,
            "id": report.id,
            "plan_sha256": canonical_sha256(plan.model_dump(mode="json")),
        },
    )
    write_viewer(plan, output / "comparison.html", report.url)
    return report.url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish an existing report-plan.json; never overwrite notebook edits",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.publish:
        plan = ReportPlan.model_validate_json(
            (args.output / "report-plan.json").read_text()
        )
        with attempt_lock(args.output / "publisher.lock"):
            print(publish_plan(plan, args.output))
    else:
        write_notebook(args.root, args.output / "report-generator.ipynb")
        print(args.output / "report-generator.ipynb")


if __name__ == "__main__":
    main()
