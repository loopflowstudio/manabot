"""Read retained experiment evidence and render an offline experiment dashboard.

ReportEvidence owns no execution state. Notebook cells select evidence and build
figures; write_dashboard only renders those choices. Matching requires every
retained run in a compatible cohort at the same stage/update coordinate.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
from io import StringIO
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Literal

from matplotlib.figure import Figure
from pydantic import ConfigDict, JsonValue

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.experiment_execution import ExperimentRun
from manabot.training.learning_dashboard import DASHBOARD_CSS, learning_story
from manabot.training.models import TrainingRegime, TrainingRun
from manabot.training.monitor_evaluation import MonitorResult
from manabot.training.monitoring import Scalar, training_dashboard
from manabot.training.report_study import ScientificStudy


class RetainedRegime(TrainingRegime):
    """Read-only report admission preserves additional frozen recipe metadata.

    This projection is never passed to execution. Known fields still validate;
    later recovery/configuration extensions survive raw exports unchanged.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue]


class RetainedRun(TrainingRun):
    """Reporting projection, not admission for training or checkpoint serving."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue]
    regime: RetainedRegime


class RetainedExecution(ExperimentRun):
    """Preserve newer lifecycle metadata without giving the report execution authority."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue]


class RetainedMonitor(MonitorResult):
    """Keep producer timestamps and extension metadata on immutable monitor exports."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue]
    started_unix: float | None = None
    finished_unix: float | None = None


@dataclass(frozen=True)
class ReportEvidence:
    root: Path
    runs: tuple[TrainingRun, ...]
    monitors: tuple[MonitorResult, ...]
    executions: tuple[ExperimentRun, ...]
    failed_directories: tuple[Path, ...]
    paths: tuple[Path, ...]

    def label(self, run_id: str) -> str:
        return next(
            (f"{r.regime.id} · seed {r.seed}" for r in self.runs if r.id == run_id),
            run_id,
        )

    def metrics(self, run: TrainingRun) -> list[dict[str, Scalar]]:
        """Expose all saved diagnostics for notebook-selected deeper analysis."""
        rows = training_dashboard(run).rows
        diagnostics = {
            (s.id, i): d for s in run.stages for i, d in enumerate(s.diagnostics)
        }
        for row in rows:
            diagnostic = diagnostics.get(
                (row.get("stage/id"), row.get("stage/ordinal")), {}
            )
            for key, value in diagnostic.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row.setdefault(f"diagnostic/{key}", value)
        return rows


def load_evidence(root: Path) -> ReportEvidence:
    """Read saved JSON only; malformed records fail instead of disappearing."""
    root = root.resolve()
    runs = tuple(sorted(root.rglob("run.json")))
    monitors = tuple(sorted(root.rglob("monitor.json")))
    executions = tuple(sorted(root.rglob("experiment.json")))
    attempts = tuple(sorted(root.rglob("attempt-*/attempt.json")))
    failed = tuple(
        p.parent
        for p in attempts
        if json.loads(p.read_text())["status"] in {"failed", "interrupted", "running"}
    )
    return ReportEvidence(
        root,
        tuple(RetainedRun.model_validate_json(p.read_text()) for p in runs),
        tuple(
            RetainedMonitor.model_validate_json(p.read_text())
            for p in monitors
            if not any(parent in failed for parent in p.parents)
        ),
        tuple(RetainedExecution.model_validate_json(p.read_text()) for p in executions),
        failed,
        (*runs, *monitors, *executions, *attempts, *tuple(root.glob("snapshot.json"))),
    )


def compatible_panels(evidence: ReportEvidence) -> dict[str, list[MonitorResult]]:
    """No pooling across protocol, opponent, world, ABI or inference envelopes."""
    panels: dict[str, list[MonitorResult]] = {}
    for result in evidence.monitors:
        if result.status != "completed" or result.win is None:
            continue
        identity = canonical_sha256(
            {
                "protocol": result.protocol.model_dump(mode="json"),
                "opponent": result.opponent.model_dump(mode="json"),
                "arena": result.key.model_dump(mode="json"),
            }
        )
        panels.setdefault(identity, []).append(result)
    return panels


def matched_milestones(evidence: ReportEvidence) -> dict[str, list[MonitorResult]]:
    """Intersect stage/update coordinates across all expected runs, without interpolation."""
    expected = {r.id for r in evidence.runs}
    if any(
        a.status in {"pending", "running", "failed", "interrupted"}
        and a.run_id not in expected
        for e in evidence.executions
        for a in e.attempts
    ):
        return {}
    matched: dict[str, list[MonitorResult]] = {}
    for panel, results in compatible_panels(evidence).items():
        by_coordinate: dict[tuple[str, int], list[MonitorResult]] = {}
        for result in results:
            key = (result.coordinates.stage_id, result.coordinates.updates)
            by_coordinate.setdefault(key, []).append(result)
        rows = [
            r
            for group in by_coordinate.values()
            if {r.run_id for r in group} == expected and len(group) == len(expected)
            for r in group
        ]
        if rows:
            matched[panel] = rows
    return matched


def strength_figures(
    evidence: ReportEvidence,
    axis: Literal["training_seconds", "environment_decisions"],
    *,
    per_run: bool = False,
) -> list[Figure]:
    """Plot matched milestones per seed; bands are saved deal uncertainty, not seed CIs."""
    figures: list[Figure] = []
    panels = matched_milestones(evidence)
    if per_run:
        # Explicit within-run histories do not claim compatibility across variants.
        panels = {
            f"{panel}/{run.id}": results
            for run in evidence.runs
            for panel, results in matched_milestones(
                replace(
                    evidence,
                    runs=(run,),
                    executions=(),
                    monitors=tuple(m for m in evidence.monitors if m.run_id == run.id),
                )
            ).items()
        }
    for panel, results in panels.items():
        fig = Figure(figsize=(9, 3.5), layout="constrained")
        ax = fig.subplots()
        for run_id in sorted({r.run_id for r in results}):
            rows = sorted(
                (r for r in results if r.run_id == run_id),
                key=lambda r: getattr(r.coordinates, axis),
            )
            x = [getattr(r.coordinates, axis) for r in rows]
            rates = [r.win for r in rows if r.win is not None]
            ax.errorbar(
                x,
                [r.mean for r in rates],
                yerr=(
                    [r.mean - r.lower for r in rates],
                    [r.upper - r.mean for r in rates],
                ),
                marker="o",
                capsize=4,
                label=evidence.label(run_id),
            )
        ax.set(
            xlabel="Recorded training seconds"
            if axis == "training_seconds"
            else "Native environment decisions (work proxy)",
            ylabel="Win fraction",
            ylim=(-0.03, 1.03),
            title=(
                "Within-run monitoring history"
                if per_run
                else "Matched stage/update milestones"
            )
            + f" · cohort {panel[:8]}",
        )
        ax.legend(fontsize=8)
        figures.append(fig)
    return figures


def metric_figure(evidence: ReportEvidence, metric: str) -> Figure:
    """Any saved scalar is available; missing coordinates are never fabricated."""
    fig = Figure(figsize=(9, 3), layout="constrained")
    ax = fig.subplots()
    for run in evidence.runs:
        points = [
            (row["progress/training_seconds"], row[metric])
            for row in evidence.metrics(run)
            if isinstance(row.get(metric), (float, int))
            and "progress/training_seconds" in row
        ]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker=".", label=evidence.label(run.id))
    ax.set(xlabel="Recorded training seconds", ylabel=metric)
    if ax.lines:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Unavailable in retained records",
            ha="center",
            transform=ax.transAxes,
        )
    return fig


def numerical_figures(evidence: ReportEvidence) -> list[Figure]:
    """Compact health views. Missing cadence samples remain gaps, never zeros."""
    groups = (
        ("Collection concentration", ("legal_logit_gap_max",)),
        ("Rounded legal zeros", ("legal_underflow_count",)),
        ("Gradient norm", ("gradient_norm_before", "gradient_norm_after")),
        ("Measured parameter movement", ("parameter_delta_l2",)),
        ("Gradient tensors", ("gradient_missing_tensors", "gradient_zero_tensors")),
        (
            "Optimizer steps",
            ("optimizer_steps", "skipped_steps", "rejected_steps", "rejected_updates"),
        ),
    )
    figures: list[Figure] = []
    for run in evidence.runs:
        rows = evidence.metrics(run)
        if not any(any(key.startswith("numerical/") for key in row) for row in rows):
            continue
        fig = Figure(figsize=(9, 8), layout="constrained")
        for ax, (title, names) in zip(fig.subplots(3, 2).flat, groups, strict=True):
            for name in names:
                points = [
                    (r.get("progress/updates", i), r.get(f"numerical/{name}"))
                    for i, r in enumerate(rows)
                ]
                ax.plot(
                    [x for x, _ in points],
                    [
                        float(y) if isinstance(y, (float, int)) else float("nan")
                        for _, y in points
                    ],
                    marker=".",
                    label=name.replace("_", " "),
                )
            ax.set_title(title)
            ax.set_xlabel("Completed updates")
            ax.legend(fontsize=7)
        fig.suptitle(f"{run.regime.id} / seed {run.seed}: numerical health")
        figures.append(fig)
    return figures


def diagnostic_figures(evidence: ReportEvidence, *, window: int = 25) -> list[Figure]:
    """One regime per figure; trailing means reset at missing values/stage boundaries.

    A window counts retained diagnostic records, not games or minibatch samples.
    Faint raw lines remain visible; no smoothing across unavailable observations.
    """
    if window < 1:
        raise ValueError("smoothing window must be positive")
    metrics = (
        ("rl/policy_loss", "Policy objective (Ataraxos includes KL penalties)"),
        ("rl/value_loss", "Value loss (recipe-specific units)"),
        ("rl/entropy", "Last optimized minibatch entropy (nats)"),
        ("rl/collection_kl", "Collection reverse KL (nats, unweighted)"),
        ("rl/reference_kl", "Reference reverse KL (nats, unweighted)"),
        ("rl/retained_fraction", "Filter retention (selected / collected rows)"),
    )
    figures: list[Figure] = []
    saved = {r.id: evidence.metrics(r) for r in evidence.runs}
    for regime in sorted({r.regime.id for r in evidence.runs}):
        fig = Figure(figsize=(9, 10), layout="constrained")
        axes = fig.subplots(3, 2).flat
        for ax, (metric, title) in zip(axes, metrics, strict=True):
            for run in (r for r in evidence.runs if r.regime.id == regime):
                rows = saved[run.id]
                x: list[float] = []
                raw: list[float] = []
                smooth: list[float] = []
                history: list[float] = []
                previous_stage: Scalar | None = None
                previous_update: int | float | None = None
                for row in rows:
                    stage = row.get("stage/id")
                    if stage != previous_stage:
                        history.clear()
                    previous_stage = stage
                    update = row.get("progress/updates")
                    if isinstance(update, (int, float)):
                        if previous_update is not None and update > previous_update + 1:
                            history.clear()
                            x.append(math.nan)
                            raw.append(math.nan)
                            smooth.append(math.nan)
                        previous_update = update
                    time = row.get("progress/training_seconds")
                    value = row.get(metric)
                    if not isinstance(time, (int, float)):
                        history.clear()
                        x.append(math.nan)
                        raw.append(math.nan)
                        smooth.append(math.nan)
                        continue
                    x.append(float(time))
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        history.clear()
                        raw.append(math.nan)
                        smooth.append(math.nan)
                    else:
                        raw.append(float(value))
                        history.append(float(value))
                        history = history[-window:]
                        smooth.append(mean(history))
                (line,) = ax.plot(x, smooth, label=f"seed {run.seed}")
                ax.plot(x, raw, color=line.get_color(), alpha=0.16, linewidth=0.6)
            if not any(math.isfinite(y) for line in ax.lines for y in line.get_ydata()):
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable in retained records",
                    ha="center",
                    transform=ax.transAxes,
                )
            if metric == "rl/entropy":
                ax.axhline(
                    math.log(2),
                    color="gray",
                    linestyle=":",
                    label="ln(2), not competence",
                )
            ax.set(title=title, xlabel="Recorded training seconds")
            ax.legend(fontsize=7)
        fig.suptitle(
            f"{regime} · trailing mean ≤{window} records / faint raw · gaps stay missing"
        )
        figures.append(fig)
    return figures


def _sampling_rows(evidence: ReportEvidence) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in evidence.runs:
        diagnostics = [d for stage in run.stages for d in stage.diagnostics]
        entropy = [
            float(d["entropy"])
            for d in diagnostics
            if isinstance(d.get("entropy"), (int, float))
        ]
        counts = [d for d in diagnostics if "rows" in d and "retained" in d]
        collected = sum(float(d["rows"]) for d in counts)
        retained = sum(float(d["retained"]) for d in counts)
        rows.append(
            [
                evidence.label(run.id),
                str(len(diagnostics)),
                str(sum(bool(d.get("skipped")) for d in diagnostics)),
                str(sum("loss" not in d for d in diagnostics)),
                f"{retained:,.0f} / {collected:,.0f} ({retained / collected:.1%}); {len(counts)} records"
                if collected
                else "unavailable",
                f"{sum(abs(e - math.log(2)) < 0.001 for e in entropy)} / {len(entropy)}"
                if entropy
                else "unavailable",
            ]
        )
    return rows


def _number(value: object) -> str:
    return f"{value:,.3g}" if isinstance(value, (float, int)) else "unavailable"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _svg(figure: Figure) -> str:
    stream = StringIO()
    figure.savefig(stream, format="svg")
    return stream.getvalue()[stream.getvalue().index("<svg") :]


def write_dashboard(
    evidence: ReportEvidence,
    output: Path,
    *,
    question: str,
    docs: str,
    sections: list[tuple[str, str, list[Figure]]],
    notes: str = "",
    study: ScientificStudy | None = None,
    notebook: Path | None = None,
) -> Path:
    """Render notebook-selected figures and a concise status summary; never mutate inputs."""
    now = datetime.now(timezone.utc)
    output.parent.mkdir(parents=True, exist_ok=True)
    guide = output.parent / docs
    if not guide.is_file():
        guide = Path(__file__).resolve().parents[2] / "docs/experiment-metrics.md"
    if guide.exists():
        from nbconvert.filters import markdown2html

        guide_path = output.with_name(output.stem + "-metrics.html")
        guide_html = str(markdown2html(guide.read_text()))
        guide_html = re.sub(
            r'(id|href)="(#?)([^" ]+)"',
            lambda m: (
                f'{m[1]}="{m[2]}{m[3].lower()}"' if m[1] == "id" or m[2] else m[0]
            ),
            guide_html,
        )
        guide_html = guide_html.replace(
            'href="evidence/',
            'href="https://github.com/loopflowstudio/etude/blob/main/docs/evidence/',
        )
        guide_path.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>Experiment metric guide</title><style>"
            + DASHBOARD_CSS
            + "</style><main>"
            + guide_html
            + "</main></html>"
        )
        docs = guide_path.name

    def link(text: str, anchor: str) -> str:
        return f'<a href="{escape(docs, quote=True)}#{anchor}">{escape(text)}</a>'

    complete = (
        (bool(evidence.executions) or study is not None)
        and all(e.status == "completed" for e in evidence.executions)
        and (study is None or study.status == "completed")
    )
    state = (
        "Final"
        if complete
        else "In-flight"
        if evidence.executions or study
        else "Saved"
    )
    parts = [
        f"<h1>{state} experiment report</h1><p>{escape(question)}</p>",
        f'<p class="muted">Rendered {now.isoformat(timespec="seconds")} · saved artifacts only · no automatic refresh</p>',
    ]
    if study is not None:
        parts.append(
            f"<p>Scientific study {escape(study.study)}: {escape(study.status)} · "
            f"{len(study.seeds)} independent training seeds. Thin lines show each seed; "
            "black diamonds and intervals resample training seeds and common deals. "
            "Three seeds give exploratory uncertainty, not a confirmatory claim. "
            "Development and endpoint cohorts remain separate. Scientific heartbeat: unavailable in this export.</p>"
        )
    if notes:
        parts.append(f"<p>{escape(notes)}</p>")
    parts.append(
        "<p>Pipeline and exact-replay smoke tests demonstrate software execution, "
        "not a positive control showing learned behavior improves. Positive-control "
        "training design is separate work. Entropy is not competence.</p>"
    )

    def section(title: str, anchor: str, figures: list[Figure]) -> str:
        body = f"<section><h2>{link(title, anchor)}</h2>"
        body += "".join(
            '<div class="figure" tabindex="0" role="region" aria-label="'
            + escape(title, quote=True)
            + '">'
            + _svg(f)
            + "</div>"
            for f in figures
        )
        if not figures:
            body += "<p>Unavailable: no common completed milestone across all expected runs in a compatible cohort.</p>"
        return body + "</section>"

    for execution in evidence.executions:
        as_of = (
            datetime.fromtimestamp(execution.last_seen_unix, timezone.utc).isoformat(
                timespec="seconds"
            )
            if execution.last_seen_unix
            else "unavailable"
        )
        age = (
            _number(now.timestamp() - execution.last_seen_unix)
            if execution.last_seen_unix
            else "unavailable"
        )
        statuses = {
            s: sum(a.status == s for a in execution.attempts)
            for s in ("completed", "pending", "running", "failed", "interrupted")
        }
        parts.append(
            f"<p><b>{escape(execution.hardware.name)}</b> · {escape(execution.hardware.host)} · {execution.hardware.cpu_threads} configured CPU threads<br>"
            f"{statuses['completed']}/{len(execution.attempts)} regime-seed runs complete · {statuses['running']} running · {statuses['pending']} pending · {statuses['failed'] + statuses['interrupted']} failed/interrupted<br>"
            f"{link('Evidence freshness', 'freshness')}: {as_of} · {age} s old · status {execution.status}</p>"
        )
    scientific_sections = "".join(
        section(title, anchor, figures)
        for title, anchor, figures in sections
        if anchor == "comparisons"
    )
    initialization = [r for r in evidence.monitors if r.coordinates.updates == 0]
    parts.append(
        "<p>Monitoring initialization: "
        + (
            "retained; shown only where cohort matching permits."
            if initialization
            else "unavailable; no evaluated zero-update checkpoint retained."
        )
        + " No improvement from initialization can be inferred without that baseline.</p>"
    )
    rows: list[list[str]] = []
    for run in evidence.runs:
        metrics = evidence.metrics(run)
        latest = metrics[-1] if metrics else {}
        updates = latest.get("progress/updates")
        planned = sum(getattr(s, "updates", 0) for s in run.regime.stages)
        loss_key = next(
            (k for k in ("rl/loss", "distillation/train_cross_entropy") if k in latest),
            None,
        )
        loss = (
            f"{loss_key}: {_number(latest[loss_key])}"
            if loss_key
            else "unavailable (no optimized loss)"
        )
        rows.append(
            [
                evidence.label(run.id),
                f"{_number(updates)} / {planned or 'unavailable'} updates; {run.status}",
                f"{_number(latest.get('throughput/learner_transitions_per_second'))} transitions/s",
                loss,
            ]
        )
    parts.append(f"<h2>{link('Progress, throughput and latest loss', 'learning')}</h2>")
    parts.append(
        _table(
            ["Variant / seed", "Recorded / planned", "SPS", "Latest applicable loss"],
            rows,
        )
    )
    parts.append(
        '<p class="muted">SPS = cumulative learner transitions / recorded training seconds. RL loss is the last optimized minibatch objective, never log loss. Updates include empty-filter skips.</p>'
    )
    eval_rows: list[list[str]] = []
    for run in evidence.runs:
        results = [r for r in evidence.monitors if r.run_id == run.id]
        if not results:
            eval_rows.append(
                [evidence.label(run.id), "pending / unavailable", "—", "—"]
            )
            continue
        result = max(results, key=lambda r: r.coordinates.training_seconds)
        c = result.coordinates
        latest_seconds = max(
            (
                float(row["progress/training_seconds"])
                for row in evidence.metrics(run)
                if isinstance(row.get("progress/training_seconds"), (int, float))
            ),
            default=None,
        )
        lag = (
            _number(max(0, latest_seconds - c.training_seconds))
            if latest_seconds is not None
            else "unavailable"
        )
        win = result.win
        rate = (
            f"{win.mean:.1%} [{win.lower:.1%}, {win.upper:.1%}]"
            if win and result.status == "completed"
            else "unavailable (incomplete)"
        )
        eval_rows.append(
            [
                evidence.label(run.id),
                f"{c.updates} updates; {c.learner_transitions} transitions; lag {lag} training s",
                f"{result.opponent.display_name}; {sum(r.valid for r in result.rows)}/{result.expected_games} games",
                rate,
            ]
        )
    if evidence.monitors or study is None:
        parts.append(
            f"<h2>{link('Latest monitoring evaluation · status, not a ranking', 'evaluation')}</h2>"
        )
        parts.append(
            _table(
                [
                    "Variant / seed",
                    "Checkpoint / age",
                    "Opponent / games",
                    "Win rate [95% interval]",
                ],
                eval_rows,
            )
        )
        parts.append(
            '<p class="muted">Intervals resample complete deals for one checkpoint, not training seeds. A one-deal fixture can have a zero-width interval; this is not precision or strength evidence.</p>'
        )
    parts.append(f"<h2>{link('Costs, resources and failures', 'costs')}</h2>")
    rss = [
        s.sampled_peak_rss_bytes
        for r in evidence.runs
        for s in r.stages
        if s.sampled_peak_rss_bytes is not None
    ]
    peak_mib = max(rss) / 1024**2 if rss else None
    for e in evidence.executions:
        parts.append(
            f"<p>{e.elapsed_seconds:.2f} elapsed wall s · {e.process_seconds:.2f} additive worker-process s (includes {e.evaluator_seconds:.2f} evaluator s) · host cost {_number(e.host_dollars)} USD<br>Latest host load {escape(str(e.host_load))}; sampled training peak RSS {_number(peak_mib)} MiB (not a whole-host peak)</p>"
        )
    if study is not None:
        parts.append(
            f"<p>Scientific evaluation: {study.seconds:.2f} s. {escape(study.accounting)}</p>"
        )
    parts.append(
        _table(
            [
                "Regime / seed",
                "Collection s",
                "Learning s",
                "Export s",
                "Diagnostics s",
            ],
            [
                [
                    evidence.label(r.id),
                    *[
                        _number(sum(getattr(stage, field) for stage in r.stages))
                        for field in (
                            "collection_seconds",
                            "learning_seconds",
                            "export_seconds",
                            "diagnostic_seconds",
                        )
                    ],
                ]
                for r in evidence.runs
            ],
        )
    )
    if not evidence.executions:
        parts.append(
            f"<p>Sampled training peak RSS {_number(peak_mib)} MiB. Host price and execution heartbeat unavailable.</p>"
        )
    failures = [f"{e.id}: {e.error}" for e in evidence.executions if e.error]
    failures += [
        f"{r.id}/{s.id}: {s.error}" for r in evidence.runs for s in r.stages if s.error
    ]
    failures += [
        f"{a.case}/{a.seed}: {a.status}: {a.error}"
        for e in evidence.executions
        for a in e.attempts
        if a.error
    ]
    failures += [
        f"{p.relative_to(evidence.root)}: {p.read_text()}"
        for p in evidence.paths
        if p.name == "attempt.json" and p.parent in evidence.failed_directories
    ]
    failures += [
        f"{r.run_id}: {r.status}: {r.error}"
        for r in evidence.monitors
        if r.status != "completed"
    ]
    if study is not None:
        failures += [
            f"Scientific {c.a} seed {c.training_seed} cutoff {c.cutoff}: incomplete/replay failure"
            for c in study.comparisons
            if not c.complete
        ]
    parts.append(
        "<p>"
        + escape(
            "; ".join(failures)
            if failures
            else "No additional failures in selected run/evaluation records."
        )
        + "</p>"
    )
    parts.append(
        f"<p>{link('Comparison rules', 'comparisons')}: Matched monitoring panels require common stage/update milestones across every retained run. Explicit within-run histories make no cross-regime comparison. Scientific panels use the declared study cohort. Work counts and time remain distinct; no unequal latest-checkpoint ranking.</p>"
    )
    parts.append(
        f"<h2>{link('Diagnostic sampling and missing updates', 'sampling')}</h2>"
    )
    parts.append(
        _table(
            [
                "Regime / seed",
                "Saved records",
                "Explicit skips",
                "No loss",
                "Selected / collected",
                "Entropy within .001 of ln(2) / recorded",
            ],
            _sampling_rows(evidence),
        )
    )
    parts.append(
        "<p>Counts cover saved records only. No loss can mean a skip or unavailable diagnostics. "
        "Missing historical updates cannot be reconstructed. Ataraxos loss, entropy and KL "
        "retain the last optimized timestep minibatch, not an update-wide or fixed-position average. "
        "The policy objective already includes KL penalties; unweighted KL panels are diagnostics, "
        "not an additional loss to add. Legal-action counts for that last minibatch are unavailable. "
        "Near ln(2) is consistent with nearly uniform binary choices, but does not identify their cause.</p>"
    )
    health_figures = numerical_figures(evidence)
    if health_figures:
        parts.append(
            "<details><summary>Numerical health: concentration, gradients and optimizer effect</summary>"
            + section("Numerical health", "numerical-health", health_figures)
            + "</details>"
        )
        parts.append(
            '<p>Batch counts cover collected rows; gradient and parameter effect sample the first step at iteration 1 and every 25th iteration. Missing samples are unavailable. Forced choices and filtering can explain zeros; missing gradients or zero parameter movement need investigation. <a href="https://github.com/loopflowstudio/etude/blob/main/docs/numerical-health.md">Metric definitions and failure investigation</a>.</p>'
        )
    for title, anchor, figures in sections:
        if anchor != "comparisons":
            parts.append(
                "<details><summary>"
                + escape(title)
                + "</summary>"
                + section(title, anchor, figures)
                + "</details>"
            )
    raw_path = output.with_name(output.stem + "-diagnostics.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps({r.id: evidence.metrics(r) for r in evidence.runs}, indent=2)
    )
    parts.append(
        f'<p><a href="{escape(raw_path.name, quote=True)}" download>Download all raw scalar diagnostics (JSON)</a></p>'
    )
    hashes = [
        {"path": str(p.relative_to(evidence.root)), "sha256": file_sha256(p)}
        for p in evidence.paths
    ]
    parts.append(
        "<details><summary>Evidence paths and hashes for this refresh</summary><pre>"
        + escape(json.dumps(hashes, indent=2))
        + "</pre></details>"
    )
    if notebook is not None:
        import os

        parts.insert(
            0,
            f'<p><a href="{escape(os.path.relpath(notebook.resolve(), output.parent.resolve()))}">Editable notebook generator</a></p>',
        )
    monitoring_path = output.with_name(output.stem + "-monitoring.json")
    monitoring_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in evidence.monitors], indent=2)
    )
    parts.append(
        f'<p><a href="{monitoring_path.name}">Raw monitoring rows and saved intervals</a></p>'
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Learning progress · manabot</title><style>"
        + DASHBOARD_CSS
        + "</style><main>"
        + learning_story(evidence, docs)
        + scientific_sections
        + '<details id="details"><summary>03 / Diagnostics, costs, failures &amp; source evidence</summary>'
        + "".join(parts)
        + "</details></main></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output
