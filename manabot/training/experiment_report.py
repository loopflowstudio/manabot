"""Read retained experiment evidence and render an offline experiment dashboard.

ReportEvidence owns no execution state. Notebook cells select evidence and build
figures; write_dashboard only renders those choices. Matching requires every
retained run in a compatible cohort at the same stage/update coordinate.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import StringIO
import json
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.experiment_execution import ExperimentRun
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorResult
from manabot.training.monitoring import Scalar, training_dashboard


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
        diagnostics = [d for s in run.stages for d in s.diagnostics]
        for row, diagnostic in zip(rows, diagnostics, strict=True):
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
        tuple(TrainingRun.model_validate_json(p.read_text()) for p in runs),
        tuple(
            MonitorResult.model_validate_json(p.read_text())
            for p in monitors
            if not any(parent in failed for parent in p.parents)
        ),
        tuple(ExperimentRun.model_validate_json(p.read_text()) for p in executions),
        failed,
        (*runs, *monitors, *executions, *attempts),
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
    evidence: ReportEvidence, axis: Literal["training_seconds", "environment_decisions"]
) -> list[Figure]:
    """Plot matched milestones per seed; bands are saved deal uncertainty, not seed CIs."""
    figures: list[Figure] = []
    for panel, results in matched_milestones(evidence).items():
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
            title=f"Matched stage/update milestones · cohort {panel[:8]}",
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
) -> Path:
    """Render notebook-selected figures and a concise status summary; never mutate inputs."""
    now = datetime.now(timezone.utc)

    def link(text: str, anchor: str) -> str:
        return f'<a href="{escape(docs, quote=True)}#{anchor}">{escape(text)}</a>'

    complete = bool(evidence.executions) and all(
        e.status == "completed" for e in evidence.executions
    )
    parts = [
        f"<h1>{'Final' if complete else 'In-flight'} experiment report</h1><p>{escape(question)}</p>",
        f'<p class="muted">Rendered {now.isoformat(timespec="seconds")} · saved artifacts only · no automatic refresh</p>',
    ]
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
    parts.append(
        f"<h2>{link('Latest evaluation · status, not a ranking', 'evaluation')}</h2>"
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
    failures = [f"{e.id}: {e.error}" for e in evidence.executions if e.error]
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
    parts.append(
        "<p>"
        + escape(
            "; ".join(failures)
            if failures
            else "No failures in selected saved records."
        )
        + "</p>"
    )
    parts.append(
        f"<p>{link('Comparison rules', 'comparisons')}: only common stage/update milestones across every retained run appear below. Work counts and time remain distinct; no unequal latest-checkpoint ranking.</p>"
    )
    for title, anchor, figures in sections:
        parts.append(f"<section><h2>{link(title, anchor)}</h2>")
        parts.extend(
            '<div class="figure" tabindex="0" role="region" aria-label="Chart; scroll horizontally on narrow screens">'
            + _svg(f)
            + "</div>"
            for f in figures
        )
        if not figures:
            parts.append(
                "<p>Unavailable: no common completed milestone across all expected runs in a compatible cohort.</p>"
            )
        parts.append("</section>")
    hashes = [
        {"path": str(p.relative_to(evidence.root)), "sha256": file_sha256(p)}
        for p in evidence.paths
    ]
    parts.append(
        "<details><summary>Evidence paths and hashes for this refresh</summary><pre>"
        + escape(json.dumps(hashes, indent=2))
        + "</pre></details>"
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment report</title><style>body{font:16px/1.5 system-ui;color:#202d35;background:#fafbf9;max-width:1080px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:20px;margin:24px 0 10px}a{color:#146a76;text-underline-offset:3px}.muted{color:#55646b;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #ccd6d7}svg{width:100%;height:auto}.figure{overflow-x:auto}.figure svg{min-width:740px}section{margin-top:40px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.table{overflow-x:auto}@media(max-width:600px){body{margin:20px auto;padding:0 14px}table{font-size:12px}td,th{padding:6px}h1{font-size:25px}}@media print{body{max-width:none}section{break-inside:avoid}}</style><main>'
        + "".join(parts)
        + "</main></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output
