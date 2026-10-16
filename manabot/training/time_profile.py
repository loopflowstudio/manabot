"""Training-only time accounting from retained regime and execution records.

TrainingProfile partitions one attempt's recorded elapsed time; TrainingInterval
places an attempt on a wall-clock timeline. Aggregate phase widths never imply
chronological order. Recovery copies, overlapping counters and missing timestamps
remain explicit instead of becoming fabricated work or zero-cost intervals.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import math

from manabot.training.experiment_execution import ExperimentRun
from manabot.training.models import TrainingRun


@dataclass(frozen=True)
class TimeSlice:
    name: str
    seconds: float
    children: tuple["TimeSlice", ...] = ()


@dataclass(frozen=True)
class TrainingProfile:
    run_id: str
    label: str
    status: str
    root: TimeSlice | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TrainingInterval:
    label: str
    run_id: str | None
    start: float
    end: float
    status: str
    basis: str


def _partition(name: str, seconds: float, parts: Sequence[TimeSlice]) -> TimeSlice:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name}: invalid recorded duration")
    if any(not math.isfinite(p.seconds) or p.seconds < 0 for p in parts):
        raise ValueError(f"{name}: invalid recorded phase duration")
    remainder = seconds - sum(p.seconds for p in parts)
    # Float clock arithmetic can differ by nanoseconds. Larger contradictions
    # must remain visible; never rescale overlapping measurements into a pie.
    if remainder < -max(1e-6, seconds * 1e-9):
        raise ValueError(f"{name}: phase totals exceed recorded duration")
    children = tuple(p for p in parts if p.seconds > 0)
    if remainder > 0:
        children += (TimeSlice("Unattributed", remainder),)
    return TimeSlice(name, seconds, children)


def training_profiles(runs: Sequence[TrainingRun]) -> tuple[TrainingProfile, ...]:
    """Partition attempt-local costs, excluding unchanged completed recovery stages.

    Missing parents or conflicting inherited stages leave attribution unavailable.
    Prior run seconds, recovery cost and monitoring exports are not added: the
    latter two are already inside stage elapsed time, with no stage allocation.
    """
    by_id: dict[str, TrainingRun] = {}
    for run in runs:
        if run.id in by_id and by_id[run.id].model_dump(
            exclude_unset=True
        ) != run.model_dump(exclude_unset=True):
            raise ValueError(f"conflicting retained training run {run.id}")
        by_id[run.id] = run
    profiles = []
    for run in by_id.values():
        warnings: list[str] = []
        root: TimeSlice | None = None
        try:
            if "seconds" not in run.model_fields_set:
                raise ValueError("Recorded attempt duration unavailable")
            parent = by_id.get(run.parent_run_id) if run.parent_run_id else None
            if run.parent_run_id and parent is None:
                raise ValueError("Recovery parent unavailable; stage ownership unknown")
            inherited = (
                {s.id: s for s in parent.stages if s.status == "completed"}
                if parent
                else {}
            )
            parts = []
            if "setup_seconds" in run.model_fields_set:
                parts.append(TimeSlice("Setup", run.setup_seconds))
            else:
                warnings.append("Setup duration unavailable")
            for stage in run.stages:
                if stage.id in inherited:
                    if stage != inherited[stage.id]:
                        raise ValueError("Recovered completed-stage costs conflict")
                    continue
                if "seconds" not in stage.model_fields_set:
                    warnings.append(f"{stage.id}: stage duration unavailable")
                    continue
                costs = []
                for field, label, value in (
                    ("collection_seconds", "Collection", stage.collection_seconds),
                    ("learning_seconds", "Learning", stage.learning_seconds),
                    ("export_seconds", "Stage exports", stage.export_seconds),
                    (
                        "diagnostic_seconds",
                        "Diagnostics / failure accounting",
                        stage.diagnostic_seconds,
                    ),
                ):
                    if field in stage.model_fields_set:
                        costs.append(TimeSlice(label, value))
                    else:
                        warnings.append(f"{stage.id}: {label} duration unavailable")
                try:
                    phase = _partition(
                        stage.id,
                        stage.seconds,
                        costs,
                    )
                except ValueError as error:
                    warnings.append(str(error))
                    phase = _partition(stage.id, stage.seconds, [])
                parts.append(phase)
            root = _partition("Recorded training attempt", run.seconds, parts)
        except ValueError as error:
            warnings.append(str(error))
            if (
                "seconds" in run.model_fields_set
                and math.isfinite(run.seconds)
                and run.seconds >= 0
            ):
                root = _partition("Recorded training attempt", run.seconds, [])
        profiles.append(
            TrainingProfile(
                run.id,
                f"{run.regime.id} · seed {run.seed}",
                run.status,
                root,
                tuple(warnings),
            )
        )
    return tuple(profiles)


def training_intervals(
    runs: Sequence[TrainingRun], executions: Sequence[ExperimentRun]
) -> tuple[tuple[TrainingInterval, ...], tuple[str, ...]]:
    """Use saved attempt timestamps, otherwise an explicitly recorded calendar span.

    Running intervals end at their saved heartbeat, never at report-render time.
    Process occupancy and recorded run calendar time are labeled separately.
    Pending/failed attempts without intervals stay listed as unavailable.
    """
    intervals: list[TrainingInterval] = []
    missing: list[str] = []
    covered: set[str] = set()
    seen: set[tuple[str, int]] = set()

    def add(
        label: str,
        run_id: str | None,
        start: float | None,
        end: float | None,
        status: str,
        basis: str,
    ) -> None:
        if (
            start is None
            or end is None
            or not all(math.isfinite(v) for v in (start, end))
            or end < start
        ):
            missing.append(f"{label}: {status}; timestamps unavailable or inconsistent")
            return
        intervals.append(TrainingInterval(label, run_id, start, end, status, basis))

    for execution in executions:
        for attempt in execution.attempts:
            key = (execution.id, attempt.ordinal)
            if key in seen:
                continue
            seen.add(key)
            if attempt.run_id:
                covered.add(attempt.run_id)
            end = attempt.finished_unix
            if end is None and attempt.status == "running":
                end = execution.last_seen_unix
            add(
                f"{attempt.case} · seed {attempt.seed} · attempt {attempt.ordinal}",
                attempt.run_id,
                attempt.started_unix,
                end,
                attempt.status,
                "Process lifetime (includes work outside the regime)",
            )
    for run in runs:
        if run.id in covered:
            continue
        covered.add(run.id)
        end = run.last_recorded_wall_seconds
        start = (
            end - run.calendar_seconds
            if end is not None and run.calendar_seconds > 0
            else None
        )
        add(
            f"{run.regime.id} · seed {run.seed} · {run.id[:8]}",
            run.id,
            start,
            end,
            run.status,
            "Recorded calendar span (may include downtime)",
        )
    return tuple(sorted(intervals, key=lambda i: (i.start, i.label))), tuple(missing)


def _rect(x: float, y: float, width: float, text: str, color: str) -> str:
    title = escape(text)
    label = (
        f'<text x="{x + 5:.3f}" y="{y + 20:.3f}" fill="white" font-size="12">'
        f"{escape(text[: max(0, int(width / 7) - 2)])}</text>"
        if width > 65
        else ""
    )
    return (
        f'<g><title>{title}</title><rect x="{x:.3f}" y="{y:.3f}" '
        f'width="{width:.3f}" height="31" fill="{color}" stroke="white"/>{label}</g>'
    )


def _profile_svg(root: TimeSlice) -> str:
    colors = {
        "Collection": "#53758a",
        "Learning": "#497e71",
        "Stage exports": "#9b703d",
        "Unattributed": "#77716a",
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 120" '
        'role="img" aria-label="Aggregate training phase durations">'
    ]

    def draw(node: TimeSlice, left: float, depth: int) -> None:
        scale = 1000 / root.seconds if root.seconds else 0
        parts.append(
            _rect(
                left * scale,
                depth * 39,
                node.seconds * scale,
                f"{node.name}: {node.seconds:.3f} s",
                colors.get(node.name, "#334b5b"),
            )
        )
        for child in node.children:
            draw(child, left, depth + 1)
            left += child.seconds

    draw(root, 0, 0)
    return "".join(parts) + "</svg>"


def _profile_rows(root: TimeSlice, depth: int = 0) -> str:
    row = (
        f"<tr><td>{'&nbsp;' * (depth * 4)}{escape(root.name)}</td>"
        f"<td>{root.seconds:.6f}</td></tr>"
    )
    return row + "".join(_profile_rows(child, depth + 1) for child in root.children)


def profile_body(profile: TrainingProfile) -> str:
    """One measured hierarchy, shared by the dashboard and cohort drilldown."""
    if profile.root is None:
        return "<p>Recorded duration unavailable.</p>"
    return (
        _profile_svg(profile.root)
        + "<table><thead><tr><th>Phase</th><th>Recorded seconds</th></tr></thead><tbody>"
        + _profile_rows(profile.root)
        + "</tbody></table>"
    )


def training_time_report(
    runs: Sequence[TrainingRun], executions: Sequence[ExperimentRun]
) -> str:
    """Render a training-only HTML section inside the existing offline dashboard."""
    profiles = training_profiles(runs)
    intervals, missing = training_intervals(runs, executions)
    parts = [
        '<section id="training-time" class="figure"><h2>Where training time goes</h2>',
        "<p>Training only. Phase widths are aggregate elapsed seconds, not "
        "execution order or CPU/GPU utilization. Nested costs are included "
        "in their parents. Unattributed time stays visible. Evaluation lanes "
        "and detailed collection/learning traces are not included.</p>",
    ]
    for profile in profiles:
        parts.append(
            f"<details><summary>{escape(profile.label)} · "
            f"{escape(profile.status)} · run {escape(profile.run_id)}</summary>"
        )
        parts.append(profile_body(profile))
        parts.extend(
            f"<p>Attribution warning: {escape(w)}</p>" for w in profile.warnings
        )
        parts.append("</details>")
    if not profiles:
        parts.append("<p>Training phase records unavailable.</p>")
    parts.append(
        "<h3>Training attempts over time</h3><p>Concurrent lanes overlap; "
        "their widths must not be summed into calendar time. Running lanes "
        "stop at the saved heartbeat. Cross-machine clock uncertainty is "
        "not recorded, so apparent overlap does not establish causality.</p>"
    )
    if intervals:
        origin = min(i.start for i in intervals)
        span = max(i.end for i in intervals) - origin
        parts.append(
            f"<p>Origin: {datetime.fromtimestamp(origin, timezone.utc).isoformat()} "
            f"· span {span:.3f} seconds.</p>"
        )
        for interval in intervals:
            width = 1000 * (interval.end - interval.start) / span if span else 0
            left = 1000 * (interval.start - origin) / span if span else 0
            title = f"{interval.label} · {interval.status} · {interval.basis}"
            parts.append(
                f"<p>{escape(title)} · offset {interval.start - origin:.3f} s · "
                f"duration {interval.end - interval.start:.3f} s</p>"
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 32" '
                f'role="img" aria-label="{escape(title, quote=True)}">'
                + _rect(left, 0, width, title, "#53758a")
                + "</svg>"
            )
    else:
        parts.append("<p>Timeline unavailable: no retained training timestamps.</p>")
    parts.extend(f"<p>{escape(item)}</p>" for item in missing)
    return "".join(parts) + "</section>"
