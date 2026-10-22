"""Selected logical runs for training-time reports.

CohortRun joins attempt-local timing to retained progress without summing recovery
endpoints. Native phase accounting remains in time_profile; this module owns the
cohort projection and self-contained selection page, never training execution.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from manabot.experiment.execution import ExperimentRun
from manabot.reports.records import ReportRun
from manabot.reports.time_profile import (
    TimeSlice,
    TrainingInterval,
    TrainingProfile,
    profile_body,
    training_intervals,
    training_profiles,
)


@dataclass(frozen=True)
class TimingSource:
    uri: str
    sha256: str
    # A digest of a byte range is never labeled a digest of the whole object.
    start: int | None = None
    length: int | None = None


@dataclass(frozen=True)
class CohortRun:
    id: str
    label: str
    experiment: str
    kind: str
    hardware: str
    recipe: str
    seed: int | None
    status: str
    members: tuple[str, ...]
    active_seconds: float | None
    elapsed_seconds: float | None
    learner_transitions: int | None
    updates: int | None
    environment_steps: int | None
    cost_usd: float | None
    phases: dict[str, float]
    intervals: tuple[TrainingInterval, ...]
    profiles: tuple[TrainingProfile, ...]
    warnings: tuple[str, ...]
    phase_coverage: tuple[str, ...] = ()
    source_uris: tuple[str, ...] = ()
    model: str = "unknown"


@dataclass(frozen=True)
class TimeCohort:
    runs: tuple[CohortRun, ...]
    sources: tuple[TimingSource, ...] = ()
    warnings: tuple[str, ...] = ()


def lineage_roots(parents: dict[str, str | None]) -> dict[str, str]:
    """Keep missing ancestors as family identities; cycles are corrupt evidence."""
    result: dict[str, str] = {}
    for identifier in parents:
        seen: set[str] = set()
        root = identifier
        while (parent := parents.get(root)) is not None:
            if root in seen:
                raise ValueError(f"Cyclic training ancestry: {identifier}")
            seen.add(root)
            root = parent
        result[identifier] = root
    return result


def phase_totals(profiles: Sequence[TrainingProfile]) -> dict[str, float]:
    """Sum domain phases once, retaining their nested measurements for drilldown."""
    totals: dict[str, float] = defaultdict(float)

    def visit(node: TimeSlice) -> None:
        if node.name in PHASES:
            totals[node.name] += node.seconds
        elif node.children:
            for child in node.children:
                visit(child)
        else:
            # A zero root or an unpartitioned stage is not a domain phase.
            totals[node.name if node.name in PHASES else "Unattributed"] += node.seconds

    for profile in profiles:
        if profile.root is not None:
            visit(profile.root)
    return dict(totals)


PHASES = (
    "Setup",
    "Collection",
    "Learning",
    "Stage exports",
    "Diagnostics / failure accounting",
    "Initial admission wait",
    "Unattributed",
)


def native_cohort(
    runs: Sequence[ReportRun],
    executions: Sequence[ExperimentRun] = (),
    sources: Sequence[TimingSource] = (),
) -> TimeCohort:
    profiles = {p.run_id: p for p in training_profiles(runs)}
    by_id = {run.id: run for run in runs}
    roots = lineage_roots({r.id: r.parent_run_id for r in by_id.values()})
    families: dict[str, list[ReportRun]] = defaultdict(list)
    for run in by_id.values():
        families[roots[run.id]].append(run)
    intervals, missing = training_intervals(runs, executions)
    output: list[CohortRun] = []
    for root, members in sorted(families.items()):
        children = {r.parent_run_id for r in members}
        leaves = [r for r in members if r.id not in children]
        leaf = leaves[0]
        selected = tuple(profiles[r.id] for r in members)
        warnings = [w for p in selected for w in p.warnings]
        complete_lineage = root in by_id and len(leaves) == 1
        if not complete_lineage:
            warnings.append(
                "Progress unavailable: missing ancestor or branching recovery lineage."
            )
        phases = phase_totals(selected)
        elapsed = (
            sum(p.root.seconds for p in selected if p.root is not None)
            if all(p.root is not None for p in selected)
            else None
        )
        owned = []
        for run in members:
            parent = by_id.get(run.parent_run_id or "")
            inherited = (
                {s.id for s in parent.stages if s.status == "completed"}
                if parent
                else set()
            )
            owned.extend(s for s in run.stages if s.id not in inherited)
        active_known = (
            root in by_id
            and bool(owned)
            and all(
                {"collection_seconds", "learning_seconds"} <= s.model_fields_set
                and all(
                    math.isfinite(value) and value >= 0
                    for value in (s.collection_seconds, s.learning_seconds)
                )
                for s in owned
            )
            and not any("exceed" in w or "conflict" in w for w in warnings)
        )
        active = (
            sum(stage.collection_seconds + stage.learning_seconds for stage in owned)
            if active_known
            else None
        )
        # Recovery stage work counters describe the retained trajectory, not
        # this attempt's increment. The unique leaf owns each final endpoint.
        transitions = (
            sum(s.learner_transitions for s in leaf.stages)
            if complete_lineage
            and leaf.stages
            and all("learner_transitions" in s.model_fields_set for s in leaf.stages)
            else None
        )
        learning_ids = {
            s.id
            for s in leaf.regime.stages
            if s.operation in {"train_self_play", "train_supervised", "train_compound"}
        }
        updates = (
            sum(len(s.diagnostics) for s in leaf.stages if s.id in learning_ids)
            if complete_lineage
            and leaf.stages
            and all(
                "diagnostics" in s.model_fields_set
                for s in leaf.stages
                if s.id in learning_ids
            )
            else None
        )
        hardware_labels: set[str] = set()
        for member in members:
            gpu = member.identities.get("cuda", {})
            host = member.identities.get("hardware", {})
            if isinstance(gpu, dict) and gpu.get("device"):
                hardware_labels.add(str(gpu["device"]))
            elif isinstance(host, dict) and host.get("processor"):
                hardware_labels.add(str(host["processor"]))
            else:
                hardware_labels.add("unknown")
        hardware = " / ".join(sorted(hardware_labels))
        runtime_identity = hashlib.sha256(
            json.dumps(
                [
                    {
                        key: member.identities.get(key)
                        for key in ("hardware", "cuda", "torch")
                    }
                    for member in members
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        recipes = sorted({r.regime_digest for r in members})
        if len(recipes) > 1:
            warnings.append(
                "Recovery family contains multiple recipes; throughput is not pooled with another family."
            )
        output.append(
            CohortRun(
                id=f"native:{root}",
                label=leaf.regime.id,
                experiment=leaf.regime.id,
                kind="unclassified",
                hardware=hardware or "unknown",
                recipe=" / ".join(recipes) + f" · runtime {runtime_identity}",
                seed=leaf.seed,
                model=f"width {leaf.regime.agent.hidden_dim} · {leaf.regime.agent.attention_layers} attention layers",
                status="branched" if len(leaves) > 1 else leaf.status,
                members=tuple(sorted(r.id for r in members)),
                active_seconds=active,
                elapsed_seconds=elapsed,
                learner_transitions=transitions,
                updates=updates,
                environment_steps=None,
                cost_usd=None,
                phases=phases,
                intervals=tuple(
                    i for i in intervals if i.run_id in {r.id for r in members}
                ),
                profiles=selected,
                warnings=tuple(warnings),
                phase_coverage=tuple(
                    name
                    for name, field in (
                        ("Collection", "collection_seconds"),
                        ("Learning", "learning_seconds"),
                        ("Stage exports", "export_seconds"),
                        ("Diagnostics / failure accounting", "diagnostic_seconds"),
                        ("Initial admission wait", "admission_wait_seconds"),
                    )
                    if owned
                    and all(
                        field in stage.model_fields_set
                        and getattr(stage, field) is not None
                        for stage in owned
                    )
                    and not any(
                        "exceed" in w or "conflict" in w or "parent unavailable" in w
                        for w in warnings
                    )
                )
                + (
                    ("Setup",)
                    if all("setup_seconds" in r.model_fields_set for r in members)
                    else ()
                ),
            )
        )
    # Keep attempts that failed before writing a TrainingRun discoverable.
    seen_executions: dict[str, ExperimentRun] = {}
    for execution in executions:
        if execution.id in seen_executions:
            if seen_executions[execution.id] != execution:
                raise ValueError(f"Conflicting execution snapshots: {execution.id}")
            continue
        seen_executions[execution.id] = execution
        for attempt in execution.attempts:
            if attempt.run_id in by_id:
                continue
            identifier = f"execution:{execution.id}:{attempt.ordinal}"
            output.append(
                CohortRun(
                    id=identifier,
                    label=attempt.case,
                    experiment=execution.id,
                    kind="unclassified",
                    hardware="unknown",
                    recipe="",
                    seed=attempt.seed,
                    status=attempt.status,
                    members=(identifier,),
                    active_seconds=None,
                    elapsed_seconds=None,
                    learner_transitions=None,
                    updates=None,
                    environment_steps=None,
                    cost_usd=None,
                    phases={},
                    profiles=(),
                    intervals=training_intervals(
                        [], [execution.model_copy(update={"attempts": [attempt]})]
                    )[0],
                    warnings=(
                        "Training artifact unavailable; only saved process timing can be shown.",
                    ),
                )
            )
    return TimeCohort(tuple(output), tuple(sources), missing)


def write_cohort_report(
    cohort: TimeCohort,
    output: Path,
    *,
    selected: Sequence[str] | None = None,
    top: int = 5,
    include_benchmarks: bool = False,
) -> Path:
    """Write an offline picker and a source-bound snapshot. No external assets."""
    if top < 1:
        raise ValueError("top must be positive")
    ids = {r.id for r in cohort.runs}
    if len(ids) != len(cohort.runs):
        raise ValueError("Conflicting cohort run identities")
    if selected is not None and not set(selected) <= ids:
        raise ValueError(f"Unknown selected runs: {sorted(set(selected) - ids)}")
    payload = asdict(cohort)
    snapshot = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    data = dict(
        payload,
        snapshot=snapshot,
        generated_at=datetime.now(timezone.utc).isoformat(),
        selected=list(selected) if selected is not None else None,
        top=top,
        include_benchmarks=include_benchmarks,
    )
    encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    details = []
    for index, run in enumerate(cohort.runs):
        parts = [
            f'<details data-detail="{index}"><summary>{escape(run.label)} · {escape(run.id)}</summary>'
        ]
        for profile in run.profiles:
            parts.append(
                f"<h4>{escape(profile.run_id)} · {escape(profile.status)}</h4>"
            )
            parts.append(profile_body(profile))
        parts.extend(f"<p>{escape(w)}</p>" for w in run.warnings)
        parts.extend(f"<p>Source: {escape(uri)}</p>" for uri in run.source_uris)
        parts.append("</details>")
        details.append("".join(parts))
    template = Path(__file__).with_name("time_cohort.html").read_text()
    script = Path(__file__).with_name("time_cohort.js").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path = output.with_name(output.stem + "-training-time.json")
    replacements = {
        "__DETAILS__": "".join(details),
        "__DATA__": encoded,
        "__SCRIPT__": script,
        "__SNAPSHOT_LINK__": escape(data_path.name, quote=True),
    }
    # One pass: placeholder-like text in an author's label is ordinary data.
    html = re.sub(
        r"__(?:DETAILS|DATA|SCRIPT|SNAPSHOT_LINK)__",
        lambda match: replacements[match.group()],
        template,
    )
    data_temporary = data_path.with_suffix(".json.tmp")
    data_temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    data_temporary.replace(data_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(html)
    temporary.replace(output)
    return output
