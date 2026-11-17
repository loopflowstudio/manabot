"""Selected logical runs for training-time reports.

A CohortRun is one Etude run family: its logged active time, progress
counters and attempt windows (TrainingInterval). `time_sources.load_time_sources`
builds the cohort and `write_cohort_report` writes the self-contained selection
page; nothing here trains or launches.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class TrainingInterval:
    label: str
    run_id: str | None
    start: float
    end: float
    status: str
    basis: str


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
    learner_transitions: int | None
    updates: int | None
    environment_steps: int | None
    intervals: tuple[TrainingInterval, ...]
    warnings: tuple[str, ...]
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
