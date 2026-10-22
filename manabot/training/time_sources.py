"""Read-only discovery of native training artifacts and Etude tracking sources.

The optional Etude SDK owns storage and metric decoding. A frozen read view records
full-file or byte-range digests, so the offline report identifies exactly what it
used. Source failures remain coverage warnings; no launcher or provider is called.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from manabot.training.experiment_report import RetainedExecution, RetainedRun
from manabot.training.time_cohort import (
    CohortRun,
    TimeCohort,
    TimingSource,
    lineage_roots,
    native_cohort,
)
from manabot.training.time_profile import TimeSlice, TrainingInterval, TrainingProfile

if TYPE_CHECKING:
    from etude.run.storage import Storage
    from etude.train.read import RunData


class _ReadSnapshot:
    """Freeze listed keys and fetched bytes; writes are not part of this reader."""

    def __init__(self, storage: "Storage") -> None:
        self.storage = storage
        self.name = storage.name
        self.listed = storage.keys("")
        self.cache: dict[tuple[str, int | None, int | None], bytes] = {}
        self.native_prefixes: list[str] = []

    def keys(self, prefix: str) -> list[str]:
        return [
            key
            for key in self.listed
            if key.startswith(prefix)
            and not any(key.startswith(p) for p in self.native_prefixes)
        ]

    def uri(self, key: str) -> str:
        return self.storage.uri(key)

    def get(self, key: str) -> bytes:
        identity = (key, None, None)
        if identity not in self.cache:
            self.cache[identity] = self.storage.get(key)
        return self.cache[identity]

    def get_range(self, key: str, start: int, length: int) -> bytes:
        identity = (key, start, length)
        if identity not in self.cache:
            self.cache[identity] = self.storage.get_range(key, start, length)
        return self.cache[identity]

    def put(self, key: str, data: bytes) -> None:
        raise RuntimeError("Report sources are read-only")

    def put_file(self, key: str, path: Path) -> None:
        raise RuntimeError("Report sources are read-only")

    def delete(self, key: str) -> bool:
        raise RuntimeError("Report sources are read-only")

    def receipts(self) -> tuple[TimingSource, ...]:
        return tuple(
            TimingSource(self.uri(key), hashlib.sha256(body).hexdigest(), start, length)
            for (key, start, length), body in self.cache.items()
        )


def _clock_endpoint(run: "RunData", writer: str, clock: str) -> float | None:
    """Last finite cumulative clock; an in-writer rewind is not an elapsed total."""
    previous: float | None = None
    for block in sorted(
        (b for b in run.blocks if b.header.writer == writer),
        key=lambda b: b.header.first,
    ):
        column = next((c for c in block.header.clocks if c.name == clock), None)
        if column is None:
            continue
        for number in block.array(column.data, "<f8"):
            value = float(number)
            if not math.isfinite(value):
                continue
            if value < 0 or (previous is not None and value < previous):
                return None
            previous = value
    return previous


def _etude_run(run: "RunData", training_clock: str) -> tuple[CohortRun, str | None]:
    if run.file is None:
        raise ValueError("Etude declaration unavailable")
    declared = run.file.declaration
    identifier = "etude:" + declared.source_key.key()
    parent = "etude:" + declared.continues.parent.key() if declared.continues else None
    warnings = [
        "Detailed phase attribution and measured rental cost unavailable in these tracking logs."
    ]
    names = {c.name: c.kind for c in run.clocks}
    writers = sorted(
        {a.writer for a in run.attempts} | {b.header.writer for b in run.blocks},
        key=lambda w: next(
            (a.started_ms for a in run.attempts if a.writer == w),
            min(
                (b.header.wall_first for b in run.blocks if b.header.writer == w),
                default=0,
            ),
        ),
    )
    attempts = {a.writer: a for a in run.attempts}
    active: float | None = (
        0.0 if names.get(training_clock) == "time" and writers else None
    )
    intervals = []
    for index, writer in enumerate(writers):
        attempt = attempts.get(writer)
        end = (
            _clock_endpoint(run, writer, training_clock) if active is not None else None
        )
        baseline = attempt.resumed_at.get(training_clock) if attempt else None
        if (
            baseline is None
            and index == 0
            and declared.continues is not None
            and declared.continues.clock == training_clock
        ):
            baseline = declared.continues.value
        if baseline is None and index == 0 and parent is None:
            baseline = 0.0
        if active is not None:
            if end is None or baseline is None or end < baseline:
                active = None
            else:
                active += end - baseline
        blocks = [b for b in run.blocks if b.header.writer == writer]
        if blocks:
            start_ms = (
                attempt.started_ms
                if attempt
                else min(b.header.wall_first for b in blocks)
            )
            end_ms = max(b.header.wall_last for b in blocks)
            if end_ms >= start_ms:
                intervals.append(
                    TrainingInterval(
                        writer,
                        identifier,
                        start_ms / 1000,
                        end_ms / 1000,
                        "last logged observation",
                        "Logged observation window (not process lifetime)",
                    )
                )
    if active is None:
        warnings.append(
            f"Active time unavailable: {training_clock!r} needs a declared time clock and complete attempt resume coordinates."
        )
    if not run.blocks:
        warnings.append(
            "No readable metric blocks: checkpoint coordinates alone cannot establish duration or complete attempt history."
        )
    if not intervals:
        warnings.append("Timeline unavailable: no retained metric observation windows.")
    if run.state is not None and run.state.counters.dropped:
        warnings.append(
            f"Etude reports {run.state.counters.dropped} dropped metric rows."
        )
    if run.incomplete:
        warnings.append(f"Etude reports {run.incomplete} rows with incomplete clocks.")

    def count(clock: str) -> int | None:
        if names.get(clock) != "step":
            return None
        value = _clock_endpoint(run, writers[-1], clock) if writers else None
        if value is None and writers:
            latest = attempts.get(writers[-1])
            value = latest.resumed_at.get(clock) if latest else None
        # Checkpoints carry explicit observed coordinates, not planned work.
        if not writers and run.checkpoints:
            value = max(run.checkpoints, key=lambda c: c.wall_ms).at.get(clock)
        return (
            int(value)
            if value is not None
            and math.isfinite(value)
            and value >= 0
            and value.is_integer()
            else None
        )

    status = run.state.state if run.state else "running or unfinished"
    profile = TrainingProfile(
        identifier,
        run.experiment,
        status,
        TimeSlice(
            "Logged active training", active, (TimeSlice("Unattributed", active),)
        )
        if active is not None
        else None,
        tuple(warnings),
    )
    return CohortRun(
        id=identifier,
        label=declared.job_name or run.experiment,
        experiment=run.experiment,
        kind=declared.kind or "unclassified",
        hardware="unknown",
        recipe=run.job or "",
        seed=declared.seed,
        status=status,
        members=tuple(f"{identifier}/{w}" for w in writers),
        active_seconds=active,
        elapsed_seconds=None,
        learner_transitions=count("learner_transitions"),
        updates=count("updates"),
        environment_steps=count("steps"),
        cost_usd=None,
        phases={},
        intervals=tuple(intervals),
        profiles=(profile,),
        warnings=tuple(warnings),
    ), parent


def _etude_families(
    rows: Sequence[tuple[CohortRun, str | None]],
) -> tuple[CohortRun, ...]:
    parents = {r.id: parent for r, parent in rows}
    roots = lineage_roots(parents)
    families: dict[str, list[CohortRun]] = defaultdict(list)
    for run, _ in rows:
        families[roots[run.id]].append(run)
    result = []
    for root, members in sorted(families.items()):
        leaves = [r for r in members if r.id not in set(parents.values())]
        leaf = leaves[0]
        complete = root in parents and len(leaves) == 1
        warnings = [w for r in members for w in r.warnings]
        if not complete:
            warnings.append(
                "Progress unavailable: missing ancestor or branching continuation lineage."
            )
        active = (
            sum(r.active_seconds for r in members if r.active_seconds is not None)
            if root in parents and all(r.active_seconds is not None for r in members)
            else None
        )
        result.append(
            replace(
                leaf,
                id=root,
                members=tuple(m for r in members for m in r.members),
                active_seconds=active,
                learner_transitions=leaf.learner_transitions if complete else None,
                updates=leaf.updates if complete else None,
                environment_steps=leaf.environment_steps if complete else None,
                intervals=tuple(i for r in members for i in r.intervals),
                profiles=tuple(p for r in members for p in r.profiles),
                warnings=tuple(warnings),
                source_uris=tuple(
                    dict.fromkeys(uri for r in members for uri in r.source_uris)
                ),
            )
        )
    return tuple(result)


def load_time_sources(
    sources: Sequence[str], *, training_clock: str = "training_seconds"
) -> TimeCohort:
    """Discover local/S3 native and Etude evidence; failed sources stay visible.

    Local paths are explicit: a misspelled directory is a failure, not an empty
    history. Archives are listed as a coverage gap rather than silently expanded.
    """
    # Optional report entry point: ordinary training does not require Etude.
    from etude.run.storage import storage_from_uri
    from etude.train.read import read_all

    native: dict[str, RetainedRun] = {}
    native_sources: dict[str, list[str]] = defaultdict(list)
    conflicts: set[str] = set()
    executions: list[RetainedExecution] = []
    tracked: dict[str, tuple[CohortRun, str | None]] = {}
    tracked_conflicts: set[str] = set()
    receipts: list[TimingSource] = []
    warnings: list[str] = []
    for source in dict.fromkeys(sources):
        snapshot = None
        try:
            if not source.startswith("s3://") and not Path(source).is_dir():
                raise FileNotFoundError("Evidence directory unavailable")
            snapshot = _ReadSnapshot(storage_from_uri(source))
            for key in snapshot.listed:
                if not (
                    key.endswith("/run.json")
                    or key == "run.json"
                    or key.endswith("/experiment.json")
                    or key == "experiment.json"
                ):
                    continue
                try:
                    body = snapshot.get(key)
                    data = json.loads(body)
                    if isinstance(data, dict) and "regime" in data and "stages" in data:
                        run = RetainedRun.model_validate_json(body)
                        native_sources[run.id].append(snapshot.uri(key))
                        snapshot.native_prefixes.append(key.removesuffix("run.json"))
                        if run.id in conflicts:
                            continue
                        if run.id in native and native[run.id].model_dump(
                            exclude_unset=True
                        ) != run.model_dump(exclude_unset=True):
                            conflicts.add(run.id)
                            native.pop(run.id)
                            warnings.append(
                                f"Conflicting native snapshots excluded: {run.id}"
                            )
                        else:
                            native[run.id] = run
                    elif key.endswith("experiment.json"):
                        executions.append(RetainedExecution.model_validate_json(body))
                    elif not isinstance(data, dict) or "declaration" not in data:
                        warnings.append(f"Unrecognized run record: {snapshot.uri(key)}")
                except (ValueError, OSError) as error:
                    warnings.append(
                        f"Unreadable record: {snapshot.uri(key)} ({type(error).__name__})"
                    )
            for run in read_all(snapshot):
                projected, parent = _etude_run(run, training_clock)
                row = (replace(projected, source_uris=(source,)), parent)
                identifier = row[0].id
                if identifier in tracked_conflicts:
                    continue
                if identifier in tracked and (
                    tracked[identifier][1] != parent
                    or replace(tracked[identifier][0], source_uris=())
                    != replace(row[0], source_uris=())
                ):
                    tracked.pop(identifier)
                    tracked_conflicts.add(identifier)
                    warnings.append(
                        f"Conflicting Etude snapshots excluded: {identifier}"
                    )
                else:
                    if identifier in tracked:
                        row = (
                            replace(
                                row[0],
                                source_uris=tuple(
                                    dict.fromkeys(
                                        (*tracked[identifier][0].source_uris, source)
                                    )
                                ),
                            ),
                            parent,
                        )
                    tracked[identifier] = row
            archives = sum(
                key.endswith((".tar", ".tar.gz", ".tgz", ".zip"))
                for key in snapshot.listed
            )
            if archives:
                warnings.append(
                    f"{source}: {archives} archives not expanded. Supply their verified extracted evidence directories to include those runs."
                )
            if not snapshot.listed:
                warnings.append(f"{source}: no stored objects found.")
            elif not any(
                key == "run.json" or key.endswith("/run.json")
                for key in snapshot.listed
            ):
                warnings.append(
                    f"{source}: no run declarations found among {len(snapshot.listed)} stored objects."
                )
        except Exception as error:
            # Cloud exception messages can contain credential-bearing endpoints.
            # Keep the configured source and error class, never the raw message.
            warnings.append(
                f"Source incomplete or unavailable: {source} ({type(error).__name__}). No complete-history claim is possible."
            )
        finally:
            if snapshot is not None:
                receipts.extend(snapshot.receipts())
    cohort = native_cohort(tuple(native.values()), executions, receipts)
    return TimeCohort(
        tuple(
            replace(
                run,
                source_uris=tuple(
                    uri
                    for member in run.members
                    for uri in native_sources.get(member, [])
                ),
            )
            for run in cohort.runs
        )
        + _etude_families(tuple(tracked.values())),
        cohort.sources,
        tuple(warnings) + cohort.warnings,
    )
