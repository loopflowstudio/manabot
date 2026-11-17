"""Read-only discovery of Etude runs for the training-time report.

The Etude SDK owns storage and metric decoding. A frozen read view records
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

from manabot.reports.time_cohort import (
    CohortRun,
    TimeCohort,
    TimingSource,
    TrainingInterval,
)

if TYPE_CHECKING:
    from etude.storage import Storage
    from etude.train.read import RunData


class _ReadSnapshot:
    """Freeze listed keys and fetched bytes; writes are not part of this reader."""

    def __init__(self, storage: "Storage") -> None:
        self.storage = storage
        self.name = storage.name
        self.listed = storage.keys("")
        self.cache: dict[tuple[str, int | None, int | None], bytes] = {}

    def keys(self, prefix: str) -> list[str]:
        return [key for key in self.listed if key.startswith(prefix)]

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


def _clock_before(
    run: "RunData", writers: list[str], clock: str, value: float
) -> float | None:
    """Latest finite clock value below `value` that any of `writers` logged."""
    latest: float | None = None
    for block in run.blocks:
        if block.header.writer not in writers:
            continue
        column = next((c for c in block.header.clocks if c.name == clock), None)
        if column is None:
            continue
        for number in block.array(column.data, "<f8"):
            logged = float(number)
            if math.isfinite(logged) and logged < value:
                latest = logged if latest is None else max(latest, logged)
    return latest


def _etude_run(run: "RunData", training_clock: str) -> tuple[CohortRun, str | None]:
    if run.file is None:
        raise ValueError("Etude declaration unavailable")
    declared = run.file
    identifier = "etude:" + declared.source_key.key()
    parent = "etude:" + declared.continues.parent.key() if declared.continues else None
    warnings: list[str] = []
    names = {c.name: c.kind for c in run.clocks}
    writers = sorted(
        {a.writer for a in run.writers} | {b.header.writer for b in run.blocks},
        key=lambda w: next(
            (a.started_ms for a in run.writers if a.writer == w),
            min(
                (b.header.wall_first for b in run.blocks if b.header.writer == w),
                default=0,
            ),
        ),
    )
    attempts = {a.writer: a for a in run.writers}
    # New writers establish supersession with their first row. Opening and
    # closing a tracking context alone neither replaces work nor measures time.
    observed_writers = [
        writer
        for writer in writers
        if any(b.header.writer == writer and b.header.rows for b in run.blocks)
        or (writer in attempts and attempts[writer].resumed_at)
    ]
    silent = len(writers) - len(observed_writers)
    if silent:
        warnings.append(
            f"{silent} attempts logged no clock coordinates; active time includes observed attempts only."
        )
    active: float | None = (
        0.0 if names.get(training_clock) == "time" and observed_writers else None
    )
    intervals = []
    for index, writer in enumerate(observed_writers):
        attempt = attempts.get(writer)
        blocks = [b for b in run.blocks if b.header.writer == writer]
        end = (
            _clock_endpoint(run, writer, training_clock) if active is not None else None
        )
        baseline = attempt.resumed_at.get(training_clock) if attempt else None
        if attempt is not None and attempt.first_row and baseline is not None:
            # Etude 0.4 records an attempt's first logged row, not the position
            # it resumed from. A loop logs then saves each pass, so the position
            # restored is the latest earlier row below that first row; the
            # first attempt of a run with no parent starts its clock at zero.
            baseline = (
                _clock_before(run, observed_writers[:index], training_clock, baseline)
                if index
                else None
            )
        if (
            baseline is None
            and index == 0
            and declared.continues is not None
            and declared.continues.clock == training_clock
        ):
            baseline = declared.continues.value
        if index == 0 and (attempt is None or attempt.first_row):
            # New writers record their first row, not the point they resumed from.
            # The initial writer starts at zero; a child starts at its lineage clock.
            baseline = (
                declared.continues.value
                if declared.continues is not None
                and declared.continues.clock == training_clock
                else 0.0
                if parent is None
                else None
            )
        elif baseline is None and index == 0 and parent is None:
            baseline = 0.0
        if (
            index > 0
            and attempt is not None
            and attempt.first_row
            and sum(block.header.rows for block in blocks) < 2
        ):
            # A retry's one observed coordinate establishes no elapsed interval.
            baseline = None
        if active is not None:
            if end is None or baseline is None or end < baseline:
                active = None
            else:
                active += end - baseline
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
    if run.outcome is not None and run.outcome.counters.dropped:
        warnings.append(
            f"Etude reports {run.outcome.counters.dropped} dropped metric rows."
        )
    if run.incomplete:
        warnings.append(f"Etude reports {run.incomplete} rows with incomplete clocks.")

    def count(clock: str) -> int | None:
        if names.get(clock) != "step":
            return None
        value = (
            _clock_endpoint(run, observed_writers[-1], clock)
            if observed_writers
            else None
        )
        if value is None and observed_writers:
            latest = attempts.get(observed_writers[-1])
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

    def label(metric: str) -> str:
        # RunData.points reads numeric metrics only. Decode just this text
        # column, newest writer first, rather than loading every update metric.
        for writer in reversed(observed_writers):
            for block in sorted(
                (b for b in run.blocks if b.header.writer == writer),
                key=lambda b: b.header.first,
                reverse=True,
            ):
                column = next(
                    (
                        c
                        for c in block.header.series
                        if c.name == metric and c.kind == "text"
                    ),
                    None,
                )
                if column is not None:
                    codes = block.array(column.data, "<u4")
                    if len(codes):
                        return column.labels[int(codes[-1])] or ""
        return ""

    status = run.outcome.state if run.outcome else "running or unfinished"
    return CohortRun(
        id=identifier,
        label=declared.job_name or run.experiment,
        experiment=run.experiment,
        kind=declared.kind or label("run/kind") or "unclassified",
        recipe=run.job or label("run/recipe"),
        seed=declared.seed,
        status=status,
        members=tuple(f"{identifier}/{w}" for w in writers),
        active_seconds=active,
        learner_transitions=count("learner_transitions"),
        updates=count("updates"),
        environment_steps=count("steps"),
        intervals=tuple(intervals),
        warnings=tuple(warnings),
    ), parent


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
    """Discover local/S3 Etude runs; failed sources stay visible.

    Local paths are explicit: a misspelled directory is a failure, not an empty
    history. Archives are listed as a coverage gap rather than silently expanded.
    A `run.json` that is not an Etude declaration (a saved regime record, say)
    is named in the warnings and holds no run.
    """
    # Optional report entry point: ordinary training does not require Etude.
    from etude.storage import storage_from_uri
    from etude.train.read import read_all

    tracked: dict[str, tuple[CohortRun, str | None]] = {}
    conflicts: set[str] = set()
    receipts: list[TimingSource] = []
    warnings: list[str] = []
    for source in dict.fromkeys(sources):
        snapshot = None
        try:
            if not source.startswith("s3://") and not Path(source).is_dir():
                raise FileNotFoundError("Evidence directory unavailable")
            snapshot = _ReadSnapshot(storage_from_uri(source))
            declarations = [
                key
                for key in snapshot.listed
                if key == "run.json" or key.endswith("/run.json")
            ]
            for key in declarations:
                try:
                    data = json.loads(snapshot.get(key))
                except (ValueError, OSError) as error:
                    warnings.append(
                        f"Unreadable record: {snapshot.uri(key)} ({type(error).__name__})"
                    )
                    continue
                # Etude's run.json: nested before 0.4, flat with an experiment since.
                if (
                    not isinstance(data, dict)
                    or "regime" in data
                    or not ("declaration" in data or "experiment" in data)
                ):
                    warnings.append(f"Unrecognized run record: {snapshot.uri(key)}")
            for run in read_all(snapshot):
                projected, parent = _etude_run(run, training_clock)
                row = (replace(projected, source_uris=(source,)), parent)
                identifier = row[0].id
                if identifier in conflicts:
                    continue
                if identifier in tracked and (
                    tracked[identifier][1] != parent
                    or replace(tracked[identifier][0], source_uris=())
                    != replace(row[0], source_uris=())
                ):
                    tracked.pop(identifier)
                    conflicts.add(identifier)
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
            elif not declarations:
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
    return TimeCohort(
        _etude_families(tuple(tracked.values())), tuple(receipts), tuple(warnings)
    )
