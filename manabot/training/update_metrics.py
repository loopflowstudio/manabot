"""Per-update self-play diagnostics as Etude metrics, written once and read back.

The learner logs each update's diagnostics to its Etude run (RunContext) and
nowhere else; TrainingRun keeps counters and receipts. `metric_row` is the one
shape those rows take: Etude's flat "/"-joined names, with keyed lists so a
selection group or matchup keeps its name instead of a list position.
`read_updates` reads runs' Etude rows under a directory once, and `stage_rows`
splits a run's rows by stage, from Etude for tracked self-play stages and from
the record's retained rows for every other stage.

`phase_seconds` and `time_since` give each row the seconds its update spent
in collection and learning, nested by "/" under `time/` for Etude's flame
graph. They are differences of the stage record's cumulative phase timers, so
the record stays the one measurement and each row its per-update share.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Protocol

from etude.run.listing import runs
from etude.train.read import RunData, read

from manabot.training.records import PhaseTiming, StageRecord

type MetricValue = float | int | str | bool
type MetricRow = dict[str, MetricValue]
# One Etude run's rows by completed-update coordinate.
type Updates = dict[int, MetricRow]

# Lists whose entries are named by fields, not by position.
_KEYED_LISTS = {
    "adaptive/exposures": ("cell", "seat"),
    "adaptive/allocations": ("stream", "game"),
    "adaptive/completed": ("stream", "game", "seat"),
    "adaptive/errors": ("cell",),
    "selection_groups": ("action_type", "terminal_distance"),
    "self_play_outcomes": ("deck", "opponent_deck", "position"),
}


class _Stage(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def operation(self) -> str: ...


class _Regime(Protocol):
    @property
    def stages(self) -> Sequence[_Stage]: ...


class _Run(Protocol):
    """A current TrainingRun or a retained report projection of one."""

    @property
    def regime(self) -> _Regime: ...

    @property
    def stages(self) -> Sequence[StageRecord]: ...

    @property
    def metrics_run(self) -> str | None: ...


def metric_row(diagnostic: Mapping[str, object]) -> MetricRow:
    """Flatten one update's diagnostics; None stays absent rather than zero."""
    row: MetricRow = {}
    _flatten(diagnostic, "", row)
    return row


def phase_seconds(record: StageRecord) -> dict[str, float]:
    """A stage's cumulative collection and learning seconds as `time/` metric keys.

    Profiler subphases nest below their phase by path, in lower snake case:
    learning's `Minibatches/Forward and loss` is
    `time/learning/minibatches/forward_and_loss`.
    """
    found = {
        "time/collection": record.collection_seconds,
        "time/learning": record.learning_seconds,
    }
    for phase, timings in (
        ("collection", record.collection_timings),
        ("learning", record.learning_timings),
    ):
        found |= _subphases(phase, timings or {})
    return found


def _subphases(phase: str, timings: Mapping[str, PhaseTiming]) -> dict[str, float]:
    return {
        f"time/{phase}/"
        + "/".join(
            re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
            for part in path.split("/")
        ): timing.seconds
        for path, timing in timings.items()
    }


def time_since(before: Mapping[str, float], now: Mapping[str, float]) -> MetricRow:
    """Seconds each timer gained between two `phase_seconds` readings.

    A timer that went down was restarted, as a new attempt's profiler is
    under a resumed stage's totals, so all of its value is new.
    """
    return {
        key: value - before.get(key, 0.0) if value >= before.get(key, 0.0) else value
        for key, value in now.items()
    }


def _flatten(value: object, name: str, row: MetricRow) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(item, f"{name}{key}/", row)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        keys = _KEYED_LISTS.get(name.rstrip("/")) or _KEYED_LISTS.get(
            name.rstrip("/").rsplit("/", 1)[-1]
        )
        for index, item in enumerate(value):
            if keys is not None and isinstance(item, Mapping):
                label = "/".join(str(item[key]) for key in keys)
                rest = {k: v for k, v in item.items() if k not in keys}
                _flatten(rest, f"{name}{label}/", row)
            else:
                _flatten(item, f"{name}{index}/", row)
    elif isinstance(value, (bool, int, float, str)):
        row[name.rstrip("/")] = value
    else:
        raise TypeError(f"unsupported diagnostic value at {name!r}")


def read_updates(source: Path, records: Iterable[_Run]) -> dict[str, Updates]:
    """The records' Etude rows under `source`, by run folder name.

    Etude names a run's folder for the run, at any depth, and merges every copy
    it finds, so moved evidence still resolves. Absent runs are absent.
    """
    wanted = {Path(r.metrics_run).name for r in records if r.metrics_run}
    return {
        run: _updates(read(source, experiment, run))
        for experiment, run in runs(source)
        if run in wanted
    }


def _updates(data: RunData) -> Updates:
    """Each update's latest row, by `etude.train.read`'s rule on the updates clock.

    A later attempt's rows replace an earlier attempt's from the update it
    resumed at (or first logged); a later write of one update replaces an
    earlier one. Rows without values mark a clock position, not an update.
    """
    # Blocks arrive in attempt start order; each writer is one attempt.
    written: dict[str, list[tuple[int, MetricRow]]] = {}
    start = {
        attempt.writer: attempt.resumed_at["updates"]
        for attempt in data.attempts
        if "updates" in attempt.resumed_at
    }
    for block, table in zip(data.blocks, data.tables, strict=True):
        rows = written.setdefault(block.header.writer, [])
        for index in range(table.rows):
            logged = table.row(index)
            clocks, values = logged["clocks"], logged["values"]
            if not isinstance(clocks, dict) or "updates" not in clocks:
                continue
            update = clocks["updates"]
            if not isinstance(update, (int, float)):
                raise ValueError("Etude updates clock is not a number")
            start.setdefault(block.header.writer, update)
            if isinstance(values, dict) and values:
                rows.append((int(update), _integral(values)))
    result: Updates = {}
    cutoff: float | None = None
    for writer in reversed(list(written)):
        for update, values in reversed(written[writer]):
            if cutoff is None or update <= cutoff:
                result.setdefault(update, values)
        if writer in start:
            cutoff = min(start[writer], cutoff if cutoff is not None else start[writer])
    return dict(sorted(result.items()))


def _integral(values: Mapping[str, object]) -> MetricRow:
    """Etude stores numbers as floats; counts read back as the ints they were."""
    row: MetricRow = {}
    for name, value in values.items():
        if isinstance(value, float) and value.is_integer():
            row[name] = int(value)
        elif isinstance(value, (bool, int, float, str)):
            row[name] = value
    return row


def stage_rows(run: _Run, tracked: Mapping[str, Updates]) -> dict[str, list[MetricRow]]:
    """Rows by stage id, given `read_updates` rows. Self-play stages without
    their Etude run read their retained rows: none for current runs.

    Coordinates are cumulative, so a stage owns the updates after its
    predecessors' (and its imported origin's) and through its own count. An
    earlier attempt reads only the updates it completed, not its child's.
    """
    updates = tracked.get(Path(run.metrics_run).name) if run.metrics_run else None
    stages = {stage.id: stage for stage in run.regime.stages}
    result: dict[str, list[MetricRow]] = {}
    total = 0
    for record in run.stages:
        operation = stages[record.id].operation
        if operation not in {"train_self_play", "train_compound"}:
            result[record.id] = [metric_row(d) for d in record.diagnostics]
            continue
        origin = record.learning_state_origin
        low = total + (origin.iteration if origin is not None else 0)
        total = low + record.updates
        if operation == "train_self_play" and updates is not None:
            result[record.id] = [
                row for update, row in updates.items() if low < update <= total
            ]
        else:
            result[record.id] = [metric_row(d) for d in record.diagnostics]
    return result
