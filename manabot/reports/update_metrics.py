"""Read Etude update rows and project retained regime stages for reports.

Later attempts supersede resumed coordinates; outcome tables join scalar rows.
Retained stages without tracked rows use their recorded diagnostics.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from etude.run import runs
from etude.train.manifest import StoredBlock
from etude.train.read import RunData, read

from manabot.training.records import StageRecord
from manabot.training.update_metrics import OUTCOME_TABLE, MetricRow, metric_row

# One Etude run's rows by completed-update coordinate.
type Updates = dict[int, MetricRow]

# Runs written from 1659b0e29 referenced manabot-written tables by this metric;
# earlier revisions such as bf4e280f7 read them.
_RETIRED_OUTCOME_REFERENCE = "self_play_outcomes/table"


class _Stage(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def operation(self) -> str: ...


class _Regime(Protocol):
    @property
    def stages(self) -> Sequence[_Stage]: ...


class _Run(Protocol):
    """A saved RegimeRun or its retained report projection."""

    @property
    def regime(self) -> _Regime: ...

    @property
    def stages(self) -> Sequence[StageRecord]: ...

    @property
    def metrics_run(self) -> str | None: ...


def read_updates(source: Path, records: Iterable[_Run]) -> dict[str, Updates]:
    """The records' Etude rows under `source`, by run folder name.

    Etude names a run's folder for the run, at any depth, and merges every copy
    it finds, so moved evidence still resolves. Absent runs are absent.
    """
    wanted = {Path(r.metrics_run).name for r in records if r.metrics_run}
    return {
        run: updates_from_run(read(source, experiment, run))
        for experiment, run in runs(source)
        if run in wanted
    }


def updates_from_run(data: RunData) -> Updates:
    """Each update's metrics, by `etude.train.read`'s rule on the updates clock.

    A later attempt's rows replace an earlier attempt's from the update it
    resumed at (or first logged); within an attempt, a later write of a metric
    at one update replaces an earlier one, so an outcome table logged beside
    the update's metrics joins them. Rows without values mark a clock position,
    not an update. Etude selects and verifies outcome tables the same way.
    """
    # Group blocks, not decoded rows: read one block at a time without populating
    # RunData.columns, which caches the entire history. Keep empty attempts too.
    writers: dict[str, list[StoredBlock]] = {
        attempt.writer: [] for attempt in data.writers
    }
    started = {attempt.writer: attempt.started_ms for attempt in data.writers}
    for block in data.blocks:
        writer = block.header.writer
        writers.setdefault(writer, []).append(block)
        started.setdefault(writer, block.header.wall_first)
    attempts = {attempt.writer: attempt for attempt in data.writers}
    selected: dict[int, MetricRow] = {}
    for writer in sorted(writers, key=started.__getitem__):
        attempt = attempts.get(writer)
        resumed = attempt.resumed_at.get("updates") if attempt is not None else None
        rows: dict[int, MetricRow] = {}
        for block in writers[writer]:
            table = block.columns()
            for index in range(table.rows):
                logged = table.row(index)
                clocks, values = logged.clocks, logged.values
                if "updates" not in clocks:
                    continue
                update = clocks["updates"]
                if not isinstance(update, (int, float)):
                    raise ValueError("Etude updates clock is not a number")
                if resumed is None:
                    resumed = update
                if values:
                    rows.setdefault(int(update), {}).update(_integral(values))
        # Current writers supersede from their first row, inclusive. Historical
        # writers explicitly resumed after a coordinate. Silent retries keep history.
        if resumed is not None:
            # Current writers replace the first logged coordinate itself;
            # historical checkpoint cuts retain it.
            first_row = attempt is not None and attempt.first_row
            selected = {
                u: row
                for u, row in selected.items()
                if u < resumed or (u == resumed and not first_row)
            }
        selected.update(rows)
    if any(_RETIRED_OUTCOME_REFERENCE in row for row in selected.values()):
        raise ValueError(
            f"run {data.name} records self-play outcomes in the retired "
            f"{_RETIRED_OUTCOME_REFERENCE!r} format; read it from a manabot "
            "revision that still reads it, such as bf4e280f7"
        )
    # `log_table` indexes its table's digest as a metric of the same name;
    # the rows replace it.
    for update in selected.values():
        update.pop(OUTCOME_TABLE, None)
    for table in data.read_table(OUTCOME_TABLE, "updates"):
        selected.setdefault(int(table.x), {}).update(
            metric_row({OUTCOME_TABLE: table.rows})
        )
    return dict(sorted(selected.items()))


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
