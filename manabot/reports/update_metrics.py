"""Read one Etude run's update rows for reports.

Later attempts supersede resumed coordinates; outcome tables join scalar rows.
"""

from collections.abc import Mapping
import math

from etude.train.manifest import StoredBlock
from etude.train.read import RunData

from manabot.training.update_metrics import (
    DECK_TABLE,
    OUTCOME_TABLE,
    SETTINGS_TABLE,
    MetricRow,
    metric_row,
)

# One Etude run's rows by completed-update coordinate.
type Updates = dict[int, MetricRow]

# Runs written from 1659b0e29 referenced manabot-written tables by this metric;
# earlier revisions such as bf4e280f7 read them.
_RETIRED_OUTCOME_REFERENCE = "self_play_outcomes/table"


def updates_from_run(data: RunData) -> Updates:
    """Each update's metrics, by `etude.train.read`'s rule on the updates clock.

    A later attempt's rows replace an earlier attempt's from the update it
    resumed at (or first logged); within an attempt, a later write of a metric
    at one update replaces an earlier one, so an outcome table logged beside
    the update's metrics joins them. Rows without values mark a clock position,
    not an update. Etude selects and verifies outcome tables the same way.
    """
    selected = scalar_updates(data)
    # `log_table` indexes its table's digest as a metric of the same name;
    # the rows replace it.
    for update in selected.values():
        update.pop(OUTCOME_TABLE, None)
    for table in data.read_table(OUTCOME_TABLE, "updates"):
        selected.setdefault(int(table.x), {}).update(
            metric_row({OUTCOME_TABLE: table.rows})
        )
    return dict(sorted(selected.items()))


def scalar_updates(data: RunData, through: int | None = None) -> Updates:
    """Select scalar writes, optionally through an inclusive update."""
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
            clock = next((c for c in block.header.clocks if c.name == "updates"), None)
            if clock is None:
                continue
            if resumed is None:
                resumed = clock.first
            if (
                through is not None
                and clock.increasing
                and clock.first is not None
                and clock.first > through
            ):
                continue
            table = block.columns()
            for index, coordinate in enumerate(table.clocks["updates"]):
                update = float(coordinate)
                if math.isnan(update):
                    continue
                if resumed is None:
                    resumed = update
                if through is not None and update > through:
                    continue
                values = table.row(index).values
                # Setup tables index their digests here too; they are no metrics.
                values.pop(DECK_TABLE, None)
                values.pop(SETTINGS_TABLE, None)
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
    return selected


def _integral(values: Mapping[str, object]) -> MetricRow:
    """Etude stores numbers as floats; counts read back as the ints they were."""
    row: MetricRow = {}
    for name, value in values.items():
        if isinstance(value, float) and value.is_integer():
            row[name] = int(value)
        elif isinstance(value, (bool, int, float, str)):
            row[name] = value
    return row
