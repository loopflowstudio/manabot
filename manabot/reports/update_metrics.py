"""Read one Etude run's update rows for reports.

Later attempts supersede resumed coordinates; outcome tables join scalar rows.
`outcome_rows` is the one reader of completed self-play outcomes: whatever
form a run stored, each row names its native seat.
"""

from collections.abc import Mapping
import math

from etude.train.manifest import StoredBlock
from etude.train.read import RunData
from pydantic import TypeAdapter

from manabot.training.rollout.self_play import SelfPlayOutcome
from manabot.training.update_metrics import (
    DECK_TABLE,
    GAME_SETTINGS_TABLE,
    OUTCOME_TABLE,
    RUN_TABLE,
    SEAT_LABELS,
    MetricRow,
    outcome_scalars,
    parse_outcome_scalar,
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
    for update, rows in table_outcomes(data).items():
        selected.setdefault(update, {}).update(outcome_scalars(rows))
    return dict(sorted(selected.items()))


def outcome_rows(
    data: RunData, through: int | None = None
) -> dict[int, list[SelfPlayOutcome]]:
    """Completed self-play outcomes by update, each row naming its native seat.

    Stored runs keep their bytes and read into the same rows: scalar counts
    named `play` or `draw`, tables whose rows carry that label as `position`,
    and tables written by native seat. An update present here recorded its
    outcomes, possibly none; an absent update recorded nothing.
    """
    updates: dict[int, list[SelfPlayOutcome]] = {}
    # Decode scalars only when such a series exists: current writers log the table.
    if any(
        column.name.startswith(OUTCOME_TABLE + "/")
        for block in data.blocks
        for column in block.header.series
    ):
        for update, row in scalar_updates(data, through).items():
            counts: dict[tuple[str, str, int], dict[str, int]] = {}
            for name, value in row.items():
                parsed = parse_outcome_scalar(name)
                if parsed is not None:
                    *key, count = parsed
                    counts.setdefault(tuple(key), {})[count] = int(value)
            if counts and update > 0:
                updates[update] = [
                    SelfPlayOutcome(
                        deck=deck,
                        opponent_deck=opponent,
                        seat=seat,
                        wins=c.get("wins", 0),
                        losses=c.get("losses", 0),
                        draws=c.get("draws", 0),
                    )
                    for (deck, opponent, seat), c in counts.items()
                ]
    updates.update(table_outcomes(data, through))
    return updates


def table_outcomes(
    data: RunData, through: int | None = None
) -> dict[int, list[SelfPlayOutcome]]:
    outcomes = TypeAdapter(list[SelfPlayOutcome])
    return {
        int(table.x): outcomes.validate_python(
            [
                # Tables written before the native-seat column labelled the seat.
                {
                    **{k: v for k, v in row.items() if k != "position"},
                    "seat": SEAT_LABELS.index(row["position"]),
                }
                if "position" in row
                else row
                for row in table.rows
            ],
            strict=True,
        )
        for table in data.read_table(OUTCOME_TABLE, "updates")
        if through is None or table.x <= through
    }


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
                # Corpus tables index their digests here too; they are no metrics.
                for corpus_table in (DECK_TABLE, RUN_TABLE, GAME_SETTINGS_TABLE):
                    values.pop(corpus_table, None)
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
