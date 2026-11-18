"""Write per-update self-play diagnostics as Etude metrics and outcome tables.

`metric_row` flattens diagnostics into "/"-joined names. `phase_seconds` and
`time_since` measure collection and learning time from live profilers.
"""

from collections.abc import Mapping, Sequence
import re

from manabot.infra.profiler import Profiler

type MetricValue = float | int | str | bool
type MetricRow = dict[str, MetricValue]

# The Etude row table holding each self-play update's SelfPlayOutcome rows.
OUTCOME_TABLE = "self_play_outcomes"
# The deck population resolved by the writer, keyed by deck content identity.
ROSTER_TABLE = "training_decks"
# Lists whose entries are named by fields, not by position. Self-play lists
# occur only in the offline projection of a run's outcome tables
# (`manabot.reports.update_metrics`); live updates log the table itself.
_KEYED_LISTS = {
    "adaptive/exposures": ("cell", "seat"),
    "adaptive/allocations": ("stream", "game"),
    "adaptive/completed": ("stream", "game", "seat"),
    "adaptive/errors": ("cell",),
    "selection_groups": ("action_type", "terminal_distance"),
    "self_play_outcomes": ("deck", "opponent_deck", "position"),
}


def metric_row(diagnostic: Mapping[str, object]) -> MetricRow:
    """Flatten one update's diagnostics; None stays absent rather than zero."""
    row: MetricRow = {}
    _flatten(diagnostic, "", row)
    return row


def phase_seconds(collection: Profiler, learning: Profiler) -> dict[str, float]:
    """Cumulative profiler seconds under the historical normalized `time/` keys."""
    return {
        "time/"
        + "/".join(
            re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
            for part in path.split("/")
        ): node.running_total()
        for profiler in (collection, learning)
        for path, node in profiler.node_cache.items()
    }


def time_since(
    before: Mapping[str, float], now: Mapping[str, float]
) -> dict[str, float]:
    """Seconds each timer gained between two `phase_seconds` readings of one attempt."""
    return {key: value - before.get(key, 0.0) for key, value in now.items()}


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
