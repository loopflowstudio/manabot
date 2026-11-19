"""Write per-update self-play diagnostics as Etude metrics and outcome tables.

`metric_row` flattens diagnostics into "/"-joined names. `phase_seconds` and
`time_since` measure collection and learning time from live profilers.
`outcome_scalars` and `parse_outcome_scalar` own the scalar names that runs
before the outcome table logged.
"""

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Literal

from manabot.infra.profiler import Profiler
from manabot.training.rollout.self_play import SelfPlayOutcome

type MetricValue = float | int | str | bool
type MetricRow = dict[str, MetricValue]

# The Etude row tables holding each self-play update's SelfPlayOutcome rows and
# the GameSettingCount rows of the same games.
OUTCOME_TABLE = "self_play_outcomes"
GAME_SETTINGS_TABLE = "training_game_settings"
# Update-zero tables describing a learner's corpus (`manabot.training.corpus`):
# each resolved deck entry's cards, and the facts that hold for the whole run.
DECK_TABLE = "training_deck_cards"
RUN_TABLE = "training_run"
# Lists whose entries are named by fields, not by position.
_KEYED_LISTS = {
    "adaptive/exposures": ("cell", "seat"),
    "adaptive/allocations": ("stream", "game"),
    "adaptive/completed": ("stream", "game", "seat"),
    "adaptive/errors": ("cell",),
    "selection_groups": ("action_type", "terminal_distance"),
}
# Runs before the outcome table logged each count as the scalar
# `self_play_outcomes/<deck>/<opponent>/<play|draw>/<wins|losses|draws>`, and
# the offline projection still exposes table rows under those names. The
# label's index is the native seat.
SEAT_LABELS: tuple[Literal["play"], Literal["draw"]] = ("play", "draw")
OUTCOME_COUNTS = ("wins", "losses", "draws")


def outcome_scalars(rows: Iterable[SelfPlayOutcome]) -> dict[str, int]:
    """Outcome rows under their historical scalar names."""
    return {
        f"{OUTCOME_TABLE}/{row['deck']}/{row['opponent_deck']}"
        f"/{SEAT_LABELS[row['seat']]}/{count}": row[count]
        for row in rows
        for count in OUTCOME_COUNTS
    }


def parse_outcome_scalar(name: str) -> tuple[str, str, int, str] | None:
    """Deck, opposing deck, native seat and count of a historical scalar name."""
    parts = name.split("/")
    if len(parts) != 5 or parts[0] != OUTCOME_TABLE:
        return None
    _, deck, opponent, label, count = parts
    if label not in SEAT_LABELS or count not in OUTCOME_COUNTS:
        raise ValueError(f"invalid self-play outcome {name!r}")
    return deck, opponent, SEAT_LABELS.index(label), count


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
