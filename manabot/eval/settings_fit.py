"""Evaluate whether a checkpoint trained under a planned experiment's settings.

Adapts ``training_corpus(...)`` DuckDB tables to the shared
``managym.settings_fit.compare_settings`` comparator so local evaluation
admission, Etude reports, and Academy produce identical warnings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from managym.settings_fit import (
    DEFAULT_RARE_THRESHOLD,
    SettingsFitReport,
    TrainedSettingDistribution,
    compare_settings,
)

if TYPE_CHECKING:
    import duckdb


def distributions_from_corpus(
    db: duckdb.DuckDBPyConnection, *, seat: int | None = None
) -> dict[str, TrainedSettingDistribution]:
    """Aggregate ``game_settings`` from a ``training_corpus`` DuckDB connection."""
    rows = db.sql(
        "select seat, setting, value, sum(games)::bigint as games "
        "from game_settings group by all"
    ).fetchall()
    by_setting_game: dict[str, dict[str, int]] = defaultdict(dict)
    by_setting_seat: dict[str, dict[str, int]] = defaultdict(dict)
    for row_seat, setting, value, games in rows:
        count = int(games)
        if count <= 0:
            continue
        if row_seat is None:
            by_setting_game[setting][value] = (
                by_setting_game[setting].get(value, 0) + count
            )
        elif seat is None or int(row_seat) == seat:
            by_setting_seat[setting][value] = (
                by_setting_seat[setting].get(value, 0) + count
            )
    merged: dict[str, TrainedSettingDistribution] = {}
    for setting in set(by_setting_game) | set(by_setting_seat):
        counts = by_setting_seat.get(setting) or by_setting_game.get(setting) or {}
        if counts:
            merged[setting] = TrainedSettingDistribution(setting, dict(counts))
    return merged


def corpus_settings_fit(
    db: duckdb.DuckDBPyConnection,
    planned: Mapping[str, str],
    *,
    seat: int | None = None,
    rare_threshold: float = DEFAULT_RARE_THRESHOLD,
) -> SettingsFitReport:
    """Compare planned settings against a checkpoint's training corpus."""
    return compare_settings(
        distributions_from_corpus(db, seat=seat),
        planned,
        rare_threshold=rare_threshold,
    )


def checkpoint_experiment_summary(
    db: duckdb.DuckDBPyConnection,
    planned: Mapping[str, str],
    *,
    planned_cards: Collection[str] | None = None,
    seat: int | None = None,
    rare_threshold: float = DEFAULT_RARE_THRESHOLD,
) -> str:
    """One-line fit answer for a checkpoint and a planned experiment."""
    parts = ["loads"]
    if planned_cards is not None:
        unique_planned = set(planned_cards)
        seen_rows = db.sql("select distinct card from deck_cards").fetchall()
        seen = {row[0] for row in seen_rows}
        unseen = len(unique_planned - seen)
        parts.append(f"{unseen} of {len(unique_planned)} cards new")
    report = corpus_settings_fit(db, planned, seat=seat, rare_threshold=rare_threshold)
    summary = report.summary()
    if summary:
        parts.append(summary)
    return "; ".join(parts)
