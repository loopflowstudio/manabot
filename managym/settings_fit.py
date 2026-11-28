"""Compare a policy's recorded training game settings against a planned matchup.

One pure comparison function serves Academy's policy listing and check endpoint,
local evaluation admission, and Etude experiment reports without introducing any
load gate. A mismatch informs and never refuses; an unrecorded setting reads as
``unknown``, never as matching.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

SettingStatus = Literal["match", "rare", "unseen", "unknown"]
DEFAULT_RARE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TrainedSettingDistribution:
    """Observed game counts for each value a training setting took."""

    setting: str
    counts: dict[str, int]

    @property
    def total_games(self) -> int:
        return sum(self.counts.values())

    def share(self, value: str) -> float:
        total = self.total_games
        if total <= 0:
            return 0.0
        return self.counts.get(value, 0) / total

    def shares(self) -> dict[str, float]:
        total = self.total_games
        if total <= 0:
            return {}
        return {
            key: count / total
            for key, count in sorted(
                self.counts.items(), key=lambda item: (-item[1], item[0])
            )
        }


@dataclass(frozen=True)
class SettingComparison:
    """Comparison of one planned setting against its recorded training share."""

    setting: str
    planned: str
    status: SettingStatus
    planned_share: float | None
    trained_shares: dict[str, float]
    total_games: int

    def phrase(self) -> str:
        if self.status == "unknown":
            return f"{self.setting}: training not recorded (planned {self.planned})"
        parts = ", ".join(
            f"{share * 100:.0f}% {val}" for val, share in self.trained_shares.items()
        )
        if self.status == "match":
            return f"{self.setting}: trained {parts} ({self.total_games:,} games)"
        return (
            f"{self.setting}: trained {parts} ({self.total_games:,} games) — "
            f"planned {self.planned}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "planned": self.planned,
            "status": self.status,
            "planned_share": self.planned_share,
            "trained_shares": self.trained_shares,
            "total_games": self.total_games,
            "message": self.phrase(),
        }


@dataclass(frozen=True)
class SettingsFitReport:
    """Informational comparison across all requested settings."""

    comparisons: tuple[SettingComparison, ...]

    @property
    def warnings(self) -> tuple[SettingComparison, ...]:
        return tuple(
            item for item in self.comparisons if item.status in ("unseen", "rare")
        )

    @property
    def unknown(self) -> tuple[SettingComparison, ...]:
        return tuple(item for item in self.comparisons if item.status == "unknown")

    @property
    def status(self) -> Literal["match", "mismatch", "unknown"]:
        if self.warnings:
            return "mismatch"
        if not self.comparisons or len(self.unknown) == len(self.comparisons):
            return "unknown"
        return "match"

    def warning_lines(self) -> tuple[str, ...]:
        return tuple(item.phrase() for item in self.warnings)

    def summary(self) -> str | None:
        if self.warnings:
            return "; ".join(item.phrase() for item in self.warnings)
        if self.status == "unknown":
            return "Training settings unrecorded."
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary(),
            "warnings": list(self.warning_lines()),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


def distributions_from_counts(
    raw: Mapping[str, Any] | None, *, seat: int | None = None
) -> dict[str, TrainedSettingDistribution]:
    """Parse ``{'decklists:closed': 266321, 'seat0:mana_payment:auto': ...}`` maps."""
    if not raw:
        return {}
    by_setting_game: dict[str, dict[str, int]] = defaultdict(dict)
    by_setting_seat: dict[str, dict[str, int]] = defaultdict(dict)
    for key, count in raw.items():
        if not isinstance(count, int) or count <= 0:
            continue
        parts = key.split(":")
        if len(parts) == 2:
            setting, value = parts
            by_setting_game[setting][value] = (
                by_setting_game[setting].get(value, 0) + count
            )
        elif len(parts) == 3 and parts[0].startswith("seat"):
            seat_label, setting, value = parts
            try:
                row_seat = int(seat_label[4:])
            except ValueError:
                continue
            if seat is None or row_seat == seat:
                by_setting_seat[setting][value] = (
                    by_setting_seat[setting].get(value, 0) + count
                )
    merged: dict[str, TrainedSettingDistribution] = {}
    for setting in set(by_setting_game) | set(by_setting_seat):
        counts = by_setting_seat.get(setting) or by_setting_game.get(setting) or {}
        if counts:
            merged[setting] = TrainedSettingDistribution(setting, dict(counts))
    return merged


def compare_settings(
    trained: Mapping[str, TrainedSettingDistribution] | None,
    planned: Mapping[str, str],
    *,
    rare_threshold: float = DEFAULT_RARE_THRESHOLD,
) -> SettingsFitReport:
    """Compare planned matchup settings against recorded training counts.

    Never raises for a mismatch; settings absent from ``trained`` are reported
    with status ``unknown``.
    """
    trained_map = trained or {}
    comparisons: list[SettingComparison] = []
    for setting, value in planned.items():
        dist = trained_map.get(setting)
        if dist is None or dist.total_games <= 0:
            comparisons.append(
                SettingComparison(
                    setting=setting,
                    planned=value,
                    status="unknown",
                    planned_share=None,
                    trained_shares={},
                    total_games=0,
                )
            )
            continue
        share = dist.share(value)
        if share <= 0.0:
            status: SettingStatus = "unseen"
        elif share < rare_threshold:
            status = "rare"
        else:
            status = "match"
        comparisons.append(
            SettingComparison(
                setting=setting,
                planned=value,
                status=status,
                planned_share=share,
                trained_shares=dist.shares(),
                total_games=dist.total_games,
            )
        )
    return SettingsFitReport(tuple(comparisons))
