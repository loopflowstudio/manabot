"""Offline MTG-281 figures and descriptive summaries from retained projections.

Slopes are ordinary least squares, per 100 updates, with no independence or
causality claim. Game length means native decisions per completed game from
adjacent cumulative counters; it includes decisions in recovered failed games.
The permanent-header index is reconstructed at block-end updates, not measured
manifest size or fold timing. All figures come from the same frozen snapshots.

PYTHONPATH=. uv run --no-sync python scripts/report_collection_growth.py --data experiments/data/collection-growth-281 --out .runs/collection-growth/report
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from scripts.measure_collection_growth import Capture

FloatArray = npt.NDArray[np.float64]
PHASES = {
    "Collection": "time/collection",
    "Inference + selection": "time/collection/policy_inference_and_action_selection",
    "Observation preparation": "time/collection/observation_preparation",
    "Rollout recording": "time/collection/rollout_recording",
    "Native stepping": "time/collection/environment_step_and_encoding/native_stepping",
    "Learning": "time/learning",
}
WINDOWS = [
    (1, 50),
    (400, 450),
    (500, 550),
    (550, 600),
    (600, 650),
    (800, 850),
    (850, 900),
    (1600, 1700),
]


@dataclass(frozen=True)
class Reading:
    name: str
    source: Capture
    update: FloatArray
    metrics: dict[str, FloatArray]
    games: FloatArray
    decisions_per_game: FloatArray
    headers_mb: FloatArray
    blocks: FloatArray
    permanent_endpoints: FloatArray


@dataclass(frozen=True)
class Window:
    first: int
    last: int
    count: int
    medians: dict[str, float]
    collection_mean: float
    collection_p90: float


class FoldStatistics(TypedDict):
    count: int
    mean_seconds: float
    median_seconds: float


def reading(name: str, source: Capture) -> Reading:
    rows = [r for r in source.updates if "time/collection" in r.metrics]
    update = np.array([r.update for r in rows], dtype=float)
    if len(update) < 2 or not np.all(np.diff(update) == 1) or update[0] != 1:
        raise ValueError("report requires complete consecutive updates starting at one")
    names = {key for row in rows for key in row.metrics}
    metrics = {k: np.array([r.metrics.get(k, np.nan) for r in rows]) for k in names}
    games = np.diff(metrics["coordinates/games"], prepend=0)
    decisions = np.diff(metrics["coordinates/environment_decisions"], prepend=0)
    if np.any(games <= 0) or np.any(decisions < 0):
        raise ValueError("invalid cumulative work counters")
    permanent = sorted(
        (b.last_update, b.header_bytes)
        for b in source.blocks
        if b.level >= 2 and b.last_update is not None
    )
    endpoints = np.array([end for end, _ in permanent], dtype=float)
    # Each endpoint contributes once; updates include all blocks ending there.
    counts = np.searchsorted(endpoints, update, side="right")
    header_totals = np.cumsum([0, *(size for _, size in permanent)])
    return Reading(
        name=name,
        source=source,
        update=update,
        metrics=metrics,
        games=games,
        decisions_per_game=decisions / games,
        headers_mb=header_totals[counts] / 1e6,
        blocks=counts.astype(float),
        permanent_endpoints=endpoints,
    )


def median25(values: FloatArray) -> FloatArray:
    # Centered 25-update median is descriptive smoothing, never an independent sample.
    return np.array(
        [np.median(values[max(0, i - 12) : i + 13]) for i in range(len(values))]
    )


def windows(run: Reading) -> list[Window]:
    values = {
        **{label: run.metrics[key] for label, key in PHASES.items()},
        "RSS GB": run.metrics["coordinates/rss_bytes"] / 1e9,
        "Rows": run.metrics["rows"],
        "Games": run.games,
        "Native decisions/game": run.decisions_per_game,
        "Host load": run.metrics["coordinates/host_load_1m"],
        "Permanent header MB": run.headers_mb,
        "Permanent blocks": run.blocks,
    }
    result = []
    for first, last in WINDOWS:
        mask = (run.update >= first) & (run.update <= last)
        if mask.any():
            result.append(
                Window(
                    first,
                    last,
                    int(mask.sum()),
                    {k: float(np.median(v[mask])) for k, v in values.items()},
                    float(np.mean(values["Collection"][mask])),
                    float(np.quantile(values["Collection"][mask], 0.9)),
                )
            )
    return result


def slopes(run: Reading, end: int) -> dict[str, float]:
    mask = (run.update >= 51) & (run.update <= end)
    series = {
        **{label: run.metrics[key] for label, key in PHASES.items()},
        "RSS GB": run.metrics["coordinates/rss_bytes"] / 1e9,
        "Collection seconds/1000 rows": run.metrics["time/collection"]
        / run.metrics["rows"]
        * 1000,
    }
    return {
        key: float(np.polyfit(run.update[mask] / 100, value[mask], 1)[0])
        for key, value in series.items()
    }


def fold_alignment(run: Reading, end: int) -> dict[str, FoldStatistics]:
    """Associate updates with nearest permanent block endpoint, not wall fold time."""
    groups: dict[int, list[float]] = {offset: [] for offset in range(-3, 5)}
    for update, value in zip(run.update, run.metrics["time/collection"], strict=True):
        if 300 <= update <= end:
            nearest = min(run.permanent_endpoints, key=lambda e: abs(e - update))
            offset = int(update - nearest)
            if offset in groups:
                groups[offset].append(float(value))
    return {
        str(offset): FoldStatistics(
            count=len(values),
            mean_seconds=float(np.mean(values)),
            median_seconds=float(np.median(values)),
        )
        for offset, values in groups.items()
        if values
    }


def report(data: Path, out: Path) -> None:
    # A fresh output directory keeps reproduction separate from frozen evidence.
    out.mkdir(parents=True)
    runs = [
        reading(name, Capture.model_validate_json((data / f"{name}.json").read_bytes()))
        for name in ("w64", "w160")
    ]
    common = int(min(r.update[-1] for r in runs))
    summary = {}
    for run in runs:
        endpoints = {}
        for end in (500, 600, 650, 700, 800, 900):
            row = next((r for r in run.source.updates if r.update == end), None)
            if row is not None:
                hours = (row.wall_ms - run.source.attempts[0].started_ms) / 3600000
                endpoints[str(end)] = {
                    "historical_elapsed_hours": hours,
                    "two_equal_duration_arms_at_0.59": 2 * hours * 0.59,
                }
        summary[run.name] = {
            "run": run.source.run,
            "last_update": int(run.update[-1]),
            "windows": [asdict(w) for w in windows(run)],
            "slopes_per_100_updates_51_to_common": slopes(run, common),
            "common_last_update": common,
            "horizon_cost_sensitivity": endpoints,
            "fold_offsets_updates_300_to_common": fold_alignment(run, common),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    colors = ["#176b89", "#ad4b25"]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for ax, (label, key) in zip(axes.flat, PHASES.items(), strict=True):
        for run, color in zip(runs, colors, strict=True):
            y = run.metrics[key]
            ax.plot(run.update, y, color=color, alpha=0.13, linewidth=0.5)
            ax.plot(run.update, median25(y), color=color, label=run.name)
        ax.set(xlabel="Completed update", ylabel="Seconds/update", title=label)
        ax.legend()
    fig.suptitle(
        "Historical collection phases — raw values and centered 25-update medians\nFrozen source 6cd54d349; independent L4 hosts, seed 21501"
    )
    fig.savefig(out / "phases.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for run, color in zip(runs, colors, strict=True):
        collection = run.metrics["time/collection"]
        rss = run.metrics["coordinates/rss_bytes"] / 1e9
        axes[0, 0].plot(run.update, rss, color=color, alpha=0.5, label=run.name)
        axes[0, 1].plot(run.update, run.headers_mb, color=color, label=run.name)
        axes[1, 0].scatter(
            run.blocks, collection, color=color, s=4, alpha=0.25, label=run.name
        )
        axes[1, 1].scatter(
            run.headers_mb, rss, color=color, s=4, alpha=0.25, label=run.name
        )
        axes[2, 0].plot(
            run.update, median25(run.decisions_per_game), color=color, label=run.name
        )
        axes[2, 1].scatter(
            run.decisions_per_game,
            collection,
            color=color,
            s=4,
            alpha=0.25,
            label=run.name,
        )
    specs = [
        ("Completed update", "RSS (decimal GB)", "Both runs accumulate host memory"),
        (
            "Completed update",
            "Permanent header bytes (MB)",
            "Reconstructed index lower bound; block-end coordinate",
        ),
        (
            "Permanent blocks (reconstructed)",
            "Collection seconds/update",
            "Collection vs permanent block count",
        ),
        (
            "Permanent header bytes (MB)",
            "RSS (decimal GB)",
            "Memory vs reconstructed index growth",
        ),
        (
            "Completed update",
            "Native decisions / completed game",
            "Game length proxy — 25-update median",
        ),
        (
            "Native decisions / completed game",
            "Collection seconds/update",
            "Collection vs game length proxy",
        ),
    ]
    for ax, (x, y, title) in zip(axes.flat, specs, strict=True):
        ax.set(xlabel=x, ylabel=y, title=title)
        ax.legend()
    fig.suptitle(
        "Historical growth and completed work\nHeader sums omit manifest overviews/latest values; historical fold times are unavailable"
    )
    fig.savefig(out / "growth.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                name: {
                    "last_update": values["last_update"],
                    "slopes": values["slopes_per_100_updates_51_to_common"],
                }
                for name, values in summary.items()
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.data, args.out)


if __name__ == "__main__":
    main()
