"""Rebuild MTG-252 score tables from retained Etude evaluation blocks.

Each observation is one four-game deck/seat pairing. Seeds receive equal
weight; bootstrap intervals resample seeds and shared pairing blocks, keeping
each pairing's index common across seeds. Paired contrasts use the same
resampled indices. Three seeds and an exploratory extension limit inference.

    uv run --extra reports --extra evaluation python -m scripts.report_value_readout \
        --source DIR --out experiments/data/value-readout-252
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast

import duckdb
from etude.eval.records import Match, Model
from etude.train.read import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.value_readout_252 import DEEPER, DEEPER_THIRDS, FINALS, PARENTS, THIRDS
from manabot.training.update_metrics import MetricRow, _updates

EVALUATIONS = (
    "value-readout-252-v1-0b4f3fc61213",
    "value-readout-252-v1-2daf29f915a4",
    "value-readout-252-v1-76b16b56add5",
    "value-readout-252-v1-ef63a17c6cee",
)
SEEDS = (25201, 25202, 25203)
UPDATES = (20, 40, 75, 150, 200, 250, 300)
ARMS = (
    "masked_mean",
    "value_token",
    "attention_readout",
    "masked_mean_depth4",
    "value_token_depth4",
)
LABELS = (
    "Depth 2 · masked mean",
    "Depth 2 · value token",
    "Depth 2 · attention readout",
    "Depth 4 · masked mean",
    "Depth 4 · value token",
)


@dataclass(frozen=True)
class PairScore:
    match: str
    first: str
    second: str
    seed: int
    update: int
    deal: int
    score: float


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    seeds: tuple[float, ...]


def _identities() -> dict[str, tuple[str, int]]:
    found = {
        run: key for table in (PARENTS, FINALS, THIRDS) for key, run in table.items()
    }
    found.update(
        {
            run: (f"{arm}_depth4", seed)
            for table in (DEEPER, DEEPER_THIRDS)
            for (arm, seed), run in table.items()
        }
    )
    return found


def read_scores(source: Path) -> tuple[list[PairScore], dict[str, int]]:
    identities = _identities()
    matches: dict[str, Match] = {}
    states: Counter[str] = Counter()
    plays: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for run in EVALUATIONS:
        folder = source / run / "blocks"
        blocks = sorted(folder.glob("*/block.json"))
        if not blocks:
            raise ValueError(f"No blocks for {run}")
        for block in blocks:
            # Etude's JSON boundary is validated as Match before domain use.
            document = cast(
                dict[str, list[dict[str, object]]], json.loads(block.read_text())
            )
            for item in document["matches"]:
                match = Match.model_validate(item["match"])
                key = str(match.key)
                if key in matches:
                    raise ValueError(f"Duplicate retained match {key}")
                matches[key] = match
                states[str(item["state"])] += 1
            table = block.with_name("plays.parquet")
            with duckdb.connect() as database:
                rows = database.execute(
                    "SELECT match, seat, game, score, valid FROM read_parquet(?)",
                    [str(table)],
                ).fetchall()
            for key, seat, game, score, valid in rows:
                if valid is not True or score is None:
                    raise ValueError(
                        f"Invalid game retained in {key}; report missing/failure explicitly"
                    )
                if (seat, game) in plays[key]:
                    raise ValueError(f"Duplicate game {key}/{seat}/{game}")
                plays[key][seat, game] = float(score)
    if set(matches) != set(plays):
        raise ValueError("Match metadata and game results differ")
    found: list[PairScore] = []
    for key, match in matches.items():
        games = plays[key]
        if len(games) != 8 or {seat for seat, _ in games} != {0, 1}:
            raise ValueError(f"Incomplete four-game pairing {key}")
        for game in {game for _, game in games}:
            if not np.isclose(games[0, game] + games[1, game], 1.0):
                raise ValueError(f"Non-complementary scores {key}/{game}")
        first, second = match.players
        if not isinstance(first, Model):
            raise ValueError("First participant must be a model")
        arm, seed = identities[first.checkpoint.key.run]
        update = int(first.checkpoint.at["updates"])
        opponent = "heuristic"
        if isinstance(second, Model):
            opponent, other_seed = identities[second.checkpoint.key.run]
            if seed != other_seed or update != int(second.checkpoint.at["updates"]):
                raise ValueError("Head-to-head mismatched seed or update")
        found.append(
            PairScore(
                key,
                arm,
                opponent,
                seed,
                update,
                int(match.conditions["deal"]),
                sum(v for (seat, _), v in games.items() if seat == 0) / 4,
            )
        )
    return found, dict(states)


def _matrix(rows: list[PairScore]) -> np.ndarray:
    table = {(row.seed, row.deal): row.score for row in rows}
    deals = sorted({row.deal for row in rows})
    if len(rows) != 150 or len(table) != 150 or len(deals) != 50:
        raise ValueError("Expected three seeds by fifty distinct pairings")
    return np.array([[table[seed, deal] for deal in deals] for seed in SEEDS])


def estimate(values: np.ndarray) -> Estimate:
    """Crossed seed/pairing bootstrap, 4,000 draws; not multiplicity-adjusted."""
    if values.shape != (3, 50):
        raise ValueError("Expected three seeds by fifty shared pairings")
    rng = np.random.default_rng(252)
    seeds = rng.integers(0, 3, size=(4000, 3))
    deals = rng.integers(0, 50, size=(4000, 50))
    draws = values[seeds[:, :, None], deals[:, None, :]].mean(axis=(1, 2))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return Estimate(
        float(values.mean()),
        float(lower),
        float(upper),
        tuple(float(v) for v in values.mean(axis=1)),
    )


def training_summary(
    source: Path, out: Path, scores: dict[str, dict[str, object]]
) -> None:
    """Join segments by absolute update; count each learner row/exposure once.

    Work time sums recorded collection and learning durations, excluding setup,
    idle rental time and checkpoint export. It is not GPU kernel time. Host RSS
    is the maximum sampled process RSS across the run's executed segments.
    """
    series: dict[tuple[str, int], dict[int, MetricRow]] = defaultdict(dict)
    peaks: dict[tuple[str, int], int] = defaultdict(int)
    retained: dict[str, dict[str, object]] = {}
    keep = (
        "rows",
        "optimizer_exposures",
        "actor_exposures",
        "critic_exposures",
        "value_scale/explained_variance",
        "value_scale/advantage_std",
        "value_scale/floor_binds",
        "numerical/empty_actor_batch",
        "entropy",
        "collection_kl",
        "reference_kl",
        "time/collection",
        "time/learning",
    )
    for table, deep in (
        (PARENTS, False),
        (FINALS, False),
        (THIRDS, False),
        (DEEPER, True),
        (DEEPER_THIRDS, True),
    ):
        for (arm, seed), run in table.items():
            key = (f"{arm}_depth4" if deep else arm, seed)
            rows = _updates(read(source, "value_readout_252", run))
            selected = {
                update: {name: row[name] for name in keep if name in row}
                for update, row in rows.items()
            }
            series[key].update(selected)
            records = sorted((source / run / "attempts").glob("*/training/run.json"))
            if len(records) != 1:
                raise ValueError(f"Expected one training record for {run}")
            document = cast(dict[str, object], json.loads(records[0].read_text()))
            stages = cast(list[dict[str, object]], document["stages"])
            peaks[key] = max(
                peaks[key], int(cast(int, stages[0]["sampled_peak_rss_bytes"]))
            )
            retained[run] = {"arm": key[0], "seed": seed, "updates": selected}
    summary: dict[str, object] = {}
    clocks: dict[tuple[str, int], dict[str, list[float]]] = {}
    for (arm, seed), rows in series.items():
        if set(rows) != set(range(1, 301)):
            raise ValueError(f"Incomplete update history for {arm}/{seed}")
        ordered = [rows[u] for u in range(1, 301)]

        def numbers(name: str) -> np.ndarray:
            return np.array([float(row[name]) for row in ordered])

        clocks[arm, seed] = {
            "transitions": np.cumsum(numbers("rows")).tolist(),
            "exposures": np.cumsum(numbers("optimizer_exposures")).tolist(),
            "work_hours": (
                np.cumsum(numbers("time/collection") + numbers("time/learning")) / 3600
            ).tolist(),
        }
        summary[f"{arm}|{seed}"] = {
            "transitions": clocks[arm, seed]["transitions"][-1],
            "optimizer_exposures": clocks[arm, seed]["exposures"][-1],
            "actor_exposures": float(numbers("actor_exposures").sum()),
            "critic_exposures": float(numbers("critic_exposures").sum()),
            "work_hours": clocks[arm, seed]["work_hours"][-1],
            "peak_host_rss_gib": peaks[arm, seed] / 1024**3,
            "floor_binding_updates": int(numbers("value_scale/floor_binds").sum()),
            "empty_updates": int(numbers("numerical/empty_actor_batch").sum()),
            "last_75": {
                name: float(numbers(name)[225:].mean())
                for name in (
                    "value_scale/explained_variance",
                    "value_scale/advantage_std",
                    "entropy",
                    "collection_kl",
                    "reference_kl",
                )
            },
        }
    common: dict[str, object] = {}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, clock, scale, label in zip(
        axes,
        ("transitions", "exposures", "work_hours"),
        (1e6, 1e6, 1),
        (
            "Collected transitions (millions)",
            "Optimizer exposures (millions)",
            "Collection + learning (hours)",
        ),
        strict=True,
    ):
        start = max(clocks[arm, seed][clock][19] for arm in ARMS for seed in SEEDS)
        end = min(clocks[arm, seed][clock][299] for arm in ARMS for seed in SEEDS)
        grid = np.linspace(start, end, 80)
        endpoints = {}
        for arm, title in zip(ARMS, LABELS, strict=True):
            curves = []
            for i, seed in enumerate(SEEDS):
                x = [clocks[arm, seed][clock][u - 1] for u in UPDATES]
                y = [
                    cast(list[float], scores[f"{arm}|heuristic|{u}"]["seeds"])[i]
                    for u in UPDATES
                ]
                curves.append(np.interp(grid, x, y))
            mean = np.mean(curves, axis=0)
            endpoints[arm] = float(mean[-1])
            ax.plot(grid / scale, mean * 100, label=title)
        ax.set_xlabel(label)
        common[clock] = {
            "start": start,
            "end": end,
            "interpolated_endpoint_scores": endpoints,
        }
    axes[0].set_ylabel("Score against heuristic (%)")
    axes[1].set_title(
        "Matched budgets · linear interpolation between evaluated checkpoints"
    )
    axes[2].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(out / "matched-budgets.png", dpi=160)
    plt.close(figure)
    (out / "training-summary.json").write_text(
        json.dumps({"runs": summary, "common_support": common}, indent=2) + "\n"
    )
    (out / "training-metrics.json").write_text(json.dumps(retained, indent=2) + "\n")
    print("Training summary written for", len(summary), "arm/seed histories")


def build(source: Path, out: Path) -> None:
    rows, states = read_scores(source)
    grouped: dict[tuple[str, str, int], list[PairScore]] = defaultdict(list)
    for row in rows:
        grouped[row.first, row.second, row.update].append(row)
    matrices = {key: _matrix(group) for key, group in grouped.items()}
    scores = {
        f"{a}|{b}|{u}": asdict(estimate(matrix))
        for (a, b, u), matrix in sorted(matrices.items())
    }
    differences = {}
    contrasts = (
        ("value_token", "masked_mean"),
        ("attention_readout", "masked_mean"),
        ("masked_mean_depth4", "masked_mean"),
        ("value_token_depth4", "value_token"),
        ("value_token_depth4", "masked_mean_depth4"),
    )
    for first, reference in contrasts:
        for update in UPDATES:
            left, right = (
                matrices[first, "heuristic", update],
                matrices[reference, "heuristic", update],
            )
            differences[f"{first}|{reference}|{update}"] = asdict(
                estimate(left - right)
            )
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "matches": len(rows),
        "games": 4 * len(rows),
        "states": states,
        "evaluation_runs": EVALUATIONS,
        "scores": scores,
        "paired_differences": differences,
        "interval": "95% crossed seed/pairing bootstrap; 4000 draws, seed 252; resample seeds and shared deal blocks; exploratory, not multiplicity-adjusted",
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "pairing_scores.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n"
    )
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, ax = plt.subplots(figsize=(9, 5))
    for arm, label in zip(ARMS, LABELS, strict=True):
        points = [estimate(matrices[arm, "heuristic", update]) for update in UPDATES]
        ax.plot(
            UPDATES,
            [p.mean * 100 for p in points],
            marker="o",
            label=label,
            linewidth=2,
        )
    ax.set(
        xlabel="Training update",
        ylabel="Score against heuristic (%)",
        title="Critic readout comparison · three seeds · Elemental",
    )
    ax.axhline(50, color="#999999", linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(out / "learning-curves.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (arm, label) in enumerate(zip(ARMS, LABELS, strict=True)):
        point = estimate(matrices[arm, "heuristic", 300])
        axes[0].errorbar(
            point.mean * 100,
            i,
            xerr=[
                [100 * (point.mean - point.lower)],
                [100 * (point.upper - point.mean)],
            ],
            fmt="o",
            capsize=4,
            color=f"C{i}",
        )
        axes[0].scatter(
            np.array(point.seeds) * 100, [i] * 3, s=12, color=f"C{i}", alpha=0.5
        )
    axes[0].set(
        yticks=range(5),
        yticklabels=LABELS,
        xlabel="Score against heuristic (%)",
        title="Update 300 · means, seeds and 95% intervals",
    )
    axes[0].invert_yaxis()
    axes[0].axvline(50, color="#999999", linestyle="--", linewidth=0.8)
    comparisons = (
        ("masked_mean_depth4", "masked_mean", "Depth 4 mean − depth 2 mean"),
        ("value_token_depth4", "value_token", "Depth 4 token − depth 2 token"),
        ("value_token_depth4", "masked_mean_depth4", "Depth 4 token − depth 4 mean"),
        ("value_token", "masked_mean", "Depth 2 token − depth 2 mean"),
        ("attention_readout", "masked_mean", "Depth 2 readout − depth 2 mean"),
    )
    for i, (arm, reference, _) in enumerate(comparisons):
        point = estimate(
            matrices[arm, "heuristic", 300] - matrices[reference, "heuristic", 300]
        )
        axes[1].errorbar(
            point.mean * 100,
            i,
            xerr=[
                [100 * (point.mean - point.lower)],
                [100 * (point.upper - point.mean)],
            ],
            fmt="o",
            capsize=4,
            color="#303c4c",
        )
    axes[1].set(
        yticks=range(5),
        yticklabels=[label for _, _, label in comparisons],
        xlabel="Paired score difference (percentage points)",
        title="Which changes help? · update 300",
    )
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(out / "endpoint-comparisons.png", dpi=160)
    plt.close(figure)
    print(
        json.dumps(
            {
                "matches": len(rows),
                "states": states,
                "endpoint": {
                    key: value for key, value in scores.items() if key.endswith("|300")
                },
                "differences": {
                    key: value
                    for key, value in differences.items()
                    if key.endswith("|300")
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    build(args.source, args.out)
    if args.metrics is not None:
        result = json.loads((args.out / "results.json").read_text())
        training_summary(args.metrics, args.out, result["scores"])


if __name__ == "__main__":
    main()
