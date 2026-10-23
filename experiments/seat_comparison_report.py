"""Summarize scored seat-comparison checkpoints on both decision clocks.

Reads the follower's ledger and arena rows. A checkpoint's score is its mean
against the heuristic over the fixed deals (win 1, draw 0.5). Per milestone:
each arm's mean over seeds and the seed spread; per seed, the two-seat minus
one-seat difference at matched learner decisions with a deal-resampled 95%
interval. On the environment clock, one-seat's score at two-seat's environment
decisions is interpolated linearly in log environment decisions between its
scored checkpoints; the hypothesis predicts a zero difference there too.

    uv run python -m experiments.seat_comparison_report --directory DIR
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from experiments.seat_comparison import SEEDS
from experiments.seat_comparison_follow import MILESTONES

REPLICATES = 4000


def _by_deal(evaluation: Path) -> dict[int, float]:
    """Checkpoint's mean score per deal over its eight legs."""
    games: dict[int, list[float]] = defaultdict(list)
    for rows in evaluation.glob("arena/*/rows.json"):
        for row in json.loads(rows.read_text()):
            mine = row["player_a"] == "monitor-checkpoint"
            games[row["deal_seed"]].append(
                row["score_a"] if mine else 1 - row["score_a"]
            )
    return {deal: sum(v) / len(v) for deal, v in games.items()}


def _mean(scores: dict[int, float]) -> float:
    return sum(scores.values()) / len(scores)


def _interval(diff: dict[int, float]) -> tuple[float, float]:
    rng = random.Random(17)
    deals = sorted(diff)
    draws = sorted(
        sum(diff[rng.choice(deals)] for _ in deals) / len(deals)
        for _ in range(REPLICATES)
    )
    return draws[int(0.025 * REPLICATES)], draws[int(0.975 * REPLICATES)]


def _interpolate(
    points: list[tuple[float, dict[int, float]]], x: float
) -> dict[int, float] | None:
    """Per-deal scores linearly interpolated in log x; None outside the scored range."""
    points = sorted(points)
    for (x0, s0), (x1, s1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            w = (
                0.0
                if x1 == x0
                else (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            )
            return {d: (1 - w) * s0[d] + w * s1[d] for d in s0 if d in s1}
    return None


def report(directory: Path) -> str:
    ledger = json.loads((directory / "ledger.json").read_text())
    done = {k: v for k, v in ledger["attempts"].items() if v["phase"] == "completed"}
    scored: dict[tuple[int, str, str], dict] = {}
    for key, item in done.items():
        scores = _by_deal(directory / "evaluations" / key)
        if key.endswith("-initial"):
            seed = int(key.split("-")[0])
            for arm in ("one", "both"):
                scored[(seed, arm, "0")] = dict(
                    item, learner_decisions=0, environment_decisions=0, scores=scores
                )
            continue
        seed, arm = int(item["job"].split("-")[-2]), item["job"].split("-")[-1]
        scored[(seed, arm, item["label"])] = dict(item, scores=scores)

    lines = [
        "Score vs heuristic (mean over seeds; seed SD; seeds scored)"
        " and mean training hours to the scored checkpoint",
        "",
    ]
    lines.append(
        f"{'milestone':>9} | {'one-seat':>18} | {'two-seat':>18} | {'one h':>7} | {'two h':>7}"
        " | two minus one, per seed [95% deal interval]"
    )
    for label in ["0", *(f"{m // 1_000_000}M" for m in MILESTONES)]:
        cells = []
        for arm in ("one", "both"):
            values = [
                _mean(scored[(s, arm, label)]["scores"])
                for s in SEEDS
                if (s, arm, label) in scored
            ]
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            cells.append(
                f"{statistics.mean(values):.3f} ±{sd:.3f} n={len(values)}"
                if values
                else "—"
            )
        for arm in ("one", "both"):
            hours = [
                scored[(s, arm, label)]["training_seconds"] / 3600
                for s in SEEDS
                if (s, arm, label) in scored
            ]
            cells.append(f"{statistics.mean(hours):.2f} h" if hours else "—")
        diffs = []
        for s in SEEDS:
            a, b = scored.get((s, "both", label)), scored.get((s, "one", label))
            if a and b and label != "0":
                d = {
                    k: a["scores"][k] - b["scores"][k]
                    for k in a["scores"]
                    if k in b["scores"]
                }
                lo, hi = _interval(d)
                diffs.append(f"{_mean(d):+.3f} [{lo:+.2f},{hi:+.2f}]")
        lines.append(
            f"{label:>9} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>7} | {cells[3]:>7}"
            f" | {'  '.join(diffs)}"
        )

    lines += [
        "",
        "Matched environment decisions: two-seat minus one-seat interpolated at the same environment decisions",
    ]
    for s in SEEDS:
        one = [
            (v["environment_decisions"], v["scores"])
            for (seed, arm, lab), v in scored.items()
            if seed == s and arm == "one" and lab != "0"
        ]
        parts = []
        for (seed, arm, lab), v in sorted(
            scored.items(), key=lambda kv: kv[1]["environment_decisions"]
        ):
            if seed != s or arm != "both" or lab == "0":
                continue
            other = _interpolate(one, v["environment_decisions"])
            if other is None:
                continue
            d = {k: v["scores"][k] - other[k] for k in v["scores"] if k in other}
            lo, hi = _interval(d)
            parts.append(
                f"{v['environment_decisions'] / 1e6:.1f}M env: {_mean(d):+.3f} [{lo:+.2f},{hi:+.2f}]"
            )
        lines.append(f"  seed {s}: " + ("; ".join(parts) or "not yet comparable"))

    lines += [
        "",
        "Scored checkpoints (seed, arm, label: learner / environment decisions, games, training hours)",
    ]
    for (s, arm, lab), v in sorted(scored.items()):
        if lab != "0":
            lines.append(
                f"  {s} {arm:4} {lab:>4}: {v['learner_decisions'] / 1e6:6.2f}M / {v['environment_decisions'] / 1e6:6.2f}M,"
                f" {v['games']:,} games, {v['training_seconds'] / 3600:.2f} h, score {_mean(v['scores']):.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(report(parser.parse_args().directory.resolve()))


if __name__ == "__main__":
    main()
