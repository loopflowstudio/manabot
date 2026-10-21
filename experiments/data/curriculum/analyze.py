"""Apply the curriculum's frozen analysis to merged evaluation cells.

usage: analyze.py EVALUATION_DIR COMPLEXITY_EVALUATION_DIR OUT_JSON

Primary: A − B at 64 per anchor; verdict per candidate decision anchor (the
anchor rule is applied separately once MTG-153's pairing exists). Secondary:
A − B at 32 and 128, each arm minus its frozen control, time to +10 points,
B − C (scratch with the boundary minus the complexity cohort's continuous L3
fits) for seeds 12251-12253. Missing cells leave their summary unavailable.
"""
import json
from pathlib import Path
import sys

from experiments.runners.curriculum_analysis import (
    paired_difference,
    six_seed_summary,
    time_to_progress,
    verdict,
)
from experiments.runners.curriculum_campaign import SEEDS
from manabot.training.monitor_evaluation import MonitorResult

ANCHORS = ("heuristic", "scripted_greedy", "random")


def load(path: Path) -> MonitorResult | None:
    if not (path / "monitor.json").exists():
        return None
    result = MonitorResult.model_validate_json((path / "monitor.json").read_text())
    return result if result.status == "completed" else None


def interval(summary) -> dict | None:  # type: ignore[no-untyped-def]
    if summary.interval is None:
        return None
    i = summary.interval
    return {"mean": i.mean, "lower": i.lower, "upper": i.upper, "positive_seeds": summary.positive_seeds}


def main() -> None:
    evaluation, complexity, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    report: dict[str, object] = {"missing": []}

    def cell(seed: int, name: str) -> MonitorResult | None:
        result = load(evaluation / str(seed) / name)
        if result is None:
            report["missing"].append(f"{seed}/{name}")  # type: ignore[union-attr]
        return result

    contrasts: dict[str, object] = {}
    for anchor in ANCHORS:
        for milestone in (32, 64, 128):
            effects = []
            per_seed = {}
            for seed in SEEDS:
                a = cell(seed, f"{milestone}-{anchor}-curriculum")
                b = cell(seed, f"{milestone}-{anchor}-scratch")
                if a and b:
                    d = paired_difference(a, b)
                    effects.append(d)
                    per_seed[seed] = {
                        "difference": d.difference.mean,
                        "lower": d.difference.lower,
                        "upper": d.difference.upper,
                        "cells": {f"{c.candidate_role} v {c.opponent_role}": c.improvement.mean for c in d.cells},
                        "curriculum": a.score.mean if a.score else None,
                        "scratch": b.score.mean if b.score else None,
                    }
            summary = six_seed_summary(tuple(effects), SEEDS)
            contrasts[f"{anchor}/{milestone}"] = {"seeds": per_seed, "summary": interval(summary)}
            if milestone == 64:
                contrasts[f"{anchor}/64"]["_summary_object"] = summary  # type: ignore[index]
    verdicts = {}
    random_summary = contrasts["random/64"].pop("_summary_object")  # type: ignore[union-attr]
    for anchor in ("heuristic", "scripted_greedy"):
        verdicts[anchor] = verdict(contrasts[f"{anchor}/64"].pop("_summary_object"), random_summary)  # type: ignore[union-attr]
    report["contrasts"] = contrasts
    report["verdict_by_decision_anchor"] = verdicts

    gains: dict[str, object] = {}
    progress: dict[str, object] = {}
    for anchor in ANCHORS:
        for arm in ("curriculum", "scratch"):
            series: dict[int, dict[int, float]] = {}
            for milestone in (32, 64, 128):
                effects = []
                for seed in SEEDS:
                    a = cell(seed, f"{milestone}-{anchor}-{arm}")
                    c = cell(seed, f"{milestone}-{anchor}-control")
                    if a and c:
                        d = paired_difference(a, c)
                        effects.append(d)
                        series.setdefault(seed, {})[milestone] = d.difference.mean
                gains[f"{anchor}/{arm}/{milestone}"] = interval(six_seed_summary(tuple(effects), SEEDS))
            progress[f"{anchor}/{arm}"] = {
                seed: time_to_progress(points) for seed, points in series.items()
            }
    report["gain_over_initialization"] = gains
    report["time_to_plus_10"] = progress

    boundary: dict[str, object] = {}
    for anchor in ANCHORS:
        for milestone in (64, 128):
            rows = {}
            for seed in SEEDS[:3]:
                b = cell(seed, f"{milestone}-{anchor}-scratch")
                c = load(complexity / f"l3-{seed}" / f"{milestone}-{anchor}-candidate")
                if b and c:
                    # Different runs and inference identity: report scores on
                    # identical deals, not a paired_difference identity claim.
                    rows[seed] = {"split": b.score.mean, "continuous": c.score.mean}  # type: ignore[union-attr]
            boundary[f"{anchor}/{milestone}"] = rows
    report["scratch_boundary_vs_continuous"] = boundary
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"verdicts": verdicts, "missing": len(report["missing"])}))  # type: ignore[arg-type]
    for anchor in ANCHORS:
        for milestone in (32, 64, 128):
            print(anchor, milestone, contrasts[f"{anchor}/{milestone}"]["summary"])  # type: ignore[index]


if __name__ == "__main__":
    main()
