"""Paired contrasts from the mirror screen's retained arena rows, without reruns."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import TypedDict

import numpy as np

from manabot.training.execution import atomic_json
from manabot.training.experiment_report import load_evidence
from manabot.training.learning_dashboard import candidate_deck, deck_results
from manabot.training.monitor_evaluation import MonitorResult


class Estimate(TypedDict):
    mean: float
    lower: float
    upper: float
    units: int


def interval(values: list[float]) -> Estimate:
    samples = np.asarray(values)
    indexes = np.random.default_rng(125).integers(
        0, len(samples), (10000, len(samples))
    )
    lo, hi = np.quantile(samples[indexes].mean(axis=1), [0.025, 0.975])
    return {
        "mean": float(samples.mean()),
        "lower": float(lo),
        "upper": float(hi),
        "units": len(values),
    }


def cells(result: MonitorResult) -> dict[str, dict[tuple[int, int], float]]:
    # This validates actual seat metadata, complete cells, registrations and replay.
    if not deck_results(result):
        return {}
    values: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for row in result.rows:
        extra = row.model_extra or {}
        seat = extra["player_a_seat"]
        cell = f"{candidate_deck(row)} vs {extra['seat_decks'][1 - seat]}"
        values[cell][(row.deal_seed, seat)] = float(row.score_a == 1)
    return dict(values)


def paired(
    left: dict[tuple[int, int], float], right: dict[tuple[int, int], float]
) -> Estimate:
    if left.keys() != right.keys():
        raise ValueError("paired contrast requires identical held-out deals and seats")
    deals = sorted({seed for seed, _ in left})
    return interval(
        [sum(left[(s, seat)] - right[(s, seat)] for seat in (0, 1)) / 2 for s in deals]
    )


def analyze(root: Path) -> str:
    evidence = load_evidence(root)
    plan_path = root / "plan.json"
    seeds = (
        tuple(json.loads(plan_path.read_text())["seeds"])
        if plan_path.exists()
        else (12551, 12552, 12553)
    )
    rows: list[dict[str, object]] = []
    diagnostic: list[dict[str, object]] = []
    runs = {r.id: r for r in evidence.runs}
    grouped: dict[tuple[str, str], list[MonitorResult]] = defaultdict(list)
    for result in evidence.monitors:
        if result.status != "completed":
            continue
        grouped[(result.run_id, result.protocol.opponent)].append(result)
        data = cells(result)
        if result.purpose == "supplemental-matchup-diagnostic":
            for deck in ("gw_allies", "ur_lessons"):
                diagnostic.append(
                    {
                        "policy_deck": deck,
                        "opponent": result.protocol.opponent,
                        "lessons_minus_allies": paired(
                            data[f"{deck} vs ur_lessons"], data[f"{deck} vs gw_allies"]
                        ),
                    }
                )
        for summary in deck_results(result):
            rows.append(
                {
                    "run": result.run_id,
                    "seed": result.training_seed,
                    "arm": runs[result.run_id].regime.id
                    if result.run_id in runs
                    else "frozen-Mini-diagnostic",
                    "updates": result.coordinates.updates,
                    "opponent": result.protocol.opponent,
                    "cell": summary.deck,
                    "wins": summary.wins,
                    "draws": summary.draws,
                    "games": summary.games,
                    "win": summary.interval.model_dump(),
                    "training_seconds": result.coordinates.training_seconds,
                    "learner_transitions": result.coordinates.learner_transitions,
                }
            )
    gains: list[dict[str, object]] = []
    endpoints: dict[tuple[int, str, str], dict[str, dict[tuple[int, int], float]]] = {}
    for (run_id, opponent), results in grouped.items():
        run = runs.get(run_id)
        if run is None or not run.regime.id.startswith("etu125-mirrors-"):
            continue
        initial = next((r for r in results if r.coordinates.updates == 0), None)
        if initial is None:
            continue
        start = cells(initial)
        for result in results:
            for cell, values in cells(result).items():
                gains.append(
                    {
                        "run": run_id,
                        "seed": run.seed,
                        "arm": run.regime.id,
                        "opponent": opponent,
                        "cell": cell,
                        "updates": result.coordinates.updates,
                        "gain_over_frozen_initialization": paired(values, start[cell]),
                    }
                )
            if result.coordinates.stage_id == "endpoint" and run.status == "completed":
                endpoints[(run.seed, run.regime.id, opponent)] = cells(result)
    common_time: list[dict[str, object]] = []
    for seed in seeds:
        for opponent in ("scripted_greedy", "random"):
            histories = [
                sorted(
                    (
                        result
                        for (run_id, anchor), results in grouped.items()
                        if anchor == opponent
                        and (run := runs.get(run_id)) is not None
                        and run.seed == seed
                        and run.regime.id == f"etu125-mirrors-{arm}"
                        for result in results
                    ),
                    key=lambda result: result.coordinates.training_seconds,
                )
                for arm in ("cross-balanced", "mirrors-balanced")
            ]
            if not all(histories):
                continue
            limit = min(
                history[-1].coordinates.training_seconds for history in histories
            )
            cutoffs = sorted(
                {
                    result.coordinates.training_seconds
                    for history in histories
                    for result in history
                    if 0 < result.coordinates.training_seconds <= limit
                }
            )
            for cutoff in cutoffs:
                available = [
                    [r for r in history if r.coordinates.training_seconds <= cutoff]
                    for history in histories
                ]
                if not all(available):
                    continue
                left, right = (history[-1] for history in available)
                left_cells, right_cells = cells(left), cells(right)
                for cell in left_cells.keys() & right_cells.keys():
                    common_time.append(
                        {
                            "seed": seed,
                            "opponent": opponent,
                            "cell": cell,
                            "recorded_training_cutoff_seconds": cutoff,
                            "cross_updates": left.coordinates.updates,
                            "mirror_updates": right.coordinates.updates,
                            "cross_checkpoint_seconds": left.coordinates.training_seconds,
                            "mirror_checkpoint_seconds": right.coordinates.training_seconds,
                            "mirror_minus_cross": paired(
                                right_cells[cell], left_cells[cell]
                            ),
                        }
                    )
    effects: list[dict[str, object]] = []
    for opponent in ("scripted_greedy", "random"):
        for cell in (
            "gw_allies vs gw_allies",
            "gw_allies vs ur_lessons",
            "ur_lessons vs gw_allies",
            "ur_lessons vs ur_lessons",
        ):
            differences: list[float] = []
            for seed in seeds:
                left = endpoints.get((seed, "etu125-mirrors-cross-balanced", opponent))
                right = endpoints.get(
                    (seed, "etu125-mirrors-mirrors-balanced", opponent)
                )
                if left is not None and right is not None:
                    differences.append(paired(right[cell], left[cell])["mean"])
            effects.append(
                {
                    "opponent": opponent,
                    "cell": cell,
                    "paired_seed_differences": differences,
                    "mirror_minus_cross": interval(differences)
                    if len(differences) == len(seeds)
                    else None,
                }
            )
    primary = next(
        e
        for e in effects
        if e["opponent"] == "scripted_greedy" and e["cell"] == "ur_lessons vs gw_allies"
    )["mirror_minus_cross"]
    protection = next(
        e
        for e in effects
        if e["opponent"] == "scripted_greedy" and e["cell"] == "gw_allies vs ur_lessons"
    )["mirror_minus_cross"]
    conclusion = "Incomplete: paired-seed endpoint effects remain unavailable."
    if isinstance(primary, dict) and isinstance(protection, dict):
        conclusion = (
            "Mirror inclusion meets the exploratory retention criterion."
            if primary["lower"] > 0 and protection["mean"] >= -0.05
            else "Mirror inclusion does not meet the exploratory retention criterion; retain the negative or unresolved result."
        )
    if len(seeds) < 3 and isinstance(primary, dict):
        conclusion = "Two-seed comparison complete: effects are descriptive; no retention decision is supported."
    allocation = json.loads((root / "allocation.json").read_text())
    elapsed = min(time.time(), allocation["deadline_unix"]) - allocation["started_unix"]
    deadline = datetime.fromtimestamp(allocation["deadline_unix"], timezone.utc)
    allocation_note = f"Allocation elapsed {elapsed / 3600:.2f} / {allocation['seconds'] / 3600:.0f} hours, including preparation and interruptions; hard stop {deadline:%Y-%m-%d %H:%M UTC}."
    atomic_json(
        root / "contrasts.json",
        {
            "allocation_elapsed_seconds": elapsed,
            "allocation": allocation,
            "cells": rows,
            "gains": gains,
            "diagnostic": diagnostic,
            "endpoint_effects": effects,
            "common_time_effects": common_time,
            "common_time_rule": "Last observed checkpoint at/before shared recorded-training cutoff; no interpolation. Sparse milestones do not resolve within-interval efficiency. Evaluator and allocation costs remain separate.",
            "conclusion": conclusion,
            "uncertainty": "Cell/gain/diagnostic intervals resample whole paired deals; endpoint effect intervals resample the admitted paired training seeds. These are distinct uncertainty levels.",
        },
    )
    return conclusion + " " + allocation_note
