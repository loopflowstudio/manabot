"""Saved-row on-minus-off contrasts for the complete six-run history cohort.

Four legs stay together. Paired seed/deal bootstrap draws describe this small
cohort; cost/exposure selections use only the last checkpoint already observed.
Exposure is post-treatment accounting, not a causal control.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from experiments.runners import history_input
from experiments.runners.screen_spec import ScreenSpec, specification
from manabot.training.analysis import cost_comparison, verify_saved_inputs
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun

ANCHOR = "scripted-greedy-fixed-anchor"


class _Latency(BaseModel):
    count: int = Field(ge=0)
    seconds: float = Field(ge=0)


class _Row(BaseModel):
    deal_seed: int
    leg: Literal[0, 1, 2, 3]
    score_a: float = Field(ge=0, le=1)
    player_a: str
    failure: None
    terminated: Literal[True]
    truncated: Literal[False]
    replay_passed: Literal[True]
    latency: dict[str, _Latency]


class _Cell(BaseModel):
    a: str
    b: Literal["scripted-greedy-fixed-anchor"]
    training_seed: int
    cutoff: Literal[0, 1]
    phase: Literal["development"]
    variant: Literal["raw"]
    scheduled_games: Literal[100]
    rows: list[_Row]


class Effect(BaseModel):
    arm_seed_scores: list[list[float]]
    seed_effects: list[float]
    mean_effect: float
    paired_seed_common_deal_95_percentile: list[float]


def paired_effect(values: NDArray[np.float64]) -> Effect:
    """values[arm, seed, deal] contains the mean of each four-leg block."""
    if values.shape != (2, 3, 25) or not np.isfinite(values).all():
        raise ValueError("history contrast requires complete finite 2×3×25 values")
    differences = values[1] - values[0]
    rng = np.random.default_rng(10634)
    draws = np.empty(10000)
    for index in range(len(draws)):
        seeds = rng.integers(0, 3, 3)
        deals = rng.integers(0, 25, 25)
        draws[index] = differences[seeds][:, deals].mean()
    return Effect(
        arm_seed_scores=values.mean(axis=2).tolist(),
        seed_effects=differences.mean(axis=1).tolist(),
        mean_effect=float(differences.mean()),
        paired_seed_common_deal_95_percentile=np.quantile(
            draws, [0.025, 0.975]
        ).tolist(),
    )


def _values(
    cells: list[_Cell],
    cutoffs: dict[tuple[str, int], int],
    spec: ScreenSpec = history_input,
) -> NDArray[np.float64]:
    values = np.empty((2, 3, 25), dtype=np.float64)
    for arm, name in enumerate(spec.ARMS):
        for seed_index, seed in enumerate(spec.SEEDS):
            matches = [
                c
                for c in cells
                if (c.a, c.training_seed, c.cutoff) == (name, seed, cutoffs[name, seed])
            ]
            if len(matches) != 1:
                raise ValueError("missing or duplicate history cell")
            rows = matches[0].rows
            if len(rows) != 100 or {(r.deal_seed, r.leg) for r in rows} != {
                (d, leg) for d in spec.DEALS for leg in range(4)
            }:
                raise ValueError(
                    "history cell requires each frozen deal/leg exactly once"
                )
            for deal_index, deal in enumerate(spec.DEALS):
                values[arm, seed_index, deal_index] = (
                    sum(r.score_a for r in rows if r.deal_seed == deal) / 4
                )
    return values


def history_report(out: Path, *, study: str = "history-input") -> None:
    spec = specification(study)
    # Heterogeneous study exports are the existing boundary; typed cells narrow
    # all score, latency and schedule inputs before computing contrasts.
    data = json.loads((out / "study.json").read_text())
    if data["study"] != study:
        raise ValueError("requires history-input evidence")
    verify_saved_inputs(out, data)
    result: dict[str, object] = {
        "status": "unavailable",
        "disposition": "unresolved",
        "reason": "incomplete cohort; no complete paired effect or selection",
        "attempted_cells": len(data["comparisons"]),
    }
    if data["status"] != "completed":
        atomic_json(out / "history-contrast.json", result)
        (out / "history-contrast.md").write_text(
            "# History input\n\nIncomplete cohort: paired effects and disposition unavailable. Retain every attempted run and game.\n"
        )
        return
    cells = [_Cell.model_validate(c) for c in data["comparisons"]]
    if len(cells) != 12:
        raise ValueError("history contrast requires twelve cells")
    checkpoints = [
        paired_effect(
            _values(
                cells,
                {(arm, seed): cutoff for arm in spec.ARMS for seed in spec.SEEDS},
                spec,
            )
        )
        for cutoff in range(2)
    ]
    runs = [
        TrainingRun.model_validate_json(Path(item["path"]).read_text())
        for item in data["runs"]
    ]
    if (
        len(runs) != 6
        or {(r.regime.id, r.seed) for r in runs}
        != {(a, s) for a in spec.ARMS for s in spec.SEEDS}
        or any(r.status != "completed" for r in runs)
    ):
        raise ValueError("history contrast requires all six unique completed runs")
    measurements = data["measurements"]
    expected = {(a, s, c) for a in spec.ARMS for s in spec.SEEDS for c in range(2)}
    if (
        len(measurements) != 12
        or {(m["regime"], m["seed"], m["cutoff"]) for m in measurements} != expected
        or any(
            not m["complete"] or m["opponent"] != ANCHOR or m.get("variant") != "raw"
            for m in measurements
        )
    ):
        raise ValueError("history measurements are incomplete or duplicated")
    comparisons: dict[str, object] = {}
    for axis in (
        "training_seconds",
        "learner_transitions",
        "decisions",
        "optimizer_exposures",
    ):
        # Reuse the existing integrator with the declared coordinate, retaining
        # its earlier-only selection and common observed-support contract.
        projected = [{**m, "training_seconds": m[axis]} for m in measurements]
        comparison = cost_comparison(projected, ANCHOR)
        if comparison["status"] == "available":
            selections: dict[tuple[str, int], int] = {}
            for selected in comparison["rows"]:
                matches = [
                    m
                    for m in measurements
                    if m["regime"] == selected["regime"]
                    and m["seed"] == selected["seed"]
                    and m["checkpoint"]["sha256"] == selected["checkpoint"]["sha256"]
                ]
                if len(matches) != 1:
                    raise ValueError("ambiguous selected history checkpoint")
                selections[selected["regime"], selected["seed"]] = matches[0]["cutoff"]
            comparison["effect"] = paired_effect(
                _values(cells, selections, spec)
            ).model_dump()
        comparisons[axis] = comparison
    latency = []
    for arm in spec.ARMS:
        observations = [
            r.latency[r.player_a] for c in cells if c.a == arm for r in c.rows
        ]
        count = sum(r.count for r in observations)
        latency.append(sum(r.seconds for r in observations) / count if count else None)
    ratio = (
        latency[1] / latency[0]
        if all(v is not None and v > 0 for v in latency)
        else None
    )
    endpoint = checkpoints[1]
    cost = comparisons["training_seconds"]
    disposition = "unresolved"
    if cost["status"] == "available":
        if (
            endpoint.mean_effect >= 0.05
            and min(endpoint.seed_effects) >= 0
            and cost["effect"]["mean_effect"] > 0
            and ratio is not None
            and ratio <= 1.25
        ):
            disposition = "larger-confirmatory-allocation-merited"
        elif max(endpoint.seed_effects) < 0 and endpoint.mean_effect <= -0.05:
            disposition = "reject-for-this-recipe-and-budget"
    result = {
        "status": "completed",
        "arms": spec.ARMS,
        "seeds": spec.SEEDS,
        "checkpoints": [c.model_dump() for c in checkpoints],
        "comparisons": comparisons,
        "inference_seconds_per_decision": latency,
        "on_off_latency_ratio": ratio,
        "disposition": disposition,
        "bootstrap_seed": 10634,
        "bootstrap_draws": 10000,
        "limit": "Additional information and 22,912 parameters change the coupled self-play recipe. Fixed-update endpoints are not cost-matched; two checkpoints cannot resolve learning-speed dynamics. Exposure comparisons are post-treatment accounting. Three paired seeds and one anchor are exploratory, not equivalence, general strength or default promotion.",
    }
    if study == "depth-screen":
        result["limit"] = (
            "Depth 2 minus depth 1 with scalar value-token and no history. Same-update endpoints are not equal elapsed cost; two observations cannot resolve learning-speed dynamics. Three paired seeds against greedy are a bounded screen, not full-budget strength or model promotion. EMA is retained, not scored."
        )
    atomic_json(out / "history-contrast.json", result)
    atomic_json(
        out / "history-diagnostics.json",
        [
            {
                "recipe": r.regime.id,
                "seed": r.seed,
                "stages": [s.model_dump(mode="json") for s in r.stages],
            }
            for r in runs
        ],
    )
    (out / "history-contrast.md").write_text(
        f"# History input\n\nEndpoint on-minus-off: {endpoint.mean_effect:.4f}; per-seed {endpoint.seed_effects}; descriptive 95% interval {endpoint.paired_seed_common_deal_95_percentile}.\n\nCommon-cost comparison: {json.dumps(cost, sort_keys=True)}\n\nDisposition: {disposition}. Pooled inference on/off ratio: {ratio}.\n\n{result['limit']}\n\n[Paired effects and cost/exposure selections](history-contrast.json); [per-update diagnostics and costs](history-diagnostics.json).\n"
    )
