"""Paired floor effects and their interaction from complete saved factorial cells.

Seeds and common four-leg deals are resampled together across all four arms.
Diagnostics describe optimizer exposure, not counterfactual filter causality.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from experiments.runners.pooling_filter import SEEDS, recipes
from manabot.training.analysis import verify_saved_inputs
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun


class _Row(BaseModel):
    deal_seed: int
    score_a: float
    failure: None
    terminated: Literal[True]
    truncated: Literal[False]
    replay_passed: Literal[True]


class _Cell(BaseModel):
    a: str
    training_seed: int
    cutoff: int
    rows: list[_Row]


class _Diagnostic(BaseModel):
    retained: int
    optimizer_exposures: int
    actor_exposures: int = 0
    critic_exposures: int = 0
    advantage_abs_mean: float
    value_loss: float | None = None


def factorial_report(out: Path) -> None:
    data = json.loads((out / "study.json").read_text())
    verify_saved_inputs(out, data)
    if data["status"] != "completed":
        atomic_json(
            out / "factorial.json",
            {
                "status": "unavailable",
                "reason": "incomplete cohort; retain all attempts",
            },
        )
        return
    cells = [_Cell.model_validate(cell) for cell in data["comparisons"]]
    ids = [r.id for r in recipes(40, 400)]
    if len(cells) != 24:
        raise ValueError("factorial report requires all 24 cells")
    results: list[dict[str, object]] = []
    for cutoff in range(2):
        values = np.zeros((4, 3, 25), dtype=np.float64)
        for arm, name in enumerate(ids):
            for seed_index, seed in enumerate(SEEDS):
                matches = [
                    c
                    for c in cells
                    if c.a == name and c.training_seed == seed and c.cutoff == cutoff
                ]
                if len(matches) != 1 or len(matches[0].rows) != 100:
                    raise ValueError("missing or repeated paired cell")
                for deal_index, deal in enumerate(range(961160, 961185)):
                    rows = [r for r in matches[0].rows if r.deal_seed == deal]
                    if len(rows) != 4:
                        raise ValueError("requires four legs per common deal")
                    values[arm, seed_index, deal_index] = (
                        sum(r.score_a for r in rows) / 4
                    )
        effects = np.stack(
            (
                values[1] - values[0],
                values[3] - values[2],
                (values[3] - values[2]) - (values[1] - values[0]),
            )
        )
        rng = np.random.default_rng(10624)
        draws = np.empty((10000, 3))
        for index in range(len(draws)):
            seeds = rng.integers(0, 3, 3)
            deals = rng.integers(0, 25, 25)
            draws[index] = effects[:, seeds][:, :, deals].mean(axis=(1, 2))
        results.append(
            {
                "cutoff": cutoff,
                "arm_seed_scores": values.mean(axis=2).tolist(),
                "effect_order": [
                    "masked_zero_minus_001",
                    "token_zero_minus_001",
                    "token_floor_effect_minus_masked_floor_effect",
                ],
                "seed_effects": effects.mean(axis=2).tolist(),
                "mean_effects": effects.mean(axis=(1, 2)).tolist(),
                "paired_seed_common_deal_95_percentile": np.quantile(
                    draws, [0.025, 0.975], axis=0
                ).T.tolist(),
            }
        )
    exposures: list[dict[str, object]] = []
    for entry in data["runs"]:
        run = TrainingRun.model_validate_json(Path(entry["path"]).read_text())
        for cutoff in range(2):
            stages = run.stages[: cutoff + 1]
            diagnostics = [
                _Diagnostic.model_validate(d) for s in stages for d in s.diagnostics
            ]
            exposures.append(
                {
                    "regime": run.regime.id,
                    "seed": run.seed,
                    "cutoff": cutoff,
                    "training_seconds": stages[-1].cumulative_seconds,
                    "learner_transitions": sum(s.learner_transitions for s in stages),
                    "retained_rows": sum(d.retained for d in diagnostics),
                    "empty_filter_updates": sum(d.retained == 0 for d in diagnostics),
                    "actor_exposures": sum(d.actor_exposures for d in diagnostics),
                    "critic_exposures": sum(d.critic_exposures for d in diagnostics),
                    "optimizer_exposures": sum(
                        d.optimizer_exposures for d in diagnostics
                    ),
                    "mean_absolute_advantage": float(
                        np.mean([d.advantage_abs_mean for d in diagnostics])
                    ),
                    "recorded_value_losses": [d.value_loss for d in diagnostics],
                }
            )
    atomic_json(
        out / "factorial.json",
        {
            "status": "completed",
            "arms": ids,
            "seeds": SEEDS,
            "checkpoints": results,
            "exposures": exposures,
            "bootstrap_seed": 10624,
            "bootstrap_draws": 10000,
            "limit": "Three training seeds; exploratory interaction, shared policy representations; no critic-only mechanism or retrospective causal explanation.",
        },
    )
