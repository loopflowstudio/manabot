"""Frozen pooling/floor factorial and cost-only calibration admission.

Calibration artifacts are separate from the three scientific seeds. Admission
uses the slowest whole-run update rate, including construction and exports; it
never reads arena scores. Existing TrainingRegime owns training and checkpoints.
"""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.arena.models import canonical_sha256
from manabot.training.models import TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (10621, 10622, 10623)
ARMS = (
    ("masked_mean", 0.01),
    ("masked_mean", 0.0),
    ("value_token", 0.01),
    ("value_token", 0.0),
)
# Three seeds cannot balance four positions exactly. Reverse the first row,
# then rotate by two; each seed still contains every arm exactly once.
ORDER = ((0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1))


class CalibrationArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    recipe_id: str
    run_path: str
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    updates: Literal[40] = 40
    seconds: float = Field(gt=0)


class Calibration(BaseModel):
    """Admission receipt; all costs count before selecting shared update counts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    arms: tuple[CalibrationArm, ...]
    seconds: float = Field(gt=0, lt=1800)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_identities: dict[str, str]
    strength_inspected: Literal[False] = False

    @model_validator(mode="after")
    def complete(self) -> "Calibration":
        if tuple(a.recipe_id for a in self.arms) != tuple(
            r.id for r in recipes(40, 400)
        ):
            raise ValueError("calibration must retain all four arms in order")
        if sum(a.seconds for a in self.arms) > self.seconds:
            raise ValueError("calibration elapsed cost omits run cost")
        return self

    def admitted_updates(self) -> int:
        rate = max(a.seconds / a.updates for a in self.arms)
        feasible = math.floor((21600 - self.seconds) / (12 * 1.25 * rate) / 100) * 100
        if feasible < 400:
            raise ValueError(
                f"minimum 400 updates cannot fit: slowest {rate:.6f} s/update; calibration {self.seconds:.3f} s; feasible {feasible}"
            )
        return min(800, feasible)

    def run_seconds(self) -> float:
        return (21600 - self.seconds) / 12 - 1


def recipes(updates: int, wall_seconds: float) -> list[TrainingRegime]:
    """Derive only the declared interventions from the immutable original plan."""
    original = json.loads(
        (ROOT / "experiments/plans/value-token-screen.json").read_text()
    )
    result: list[TrainingRegime] = []
    for pooling, floor in ARMS:
        recipe = TrainingRegime.model_validate(original["recipes"][0])
        recipe.id = f"pooling-{pooling.replace('_', '-')}-floor-{'zero' if floor == 0 else '001'}"
        recipe.agent.value_aggregation = pooling
        recipe.wall_seconds = wall_seconds
        for stage in recipe.stages:
            if not isinstance(stage, TrainSelfPlay):
                raise ValueError("original screen must contain self-play stages")
            stage.updates = updates // 2
            stage.learning.min_advantage = floor
            stage.execution.wall_seconds = wall_seconds / 2 - 10
        result.append(TrainingRegime.model_validate(recipe.model_dump()))
    return result


def validate_followup(
    recipes_to_check: list[TrainingRegime], evidence: str, identities: dict[str, str]
) -> Calibration:
    calibration = Calibration.model_validate_json(evidence)
    expected = recipes(calibration.admitted_updates(), calibration.run_seconds())
    if recipes_to_check != expected:
        raise ValueError(
            "follow-up must preserve frozen screen settings except pooling, floor and calibrated counts/watchdogs"
        )
    if identities != calibration.runtime_identities:
        raise ValueError("follow-up runtime differs from calibration")
    return calibration


def recipe_digests(values: list[TrainingRegime]) -> tuple[str, ...]:
    return tuple(canonical_sha256(r.model_dump(mode="json")) for r in values)
