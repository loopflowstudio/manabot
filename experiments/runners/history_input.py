"""Admission for the bounded history-input screen, without executing a workload.

The two observation contracts deliberately differ. Common runtime identities and
per-arm input bindings are frozen independently; calibration selects counts using
costs only. TrainingRegime and ordinary checkpoint admission remain authoritative.
"""

import math
from pathlib import Path
import platform
import sys
from typing import Callable, ClassVar, Literal

import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from manabot.arena.models import canonical_sha256
from manabot.env import Match, ObservationSpace
from manabot.model.world import checkpoint_world
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.training.recipes import with_recent_events, with_value_aggregation
import managym

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (10631, 10632, 10633)
CALIBRATION_SEED = 10630
DEALS = tuple(range(961260, 961285))
ARMS = ("history-off", "history-on")
ORDER = ((0, 1), (1, 0), (0, 1))
STUDY = "history-input"
RUN_SECONDS = 2400
TOTAL_SECONDS = 21600
CALIBRATION_SECONDS = 1800
TRAINING_SECONDS = 14400
EVALUATION_SECONDS = 4500
REPORT_SECONDS = 900
COMMON_KEYS = {
    "engine_extension_sha256",
    "engine_source_sha256",
    "content_manifest_sha256",
    "action_abi_sha256",
    "matchup_sha256",
    "training_source_sha256",
    "study_source_sha256",
    "runtime_environment_sha256",
}


class InputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recipe_id: Literal["history-off", "history-on", "depth-1", "depth-2"]
    observation_abi_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_history_version: Literal[0, 1]


class CalibrationArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    recipe_id: Literal["history-off", "history-on", "depth-1", "depth-2"]
    run_path: str
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_seconds: float = Field(gt=0, le=400)
    stage_seconds: tuple[float, float]
    updates: Literal[40] = 40

    @model_validator(mode="after")
    def costs(self) -> "CalibrationArm":
        if any(not math.isfinite(s) or s <= 0 or s > 190 for s in self.stage_seconds):
            raise ValueError("calibration stage exceeds its 190-second cap")
        if sum(self.stage_seconds) > self.process_seconds:
            raise ValueError("process cost omits stage costs")
        return self


class Calibration(BaseModel):
    arm_names: ClassVar[tuple[str, str]] = ARMS
    training_seconds: ClassVar[int] = TRAINING_SECONDS
    maximum_updates: ClassVar[int] = 800

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    arms: tuple[CalibrationArm, CalibrationArm]
    seconds: float = Field(gt=0, le=CALIBRATION_SECONDS)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_identities: dict[str, str]
    input_bindings: tuple[InputBinding, InputBinding]
    projected_disk_bytes: int = Field(ge=3 * 1024**3)
    strength_inspected: Literal[False] = False

    @model_validator(mode="after")
    def complete(self) -> "Calibration":
        if tuple(a.recipe_id for a in self.arms) != self.arm_names:
            raise ValueError(
                "calibration requires both arms exactly once in off/on order"
            )
        if sum(a.process_seconds for a in self.arms) > self.seconds:
            raise ValueError("calibration omits process costs")
        self.check_bindings()
        return self

    def check_bindings(self) -> None:
        validate_bindings(self.runtime_identities, self.input_bindings)

    def rate(self) -> float:
        return max(
            max(a.process_seconds / 40, *(s / 20 for s in a.stage_seconds))
            for a in self.arms
        )

    def admitted_updates(self) -> int:
        count = min(
            self.maximum_updates,
            100 * math.floor(self.training_seconds / (6 * 1.25 * self.rate() * 100)),
        )
        if count < 400:
            raise ValueError("minimum 400 updates cannot fit the training envelope")
        return count


def recipes(updates: int = 800, *, calibration: bool = False) -> list[TrainingRegime]:
    if updates not in ((40,) if calibration else (400, 500, 600, 700, 800)):
        raise ValueError("unsupported history screen update count")
    base = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/value-model-baseline-v1.json").read_text()
    )
    base = with_value_aggregation(base, id="history-base", aggregation="value_token")
    base.wall_seconds = 400 if calibration else 2400
    for stage in base.stages:
        if not isinstance(stage, TrainSelfPlay):
            raise ValueError("history baseline must use self-play")
        stage.updates = updates // 2
        stage.execution.wall_seconds = 190 if calibration else 1190
    return [
        with_recent_events(base, id=name, enabled=bool(index))
        for index, name in enumerate(ARMS)
    ]


def validate_bindings(
    common: dict[str, str], bindings: tuple[InputBinding, ...]
) -> None:
    if set(common) != COMMON_KEYS or any(
        len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in common.values()
    ):
        raise ValueError(
            "history screen requires exact common runtime/source identities"
        )
    if tuple(b.recipe_id for b in bindings) != ARMS or tuple(
        b.policy_history_version for b in bindings
    ) != (0, 1):
        raise ValueError("history input bindings require ordered off/on contracts")
    for field in (
        "observation_abi_sha256",
        "input_schema_sha256",
        "world_binding_sha256",
    ):
        if getattr(bindings[0], field) == getattr(bindings[1], field):
            raise ValueError(
                "history inputs must retain distinct observation/schema identities"
            )


def runtime_bindings(
    values: list[TrainingRegime],
    *,
    binding_validator: Callable[
        [dict[str, str], tuple[InputBinding, ...]], None
    ] = validate_bindings,
    protocol_path: str = "experiments/history-input.md",
) -> tuple[dict[str, str], tuple[InputBinding, ...]]:
    common: dict[str, str] | None = None
    bindings: list[InputBinding] = []
    for recipe in values:
        space = ObservationSpace(recipe.observation)
        fingerprints = runtime_fingerprints(
            CALIBRATION_SEED, match_hypers=recipe.match, observation_space=space
        )
        fingerprints["training_source_sha256"] = source_bundle_sha256(
            sorted((ROOT / "manabot").rglob("*.py"))
        )
        fingerprints["study_source_sha256"] = source_bundle_sha256(
            sorted((ROOT / "experiments/runners").glob("*.py"))
            + [
                ROOT / "experiments/study/training-regimes.ipynb",
                ROOT / "uv.lock",
                ROOT / "pyproject.toml",
                ROOT / "experiments/regimes/value-model-baseline-v1.json",
                ROOT / protocol_path,
            ]
        )
        fingerprints["runtime_environment_sha256"] = canonical_sha256(
            {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "ram_bytes": psutil.virtual_memory().total,
                "extension_path": str(Path(managym._managym.__file__).resolve()),
                "device": "cpu",
                "threads": 1,
            }
        )
        current: dict[str, str] = {}
        for key in COMMON_KEYS:
            value = fingerprints[key]
            if not isinstance(value, str):
                raise TypeError(f"runtime identity {key} must be text")
            current[key] = value
        if common is not None and common != current:
            raise ValueError("history arms differ in common runtime identities")
        common = current
        world = checkpoint_world(Match(recipe.match).to_rust(), space)
        bindings.append(
            InputBinding(
                recipe_id=recipe.id,
                observation_abi_sha256=fingerprints["observation_abi_sha256"],
                input_schema_sha256=canonical_sha256(world["input_schema"]),
                world_binding_sha256=canonical_sha256(world),
                policy_history_version=recipe.observation.policy_history_version,
            )
        )
    if common is None:
        raise ValueError("missing history arms")
    binding_validator(common, tuple(bindings))
    return common, tuple(bindings)


def validate_plan_recipes(
    values: list[TrainingRegime],
    evidence: str,
    common: dict[str, str],
    bindings: tuple[InputBinding, ...],
) -> Calibration:
    receipt = Calibration.model_validate_json(evidence)
    if values != recipes(receipt.admitted_updates()):
        raise ValueError(
            "history screen may change only history input and recipe ID from frozen controls"
        )
    if common != receipt.runtime_identities or bindings != receipt.input_bindings:
        raise ValueError("history runtime/input binding differs from calibration")
    return receipt


def validate_run(
    run: TrainingRun,
    recipe: TrainingRegime,
    seed: int,
    binding: InputBinding,
    common: dict[str, str],
) -> None:
    """Reject partial fixed-count jobs, drift, and missing raw/EMA exports."""
    if run.status != "completed" or run.regime != recipe or run.seed != seed:
        raise ValueError("history run failed or differs from frozen recipe/seed")
    if (
        any(
            run.identities.get(k) != v
            for k, v in common.items()
            if k not in {"study_source_sha256", "runtime_environment_sha256"}
        )
        or run.identities.get("observation_abi_sha256")
        != binding.observation_abi_sha256
    ):
        raise ValueError("history run runtime/input identity drift")
    if len(run.stages) != 2:
        raise ValueError("history run requires two complete stages")
    for stage, spec in zip(run.stages, recipe.stages, strict=True):
        assert isinstance(spec, TrainSelfPlay)
        if (
            stage.status != "completed"
            or len(stage.diagnostics) != spec.updates
            or stage.learner_transitions != spec.updates * 256
            or set(stage.artifacts) not in ({"raw", "ema"}, {"raw", "ema", "optimizer"})
        ):
            raise ValueError("history run omitted updates, transitions or exports")
