"""Depth-only mini screen specification; shared supervisor owns execution.

The scalar token baseline deliberately differs from the earlier WDL capacity
proposal. Timing alone fixes one update count shared by all six scientific runs.
"""

from typing import ClassVar

from experiments.runners import history_input as history
from experiments.runners.history_input import (
    InputBinding,
)
from manabot.training.models import TrainingRegime
from manabot.training.recipes import with_capacity

STUDY = "depth-screen"
SEEDS = (10341, 10342, 10343)
CALIBRATION_SEED = 10340
DEALS = tuple(range(963410, 963435))
ARMS = ("depth-1", "depth-2")
TOTAL_SECONDS = 28800
TRAINING_SECONDS = 21600
RUN_SECONDS = 3600
STAGE_SECONDS = 1790


def validate_bindings(
    common: dict[str, str], bindings: tuple[InputBinding, ...]
) -> None:
    if set(common) != history.COMMON_KEYS or any(
        len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in common.values()
    ):
        raise ValueError("depth screen requires complete common identities")
    if tuple(b.recipe_id for b in bindings) != ARMS or any(
        b.policy_history_version != 0 for b in bindings
    ):
        raise ValueError("depth screen requires ordered no-history inputs")
    if bindings[0].model_dump(exclude={"recipe_id"}) != bindings[1].model_dump(
        exclude={"recipe_id"}
    ):
        raise ValueError("depth arms must share exact input and world bindings")


class Calibration(history.Calibration):
    arm_names: ClassVar[tuple[str, str]] = ARMS
    training_seconds: ClassVar[int] = TRAINING_SECONDS
    maximum_updates: ClassVar[int] = 1600

    def check_bindings(self) -> None:
        validate_bindings(self.runtime_identities, self.input_bindings)


def recipes(updates: int = 800, *, calibration: bool = False) -> list[TrainingRegime]:
    if updates not in ((40,) if calibration else tuple(range(400, 1601, 100))):
        raise ValueError("unsupported depth screen update count")
    base = history.recipes(40 if calibration else 400, calibration=calibration)[0]
    base.wall_seconds = 400 if calibration else RUN_SECONDS
    for stage in base.stages:
        stage.updates = updates // 2
        stage.execution.wall_seconds = 190 if calibration else STAGE_SECONDS
    return [
        with_capacity(base, id=name, width=64, depth=1 if index == 0 else 2, heads=4)
        for index, name in enumerate(ARMS)
    ]


def runtime_bindings(
    values: list[TrainingRegime],
) -> tuple[dict[str, str], tuple[InputBinding, ...]]:
    return history.runtime_bindings(
        values,
        binding_validator=validate_bindings,
        protocol_path="experiments/model-capacity.md",
    )


def validate_plan_recipes(
    values: list[TrainingRegime],
    evidence: str,
    common: dict[str, str],
    bindings: tuple[InputBinding, ...],
) -> Calibration:
    receipt = Calibration.model_validate_json(evidence)
    if values != recipes(receipt.admitted_updates()):
        raise ValueError("depth screen may vary only depth and recipe ID")
    if common != receipt.runtime_identities or bindings != receipt.input_bindings:
        raise ValueError("depth calibration identity drift")
    return receipt


# Shared phase limits and run admission are owned by the existing supervisor.
CALIBRATION_SECONDS = history.CALIBRATION_SECONDS
EVALUATION_SECONDS = history.EVALUATION_SECONDS
REPORT_SECONDS = history.REPORT_SECONDS
ORDER = history.ORDER
validate_run = history.validate_run
