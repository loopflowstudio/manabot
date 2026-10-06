"""Typed specifications for the shared paired-screen process supervisor."""

from typing import Protocol

from experiments.runners import history_input
from experiments.runners.history_input import Calibration, InputBinding
from manabot.training.models import TrainingRegime, TrainingRun


class ScreenSpec(Protocol):
    STUDY: str
    ARMS: tuple[str, str]
    SEEDS: tuple[int, int, int]
    CALIBRATION_SEED: int
    DEALS: tuple[int, ...]
    ORDER: tuple[tuple[int, int], ...]
    TOTAL_SECONDS: int
    CALIBRATION_SECONDS: int
    TRAINING_SECONDS: int
    EVALUATION_SECONDS: int
    REPORT_SECONDS: int
    RUN_SECONDS: int
    Calibration: type[Calibration]

    def recipes(
        self, updates: int = 800, *, calibration: bool = False
    ) -> list[TrainingRegime]: ...
    def runtime_bindings(
        self, values: list[TrainingRegime]
    ) -> tuple[dict[str, str], tuple[InputBinding, ...]]: ...
    def validate_run(
        self,
        run: TrainingRun,
        recipe: TrainingRegime,
        seed: int,
        binding: InputBinding,
        common: dict[str, str],
    ) -> None: ...
    def validate_plan_recipes(
        self,
        values: list[TrainingRegime],
        evidence: str,
        common: dict[str, str],
        bindings: tuple[InputBinding, ...],
    ) -> Calibration: ...


def specification(study: str) -> ScreenSpec:
    if study == "history-input":
        return history_input
    if study == "depth-screen":
        from experiments.runners import depth_screen

        return depth_screen
    raise ValueError(f"unsupported supervised screen: {study}")
