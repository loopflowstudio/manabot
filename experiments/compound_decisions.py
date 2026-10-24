"""Compound-decision policy training: grouping crossed with value estimator.

Arms share `direct_self_play`'s setup with compound decisions enabled in the model.

`uv run --extra notebook python -m experiments.compound_decisions --out .runs/compound`
runs the four-arm study (smoke by default; see `--help`).
"""

from typing import Literal

from manabot.arena.match import selected_match
from manabot.experiment.study import main
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK
from manabot.training.regime import Learning, Schedule, TrainCompound, TrainingRegime
import managym


def _arm(
    grouping: Literal["sequential", "grouped"],
    estimator: Literal["bootstrapped", "outcome"],
) -> TrainingRegime:
    learning = Learning(
        value_lambda=0.95,
        retained_fraction=1.0,
        learning_rate=Schedule(initial=2.5e-4),
        tau=Schedule(initial=0.01),
        epochs=1,
    )
    short = {"bootstrapped": "bootstrap", "outcome": "outcome"}[estimator]
    return TrainingRegime(
        id=f"compound-{grouping}-{short}",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(semantic_pack=PACK, compound_decisions=True),
        stages=[
            TrainCompound(
                id=f"policy-{index}",
                operation="train_compound",
                initial=f"policy-{index - 1}" if index else None,
                grouping=grouping,
                estimator=estimator,
                learning=learning.model_copy(deep=True),
            )
            for index in range(2)
        ],
        wall_seconds=900,
    )


def sequential_bootstrap() -> TrainingRegime:
    return _arm("sequential", "bootstrapped")


def grouped_bootstrap() -> TrainingRegime:
    return _arm("grouped", "bootstrapped")


def sequential_outcome() -> TrainingRegime:
    return _arm("sequential", "outcome")


def grouped_outcome() -> TrainingRegime:
    return _arm("grouped", "outcome")


# Arm order is the study's; the first arm is the paired reference.
ARMS = (
    sequential_bootstrap,
    grouped_bootstrap,
    sequential_outcome,
    grouped_outcome,
)

if __name__ == "__main__":
    main("compound-decisions", ARMS)
