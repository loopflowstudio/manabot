"""Bounded local smoke of the same learner and checkpoint path as baseline."""

from typing import Literal

from etude.train.records import Step
from etude.train.run import track
from pydantic import Field

from manabot.recipes.baseline import BaselineConfig, settings
from manabot.training.loops.self_play import self_play

LOCAL_STEPS = 2
LOCAL_SECONDS = 120.0


class LocalConfig(BaselineConfig):
    updates: int = Field(default=LOCAL_STEPS, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"


def train(config: LocalConfig) -> None:
    inputs = settings(config).model_copy(update={"streams": 2, "transitions": 16})
    with track(
        clocks=[Step("updates", "optimizer updates")],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        self_play(
            inputs,
            run.out,
            seed=config.seed,
            updates=int(run.until("updates", default=config.updates)),
            run=run,
        )
