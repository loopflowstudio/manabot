"""Bounded local smoke of the same learner and checkpoint path as baseline."""

from typing import Literal

from etude.train.run import track
from pydantic import Field

from manabot.recipes.baseline import BaselineConfig, settings
from manabot.training.loops.self_play import CLOCKS, self_play


class LocalConfig(BaselineConfig):
    updates: int = Field(default=2, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"


def train(config: LocalConfig) -> None:
    inputs = settings(config).model_copy(update={"streams": 2, "transitions": 16})
    with track(clocks=CLOCKS, config=config) as run:
        self_play(inputs, run, updates=config.updates)
