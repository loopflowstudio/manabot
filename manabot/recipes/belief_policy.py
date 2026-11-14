"""PPO on Gray Ogre, producing the frozen policy used by the belief sampler."""

from typing import Literal

from etude.train.tracking import track
from pydantic import BaseModel, Field

from manabot.infra.config import MatchConfig
from manabot.recipes.builders import gray_ogre_match, gray_ogre_self_play
from manabot.training.loops.self_play import CLOCKS, self_play


class PolicyConfig(BaseModel):
    updates: int = Field(default=1, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"


def match() -> MatchConfig:
    return gray_ogre_match()


def train(config: PolicyConfig) -> None:
    settings = gray_ogre_self_play(horizon=config.updates).model_copy(
        update={"device": config.device}
    )
    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings, run, updates=config.updates)
