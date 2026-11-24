"""PPO on Gray Ogre, producing the frozen policy used by the belief sampler."""

from etude.report import Report
from etude.train.tracking import track
from pydantic import Field

from manabot.infra.config import MatchConfig
from manabot.recipes.builders import (
    SelfPlayRecipe,
    gray_ogre_match,
    gray_ogre_self_play,
)
from manabot.training.loops.self_play import CLOCKS, self_play
from manabot.training.panels import self_play_report


class PolicyConfig(SelfPlayRecipe):
    updates: int = Field(default=1, ge=1)


def match() -> MatchConfig:
    return gray_ogre_match()


def report() -> Report:
    return self_play_report()


def train(config: PolicyConfig) -> None:
    from manabot.backends import resolve

    backend = resolve(config.backend)
    settings = gray_ogre_self_play(horizon=config.updates).model_copy(
        update={"device": config.device, "run_settings": config.run_settings}
    )
    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings, run, updates=config.updates, backend=backend)
