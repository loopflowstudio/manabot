"""Bounded local smoke of the same learner and checkpoint path as baseline."""

from typing import Literal

from etude.report import Report
from etude.train.tracking import track
from pydantic import Field

from manabot.recipes.baseline import BaselineConfig, settings
from manabot.training.loops.self_play import CLOCKS, self_play
from manabot.training.panels import self_play_report


class LocalConfig(BaselineConfig):
    updates: int = Field(default=2, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"


def report() -> Report:
    return self_play_report()


def train(config: LocalConfig) -> None:
    from manabot.backends import resolve

    inputs = settings(config).model_copy(update={"streams": 2, "transitions": 16})
    # Resolve before opening the run: a missing framework leaves no failed run.
    backend = resolve(config.backend)
    with track(clocks=CLOCKS, config=config) as run:
        self_play(
            inputs,
            run,
            updates=config.updates,
            kind="smoke",
            recipe="manabot.recipes.local",
            backend=backend,
        )
        games = run.out / "games"
        if any(games.glob("*.jsonl")):
            print(
                f"games: {games} (open one with: "
                f"uv run --no-project managym academy replay {games} --game 0)"
            )
