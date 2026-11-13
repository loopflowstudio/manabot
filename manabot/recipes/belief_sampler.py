"""A recipe with several steps: train a policy, collect its histories, fit a sampler.

Each step writes one file and returns at once when that file exists, so
running the recipe again after any failure skips the steps it finished.
"""

from typing import Literal

from etude.train.records import Step
from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.belief.collect import collect_belief
from manabot.belief.train import SamplerConfig, train_belief
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import root_match, self_play


class BeliefConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=1, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def train(config: BeliefConfig) -> None:
    settings = gray_ogre_self_play(horizon=config.updates).model_copy(
        update={"device": config.device}
    )
    with track(
        clocks=[Step("updates", "collection/update iterations")],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        updates = int(run.until("updates", default=config.updates))
        policy = self_play(
            settings,
            run.out / "policy",
            seed=config.seed,
            updates=updates,
            run=run,
        )
        histories = collect_belief(
            policy,
            run.out / f"histories-{policy.stem}.json",
            match=root_match(settings),
            games=config.games,
            seed=config.seed,
            max_steps=config.max_steps,
        )
        sampler = train_belief(
            histories,
            run.out / f"belief-{policy.stem}.pt",
            config.sampler,
            seed=config.seed,
        )
        run.checkpoint(sampler, tags=("belief",), updates=updates)
