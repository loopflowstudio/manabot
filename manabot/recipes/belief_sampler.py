"""A recipe with several steps: train a policy, collect its histories, fit a sampler.

Each step writes one file and returns at once when that file exists, so
running the recipe again after any failure skips the steps it finished.
"""

from pydantic import BaseModel, Field

from manabot.belief.collect import collect_belief
from manabot.belief.train import SamplerConfig, train_belief
from manabot.recipes.builders import gray_ogre_self_play
from manabot.recipes.proposed_etude import Step, track
from manabot.training.loops.self_play import SelfPlay, root_match, self_play


class BeliefConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=1, ge=1)
    self_play: SelfPlay = Field(default_factory=gray_ogre_self_play)
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def belief(config: BeliefConfig) -> None:
    with track(
        clocks=[Step("updates", "optimizer updates")],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        policy = self_play(
            config.self_play,
            run.out / "policy",
            seed=config.seed,
            updates=config.updates,
            run=run,
        )
        histories = collect_belief(
            policy,
            run.out / "histories.json",
            match=root_match(config.self_play),
            games=config.games,
            seed=config.seed,
            max_steps=config.max_steps,
        )
        sampler = train_belief(
            histories, run.out / "belief.pt", config.sampler, seed=config.seed
        )
        run.checkpoint(sampler, tags=("belief",), updates=config.updates)
