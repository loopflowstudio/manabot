"""Collect histories and fit a sampler from a linked policy run.

Each attempt restarts collection and fitting. Etude checks and carries the
immutable policy input, and records the resulting sampler checkpoint.
"""

from decimal import Decimal

from etude.jobs import training
from etude.launch.records import Job
from etude.train.records import Step
from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.belief.collect import collect_belief
from manabot.belief.train import SamplerConfig, train_belief
from manabot.recipes import belief_policy
from manabot.recipes.belief_policy import PolicyConfig


class BeliefConfig(BaseModel):
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def train(config: BeliefConfig) -> None:
    with track(clocks=[Step("steps", "sampler gradient steps")], config=config) as run:
        policy = run.input("start")  # the policy run's final checkpoint
        histories = collect_belief(
            policy,
            run.out / "histories.json",
            match=belief_policy.match(),
            games=config.games,
            seed=run.seed,
            max_steps=config.max_steps,
        )
        sampler = train_belief(
            histories, run.out / "belief.pt", config.sampler, seed=run.seed
        )
        run.checkpoint(sampler, steps=config.sampler.steps)


def runs() -> list[Job]:
    policy = training(
        belief_policy.train,
        name="policy",
        config=PolicyConfig(),
        seeds=(197,),
        max_hours=1,
        max_usd=Decimal(1),
    )
    sampler = training(
        train,
        name="sampler",
        config=BeliefConfig(),
        start=policy,
        seeds=(197,),
        max_hours=1,
        max_usd=Decimal(1),
    )
    return [policy, sampler]
