"""Collect histories and fit a sampler from a linked policy run.

Each attempt restarts collection and fitting. Etude checks and carries the
immutable policy input, and records the resulting sampler checkpoint.
"""

from decimal import Decimal

from etude.experiment import observe
from etude.jobs import training
from etude.spec import Job
from etude.storage import default_root
from etude.train.tracking import track
from pydantic import BaseModel, Field

from manabot.belief.collect import collect_belief
from manabot.belief.train import SamplerConfig, train_belief
from manabot.recipes import belief_policy


class BeliefConfig(BaseModel):
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def train(config: BeliefConfig) -> None:
    with track(clocks=["steps"], config=config) as run:
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


def runs() -> Job:
    """Fit from the completed local policy; run belief_policy.py first."""
    policy = observe(default_root(), "belief_policy").run("belief_policy").final
    if policy is None:
        raise ValueError("finish belief_policy before starting the sampler")
    return training(
        train,
        config=BeliefConfig(),
        start=policy,
        max_hours=1,
        max_usd=Decimal(1),
    )
