"""Belief sampling in two stages: train a policy, then fit a sampler from its games.

This file is the experiment `belief` for Etude, and the template for a recipe
whose later stage starts from an earlier run's checkpoint. `training_jobs(state)`
always declares the job `policy` (`manabot.recipes.belief_policy`); once the
experiment's stored state shows that run finished, it also declares `sampler`
(`manabot.recipes.belief_sampler`) with `start=` the policy's final checkpoint.
Etude delivers that checkpoint to the sampler run as its input `start` and
records the lineage.

    uv run etude train experiments/belief.py --job policy
    uv run etude train experiments/belief.py --job sampler

Under `etude launch experiments/belief.py`, the first pass launches the policy
and a later pass, typed again once it has finished, launches the sampler.
"""

from decimal import Decimal

from etude.jobs import ExperimentState, TrainingJob

from manabot.recipes import belief_policy, belief_sampler

POLICY = "policy"
SAMPLER = "sampler"


def training_jobs(state: ExperimentState) -> list[TrainingJob]:
    """The policy job, and the sampler job once a policy run has finished."""
    jobs: list[TrainingJob] = [
        TrainingJob(
            name=POLICY,
            train=belief_policy.train,
            config=belief_policy.PolicyConfig(),
            max_hours=1,
            max_usd=Decimal(1),
        )
    ]
    # `state.run(name)` raises for a run not stored yet, so select by job.
    policy = next(
        (run.final for run in state.runs if run.launch == POLICY and run.final), None
    )
    if policy is not None:
        jobs.append(
            TrainingJob(
                name=SAMPLER,
                train=belief_sampler.train,
                config=belief_sampler.BeliefConfig(),
                start=policy,
                max_hours=1,
                max_usd=Decimal(1),
            )
        )
    return jobs
