"""Does choosing mana sources one tap at a time slow early learning?

Two arms on the Elemental recipe, differing only in the native payment
decision of a cast or activation: `automatic` leaves source choice to the
automatic payer; `taps` offers each useful fixed source as a tap step beside
`PayCost`, which finishes automatically (`MANAGYM_EXPERIMENT_SOURCE_TAPS`).
Both spend pooled units automatically. `runs` trains each arm; `eval` scores
every scheduled checkpoint against the heuristic, which always takes
`PayCost`, with each arm keeping its own payment mode.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run

from experiments.elemental_variants import elemental_recipe
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    FINALIZATION_SECONDS,
    SCORED_UPDATES,
    SETUP_SECONDS,
    benchmark_jobs,
    scheduled_matchups,
)
from manabot.experiment.job import TrainingJob
from manabot.training.regime import TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[1]
Arm = Literal["automatic", "taps"]
ARMS: tuple[Arm, ...] = ("automatic", "taps")


def _in_payment_mode(job: Job, arm: Arm) -> Job:
    """Pin the arm's native payment switches in the command; one attempt."""
    return job.model_copy(
        update={
            "command": "export MANAGYM_EXPERIMENT_AUTOSPEND=1\n"
            f"export MANAGYM_EXPERIMENT_SOURCE_TAPS={int(arm == 'taps')}\n"
            + job.command,
            "max_attempts": 1,
        }
    )


def runs(
    revision: str,
    updates: int = 75,
    hours: float = 2.0,
    seeds: str = "34701,34702,34703,34704",
) -> list[Run]:
    """Each arm on the Elemental recipe, once per seed, exporting scored milestones."""
    wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    result: list[Run] = []
    for arm in ARMS:
        regime = elemental_recipe(f"source-taps-{arm}", updates)
        regime.wall_seconds = wall_seconds
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.execution.wall_seconds = wall_seconds
        job = TrainingJob(
            regime=TrainingRegime.model_validate(regime.authored_dump()),
            experiment="source_taps",
            milestones=tuple(u for u in SCORED_UPDATES if u < updates),
            finalization_seconds=FINALIZATION_SECONDS,
        ).job(repo=ROOT, revision=revision, max_hours=hours, max_usd=Decimal("3"))
        result.extend(
            _in_payment_mode(job, arm).runs(*(int(seed) for seed in seeds.split(",")))
        )
    return result


def eval(state: ExperimentState) -> list[Job]:
    """Replay-checked heuristic scores at each scheduled checkpoint, per arm."""
    jobs: list[Job] = []
    for arm in ARMS:
        matchups = tuple(
            matchup
            for run in state.runs
            if run.final is not None and f"regime:source-taps-{arm}" in run.final.tags
            for matchup in scheduled_matchups(run, ELEMENTAL_PAIRS)
        )
        jobs += [
            _in_payment_mode(job, arm)
            for job in benchmark_jobs(
                matchups, f"source-taps-{arm}", ELEMENTAL_PAIRS, hours=3, usd=5
            )
        ]
    return jobs
