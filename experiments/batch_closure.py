"""One-seat exact quota versus one-seat complete games, all else matched (Task fc6ffb).

Does stopping every stream mid-game and bootstrapping the tail cost learning, and
does finishing games cost time? Protocol: experiments/protocols/closure-lambda-v1.md.
Recipe: `experiments.batched_baseline.closure_comparison`.

    uv run etude eval experiments/batch_closure.py --rent SUPPLY
    uv run etude report experiments/batch_closure.py --source r2://manabot/jobs --open
"""

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments.batched_baseline import closure_comparison
from experiments.learning_signal import evaluation_jobs, learning_report, training_runs
from manabot.training.regime import BatchClosure

EXPERIMENT = "batch_closure"
SEEDS = (65101, 65102, 65103)
UPDATES = 200
# Closure, rental hours, dollars per run: the calibrated L4 pace of each arm.
ARMS: tuple[tuple[BatchClosure, float, float], ...] = (
    ("exact-quota", 1.9, 1.14),
    ("complete-games", 2.5, 1.5),
)
BASELINE = "batch-closure-exact-quota-v1"


def runs(
    revision: str,
    updates: int = UPDATES,
    seeds: tuple[int, ...] = SEEDS,
    experiment: str = EXPERIMENT,
) -> list[Run]:
    return [
        run
        for arm, hours, usd_per_run in ARMS
        for run in training_runs(
            (closure_comparison(arm, updates),),
            updates=updates,
            revision=revision,
            experiment=experiment,
            seeds=seeds,
            hours=hours,
            usd_per_run=usd_per_run,
        )
    ]


def eval(state: ExperimentState) -> list[Job]:
    return evaluation_jobs(state, "batch-closure-v1", hours=0.75, usd=0.96)


def report() -> Report:
    return learning_report(BASELINE)
