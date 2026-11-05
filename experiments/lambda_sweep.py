"""Does more direct game-result credit help on complete games? (Task fc6ffb)

Two-seat complete-game recipe, lambda alone varying above the current .5/.8. At
twenty decisions from the end those traces give the result weight about 1e-6 and
.012. Protocol: experiments/protocols/closure-lambda-v1.md. Recipe:
`experiments.batched_baseline.lambda_sweep`. Critic quality by distance from game
end is each update's `selection_groups` metrics, under the report's ML details.

    uv run etude eval experiments/lambda_sweep.py --rent SUPPLY
    uv run etude report experiments/lambda_sweep.py --source r2://manabot/jobs --open
"""

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments.batched_baseline import lambda_sweep
from experiments.learning_signal import evaluation_jobs, learning_report, training_runs

EXPERIMENT = "lambda_sweep"
SEEDS = (65201, 65202, 65203)
UPDATES = 100
# (policy, value). The first is the current recipe and the baseline.
ARMS: tuple[tuple[float, float], ...] = (
    (0.5, 0.8),
    (0.8, 0.8),
    (0.5, 0.95),
    (0.8, 0.95),
    (0.95, 1.0),
)
BASELINE = "lambda-sweep-p0.5-v0.8-v1"


def runs(
    revision: str,
    hours: float = 1.2,
    usd_per_run: float = 0.72,
    updates: int = UPDATES,
    seeds: tuple[int, ...] = SEEDS,
    experiment: str = EXPERIMENT,
) -> list[Run]:
    return training_runs(
        (lambda_sweep(*arm, updates) for arm in ARMS),
        updates=updates,
        revision=revision,
        experiment=experiment,
        seeds=seeds,
        hours=hours,
        usd_per_run=usd_per_run,
    )


def eval(state: ExperimentState) -> list[Job]:
    return evaluation_jobs(state, "lambda-sweep-v1", hours=0.75, usd=0.96)


def report() -> Report:
    return learning_report(BASELINE)
