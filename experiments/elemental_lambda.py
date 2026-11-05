"""Does more direct game-result credit help the ten-deck Elemental recipe? (Task 07c490)

The two-seat complete-game Elemental recipe, lambda alone varying above its .5/.8. At
twenty decisions from the end those traces give the result weight about 1e-6 and
.012. MTG-251 asked this on Allies/Lessons (`experiments/lambda_sweep.py`). Protocol:
experiments/protocols/closure-lambda-elemental-v1.md. Recipe:
`experiments.elemental_variants.lambda_sweep`. Critic quality by distance from game
end is each update's `selection_groups` metrics, under the report's ML details.

    uv run etude launch experiments/elemental_lambda.py --rent 07c490-l4 \\
        --cohort elemental-lambda-p0.5-v0.8 --parallel 3 --revision COMMIT --arms 0.5/0.8
    uv run etude eval experiments/elemental_lambda.py --rent 07c490-cpu32 --source r2://manabot/jobs
    uv run etude report experiments/elemental_lambda.py --source r2://manabot/jobs --open
"""

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments.elemental_variants import lambda_sweep
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    evaluation_jobs,
    learning_report,
    training_runs,
)

EXPERIMENT = "elemental_lambda"
SEEDS = "70201,70202,70203"
UPDATES = 75
HOURS = 1.15
USD_PER_RUN = 0.68
# (policy, value). The first is the baseline, the recipe's own lambdas.
ARMS: tuple[tuple[float, float], ...] = (
    (0.5, 0.8),
    (0.8, 0.8),
    (0.5, 0.95),
    (0.8, 0.95),
    (0.95, 1.0),
)
BASELINE = "elemental-lambda-p0.5-v0.8-v1"


def _arm(name: str) -> tuple[float, float]:
    policy, value = name.split("/")
    return float(policy), float(value)


def runs(
    revision: str,
    updates: int = UPDATES,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
    seeds: str = SEEDS,
    arms: str = ",".join(f"{p:g}/{v:g}" for p, v in ARMS),
) -> list[Run]:
    """`arms` ("0.5/0.8,...") and `seeds` are comma-separated, so one arm launches
    all-or-nothing."""
    chosen = tuple(arm for arm in ARMS if arm in {_arm(a) for a in arms.split(",")})
    return training_runs(
        (lambda_sweep(*arm, updates) for arm in chosen),
        updates=updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=tuple(int(seed) for seed in seeds.split(",")),
        hours=hours,
        usd_per_run=usd_per_run,
    )


def eval(state: ExperimentState) -> list[Job]:
    return evaluation_jobs(
        state, "elemental-lambda-v1", ELEMENTAL_PAIRS, hours=1.5, usd=1.8
    )


def report() -> Report:
    return learning_report(BASELINE)
