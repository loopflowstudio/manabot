"""One-seat exact quota versus one-seat complete games on the ten-deck Elemental recipe (Task 07c490).

MTG-251 asked this on Allies/Lessons (`experiments/batch_closure.py`). Jack Heart
placed the current recipe at the ten-deck Elemental setup, so this repeats it there,
with the model scored on random deck pairs against the heuristic. Protocol:
experiments/protocols/closure-lambda-elemental-v1.md. Recipe:
`experiments.elemental_variants.closure_comparison`.

    uv run etude launch experiments/elemental_closure.py --rent 07c490-l4 \\
        --cohort elemental-closure-exact --parallel 3 --revision COMMIT --arms exact-quota
    uv run etude eval experiments/elemental_closure.py --rent 07c490-cpu32 --source r2://manabot/jobs
    uv run etude report experiments/elemental_closure.py --source r2://manabot/jobs --open
"""

from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments.elemental_variants import closure_comparison
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    evaluation_jobs,
    learning_report,
    training_runs,
)
from manabot.training.regime import BatchClosure

EXPERIMENT = "elemental_closure"
SEEDS = "70101,70102,70103"
UPDATES = 150
# Rental hours and dollars per run at the L4 rate the protocol budgets.
HOURS = 1.5
USD_PER_RUN = 0.89
ARMS: tuple[BatchClosure, ...] = ("exact-quota", "complete-games")
BASELINE = "elemental-closure-exact-quota-v1"


def runs(
    revision: str,
    updates: int = UPDATES,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
    seeds: str = SEEDS,
    arms: str = ",".join(ARMS),
) -> list[Run]:
    """`arms` and `seeds` are comma-separated, so one arm launches all-or-nothing."""
    chosen = tuple(arm for arm in ARMS if arm in arms.split(","))
    return training_runs(
        (closure_comparison(arm, updates) for arm in chosen),
        updates=updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=tuple(int(seed) for seed in seeds.split(",")),
        hours=hours,
        usd_per_run=usd_per_run,
    )


def eval(state: ExperimentState) -> list[Job]:
    return evaluation_jobs(
        state, "elemental-closure-v1", ELEMENTAL_PAIRS, hours=1.0, usd=1.2
    )


def report() -> Report:
    return learning_report(BASELINE)
