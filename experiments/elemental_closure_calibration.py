"""The unscored calibration run that sets the Elemental exact arm's quota (Task 07c490).

One seed of `experiments/elemental_closure.py`'s complete-games arm for 30 updates.
Etude names a run's experiment after its launch file, so the calibration has its own
file and no evaluation pass of the closure study sees it. Protocol:
experiments/protocols/closure-lambda-elemental-v1.md, calibration section.

    uv run etude launch experiments/elemental_closure_calibration.py --rent 07c490-l4 \\
        --revision COMMIT
"""

from etude.launch.records import Run

from experiments.elemental_variants import closure_comparison
from experiments.learning_signal import training_runs

EXPERIMENT = "elemental_closure_calibration"
SEED = 70100
UPDATES = 30
# The window and its reservation (window plus setup at $.60) from the protocol.
HOURS = 0.75
USD_PER_RUN = 0.6


def runs(revision: str) -> list[Run]:
    return training_runs(
        [closure_comparison("complete-games", UPDATES)],
        updates=UPDATES,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=(SEED,),
        hours=HOURS,
        usd_per_run=USD_PER_RUN,
    )
