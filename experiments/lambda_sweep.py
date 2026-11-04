"""Does more direct game-result credit help on complete games? (Task fc6ffb)

Two-seat complete-game recipe, lambda alone varying above the current .5/.8. At
twenty decisions from the end those traces give the result weight about 1e-6 and
.012. Protocol: experiments/protocols/closure-lambda-v1.md. Recipe:
`experiments.batched_baseline.lambda_sweep`. Critic quality by distance from game
end is each update's `selection_groups` metrics, in the report's ML panel.

    uv run etude eval experiments/lambda_sweep.py --rent SUPPLY
    uv run etude report experiments/lambda_sweep.py --source r2://manabot/jobs --open
"""

from decimal import Decimal
from pathlib import Path

from etude.eval.match import Tables
from etude.eval.records import Match
from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments import lambda_sweep, learning_signal
from experiments.batched_baseline import lambda_sweep as sweep_recipe
from manabot.experiment.job import TrainingJob
from manabot.training.regime import TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "lambda_sweep"
SEEDS = (65201, 65202, 65203)
# (policy, value). The first is the current recipe and the baseline.
ARMS: tuple[tuple[float, float], ...] = (
    (0.5, 0.8),
    (0.8, 0.8),
    (0.5, 0.95),
    (0.8, 0.95),
    (0.95, 1.0),
)
BASELINE = "lambda-sweep-p0.5-v0.8-v1"
UPDATES = 300
SCORED_UPDATES = (0, 20, 40, 75, 150, 300)
SETUP_SECONDS = 900
FINALIZATION_SECONDS = 300


def regime(arm: tuple[float, float], updates: int, hours: float) -> TrainingRegime:
    found = sweep_recipe(*arm, updates)
    stage = found.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    found.wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    stage.execution.wall_seconds = found.wall_seconds
    return TrainingRegime.model_validate(found.model_dump(mode="json"))


def runs(
    revision: str,
    hours: float = 2.5,
    usd_per_run: float = 2.0,
    updates: int = UPDATES,
    seeds: tuple[int, ...] = SEEDS,
    experiment: str = EXPERIMENT,
) -> list[Run]:
    """Each arm's job, once per seed; a watchdog stop keeps its exports."""
    found: list[Run] = []
    for arm in ARMS:
        found += (
            TrainingJob(
                regime=regime(arm, updates, hours),
                experiment=experiment,
                milestones=tuple(u for u in SCORED_UPDATES if 0 < u < updates),
                finalization_seconds=FINALIZATION_SECONDS,
            )
            .job(
                repo=ROOT,
                revision=revision,
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
            )
            .runs(*seeds)
        )
    return found


def play(match: Match) -> Tables:
    return learning_signal.play_deal(match)


def eval(state: ExperimentState) -> list[Job]:
    return learning_signal.evaluation_jobs(
        state, "lambda-sweep-v1", SCORED_UPDATES, lambda_sweep.play, 8, 3.0
    )


def report() -> Report:
    return learning_signal.learning_report(BASELINE)
