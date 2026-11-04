"""One-seat exact quota versus one-seat complete games, all else matched (Task fc6ffb).

Does stopping every stream mid-game and bootstrapping the tail cost learning, and
does finishing games cost time? Protocol: experiments/protocols/closure-lambda-v1.md.
Recipe: `experiments.batched_baseline.closure_comparison`.

    uv run etude eval experiments/batch_closure.py --rent SUPPLY
    uv run etude report experiments/batch_closure.py --source r2://manabot/jobs --open
"""

from decimal import Decimal
from pathlib import Path

from etude.eval.match import Tables
from etude.eval.records import Match
from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments import batch_closure, learning_signal
from experiments.batched_baseline import closure_comparison
from manabot.experiment.job import TrainingJob
from manabot.training.regime import BatchClosure, TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "batch_closure"
SEEDS = (65101, 65102, 65103, 65104)
ARMS: tuple[BatchClosure, ...] = ("exact-quota", "complete-games")
BASELINE = "batch-closure-exact-quota-v1"
UPDATES = 300
SCORED_UPDATES = (0, 20, 40, 75, 150, 300)
SETUP_SECONDS = 900
FINALIZATION_SECONDS = 300


def regime(arm: BatchClosure, updates: int, hours: float) -> TrainingRegime:
    found = closure_comparison(arm, updates)
    stage = found.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    found.wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    stage.execution.wall_seconds = found.wall_seconds
    return TrainingRegime.model_validate(found.model_dump(mode="json"))


def runs(
    revision: str,
    hours: float = 3.0,
    usd_per_run: float = 2.5,
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
        state, "batch-closure-v1", SCORED_UPDATES, batch_closure.play, 8, 3.0
    )


def report() -> Report:
    return learning_signal.learning_report(BASELINE)
