"""One-seat versus two-seat self-play from scratch, matched per learner decision (MTG-165).

Jack Heart's question (2026-10-22): is score against the heuristic at N
learner decisions the same whether the decisions come from one seat or both?
If so, two-seat at E environment decisions matches one-seat at 2E. Four seeds
per arm to about 64M learner decisions; the protocol is
experiments/protocols/seat-comparison-v1.md and the matched recipe
`experiments.batched_baseline.seat_comparison`.

This file is the experiment `seat_comparison` for Etude: `runs()` declares one
job per arm (each run once per seed), and `report()` draws their scores on the
learner-decision clock. Each arm's job is named for its regime, so the
one-seat arm `seat-comparison-one-v1` is the dotted baseline.

    uv run etude launch experiments/seat_comparison.py --revision COMMIT --rent l4 \
        --max-usd 60
    uv run etude report experiments/seat_comparison.py --source r2://manabot/jobs --open

v1 ran before Etude evaluation: its retained scores are arena rows under
`experiments/data/seat-comparison-v1`, summarized by
`python -m experiments.seat_comparison_report`, not Etude eval runs. Its
evaluation (the two-deck heuristic benchmark, `manabot.eval.benchmark`) is
bound to each checkpoint's training record, so it is not yet an `eval(state)`.
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.launch.records import Run
from etude.report import Curve, Matrix, Ranking, Report, Score

from experiments.batched_baseline import seat_comparison
from manabot.experiment.job import TrainingJob
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
SEEDS = (64001, 64002, 64003, 64004)
# About 27k learner decisions per update in both arms (L4 calibration, seed
# 64000): 2,700 updates passes 64M. Exports every 20 updates.
UPDATES = 2_700
CHECKPOINT_UPDATES = 20
ARMS: tuple[Literal["one", "both"], ...] = ("one", "both")
BASELINE = "seat-comparison-one-v1"
# Machine setup and the final export, inside each run's hours.
SETUP_SECONDS = 900
FINALIZATION_SECONDS = 300


def runs(
    revision: str,
    hours: float = 11.9,
    usd_per_run: float = 7.27,
    updates: int = UPDATES,
) -> list[Run]:
    """Each arm's job, run once per seed, with its watchdog fitted to `hours`.

    A run that has not reached `updates` when its watchdog ends fails, keeping
    the checkpoints it exported; matched comparisons extend only to the
    learner decisions every run reached.
    """
    found: list[Run] = []
    for arm in ARMS:
        regime = seat_comparison(arm, updates)
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        regime.wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
        stage.execution.wall_seconds = regime.wall_seconds
        job = TrainingJob(
            regime=TrainingRegime.model_validate(regime.model_dump(mode="json")),
            experiment=EXPERIMENT,
            milestones=tuple(range(CHECKPOINT_UPDATES, updates, CHECKPOINT_UPDATES)),
            finalization_seconds=FINALIZATION_SECONDS,
        ).job(
            repo=ROOT,
            revision=revision,
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
        found += job.runs(*SEEDS)
    return found


def report() -> Report:
    """Score against the heuristic at equal learner decisions, by matchup and seat.

    `plays` is the two-deck benchmark's table (`manabot.eval.benchmark`): a
    row per leg and player, `score` 1 for a win and 0.5 for a draw.
    """
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="learner_transitions",
        baseline=BASELINE,
        evals=(
            Curve(title="Score against the heuristic"),
            Curve(title="By matchup", split="matchup"),
            Matrix(
                title="Model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
