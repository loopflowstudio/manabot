"""The BF16 learning gate at w384l8 (MTG-324).

Training is frozen: `runs` refuses and names the revision holding the original
float32/BF16 arm builder and executor. `eval` schedules heuristic and
head-to-head matches at retained checkpoints; `report` displays scores and
learning trajectories. The numerical gate rejected BF16 before evaluation. The
frozen prediction, measurements, costs and launch history are in
`experiments/precision_ladder.md`.

    uv run etude eval experiments/precision_ladder.py --rent cpu32 --source r2://manabot/jobs
    uv run etude report experiments/precision_ladder.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from etude.eval.records import Inference, Seating
from etude.experiment import ExperimentState, TrainingRun
from etude.launch.records import Job
from etude.report import Curve, Matches, Ranking, Report, Score

from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    Conditions,
    benchmark_jobs,
    scheduled_matchups,
)
from manabot.infra.runtime import regime_tag
from manabot.training.learning import Precision
from manabot.training.panels import ML, PERF
from manabot.training.regime import refuse_frozen_execution

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
# float32 first: the reference arm and the report's dotted baseline.
ARMS: tuple[Precision, ...] = ("float32", "bf16")
WIDTH, LAYERS = 384, 8
EVALUATION = "precision-ladder-v1"
# 20 heuristic matchups and 4 head-to-heads × 50 pairings × 4 legs: 4,800 games,
# about half of value_readout_252's pass (9,000 games in 1.25 h on cpu32).
EVALUATION_HOURS = 1.0
EVALUATION_USD = 2.0


def regime_id(precision: Precision) -> str:
    return f"precision-ladder-w{WIDTH}l{LAYERS}-{precision}-v1"


def runs(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    refuse_frozen_execution()


# --------------------------------------------------------------------------- evaluation


def arm_runs(state: ExperimentState, precision: Precision) -> list[TrainingRun]:
    """The arm's runs: their checkpoints' regime tag names them."""
    tag = regime_tag(regime_id(precision))
    return [run for run in state.runs if run.checkpoints.tagged(tag)]


def eval(state: ExperimentState) -> list[Job]:
    return benchmark_jobs(
        evaluation_matchups(state),
        EVALUATION,
        ELEMENTAL_PAIRS,
        hours=EVALUATION_HOURS,
        usd=EVALUATION_USD,
    )


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Scheduled checkpoints against the heuristic, then BF16 against float32.

    A checkpoint not yet stored is left out, so a later pass plays it. A
    head-to-head pairs two ended runs at the last update both checkpointed,
    float32 first; the four legs give each both decks on the play and the draw.
    """
    found = [
        matchup
        for run in state.runs
        for matchup in scheduled_matchups(run, ELEMENTAL_PAIRS)
    ]
    # The same source as `scheduled_matchups`, so a checkpoint is one player in both.
    source = [
        *sorted((ROOT / "manabot").rglob("*.py")),
        ROOT / "experiments/learning_signal.py",
    ]
    inference = Inference(source=source)
    conditions = tuple(
        Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
        for pairing in ELEMENTAL_PAIRS.pairings
    )
    reference, candidate = (arm_runs(state, precision) for precision in ARMS)
    for left, right in product(reference, candidate):
        if not (left.ended and right.ended):
            continue
        shared = {int(c.at["updates"]) for c in left.checkpoints} & {
            int(c.at["updates"]) for c in right.checkpoints
        }
        if not shared:
            continue
        update = max(shared)
        first = left.checkpoints.at("updates", update).last
        second = right.checkpoints.at("updates", update).last
        assert first is not None and second is not None
        found.append(
            Seating(
                name=f"{ARMS[0]} {left.name} v {ARMS[1]} {right.name} u{update}",
                players=(first.model(inference), second.model(inference)),
                conditions=conditions,
                games=ELEMENTAL_PAIRS.legs,
                tags=("head-to-head",),
            )
        )
    return tuple(found)


def report() -> Report:
    """Score against the heuristic by update; the ML panels hold loss, clip and KL."""
    heuristic = Matches(tag="heuristic")
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
        baseline=regime_id(ARMS[0]),
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
