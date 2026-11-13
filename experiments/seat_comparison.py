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
`experiments/data/seat-comparison-v1` (its report is
`experiments/data/seat-comparison-v1/report.md`), not Etude eval runs. Its
evaluation (the two-deck heuristic benchmark, `manabot.eval.benchmark`) is
bound to each checkpoint's training record, so it is not yet an `eval(state)`.
"""

from pathlib import Path
from typing import Literal

from etude.report import Curve, Matrix, Ranking, Report, Score

from manabot.training.panels import ML, PERF

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


def runs(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    raise RuntimeError(
        "Frozen execution requires revision f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653"
    )


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
