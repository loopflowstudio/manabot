"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report owns the design and software prerequisites. This declaration cannot
launch until those prerequisites are delivered; the existing recent-event option
must not silently stand in for a decision-record encoder. No rentals or games run
on import. The implementation step will replace the explicit launch stop with
recipes from manabot.recipes.builders and replay-checked evaluation jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from etude.report import Curve, Matches, Matrix, Ranking, Report, Score

if TYPE_CHECKING:
    from etude.jobs import ExperimentState
    from etude.launch.records import Job, Run


@dataclass(frozen=True)
class Arm:
    """One change from the best guess; pass filtering acts on one shared log."""

    name: str
    encoder: Literal["none", "recent-events", "transformer", "gru"]
    summaries: int
    passes: Literal["eligible", "omit"] = "eligible"


ARMS = (
    Arm("no-history", "none", 0),
    Arm("recent-events", "recent-events", 0),
    Arm("transformer-k4", "transformer", 4),
    Arm("gru-k4", "gru", 4),
    Arm("transformer-k1", "transformer", 1),
    Arm("transformer-k4-no-passes", "transformer", 4, "omit"),
)
SEEDS = (39101, 39102, 39103)
DISCLOSURES = ("open", "closed")
# Inclusive of calibration, training, evaluation, startup and failed attempts.
MAX_USD = 50


def runs(revision: str) -> list[Run]:
    """Stop before job construction until the shared memory software ships."""
    raise RuntimeError(
        "Opponent-memory software is not delivered at this design revision. "
        "Implement and publish the viewer-safe whole-game decision log, native/"
        "Python inputs, transformer/GRU summary encoders and recipe builders; "
        "then wire these arms and calibrate before allocating training. "
        "See experiments/opponent_memory.md."
    )


def eval(state: ExperimentState) -> list[Job]:
    """Stop rather than score unmatched updates as matched learner decisions."""
    raise RuntimeError(
        "Wire paired Elemental 10x10 evaluation to retained learner-decision "
        "checkpoint clocks after the software prerequisite; see the report."
    )


def report() -> Report:
    """Use the existing Elemental matchup panels and the learner-decision clock."""
    from manabot.training.panels import ML, PERF

    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="learner_transitions",
        evals=(
            Curve(
                title="Head-to-head against no history",
                matches=Matches(tag="control"),
                split="decklists",
            ),
            Curve(
                title="Intact memory against blanked memory",
                matches=Matches(tag="reliance"),
                split="decklists",
            ),
            Matrix(
                title="Descriptive arm versus control matchups",
                rows="opponent_deck",
                columns="deck",
                matches=Matches(tag="control"),
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
