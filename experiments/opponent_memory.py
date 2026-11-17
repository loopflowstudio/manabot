"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report owns the design and software prerequisites. This declaration cannot
launch until those prerequisites are delivered; the existing recent-event option
must not silently stand in for a decision-record encoder. No rentals or games run
on import. The implementation step will replace the explicit launch stop with
recipes from manabot.recipes.builders and replay-checked evaluation jobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.jobs import ExperimentState
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.spec import Job, RunRequest

from manabot.experiment.job import PretrainedInitialization, TrainingJob
from manabot.training.regime import TrainingRegime


@dataclass(frozen=True)
class Arm:
    """One encoder/summary choice over the shared viewer decision log."""

    name: str
    encoder: Literal["none", "transformer", "gru"]
    summaries: int


# K=1 selected by the corrected supervised probe; three pilot arms.
ARMS = (
    Arm("control", "none", 0),
    Arm("transformer-k1", "transformer", 1),
    Arm("gru-k1", "gru", 1),
)
SEEDS = (39101, 39102, 39103)
DISCLOSURES = ("open", "closed")
# Inclusive of calibration, training, evaluation, startup and failed attempts.
MAX_USD = 50


def training_jobs(revision: str) -> list[RunRequest]:
    """Stop before allocation until CUDA admission and calibration are complete."""
    raise RuntimeError(
        "Opponent-memory CUDA peaks, worker/update and checkpoint-reload admission "
        "remain unmeasured at 2/2/2 execution limits. Calibrate a common learner-"
        "decision horizon and wire jobs before allocating training. "
        "See experiments/opponent_memory.md."
    )


def branch_jobs(
    revision: str,
    *,
    regime: TrainingRegime,
    trunks: Mapping[int, PretrainedInitialization],
    max_hours: float,
    max_usd: Decimal,
) -> list[RunRequest]:
    """Freeze matched branches from published trunk inputs without launching.

    Calibration supplies the shared branch recipe, horizon, gate scale and caps.
    Each seed maps to exactly one producer receipt and EMA export, reused by all
    three arms. The ordinary training declaration stays stopped until CUDA
    admission and calibration.
    """
    if not trunks or any(seed not in SEEDS for seed in trunks):
        raise ValueError("branches require declared pilot seed labels")
    if regime.agent.decision_memory != "none":
        raise ValueError("the shared branch recipe must describe the control")
    if any(trunk.weights != "ema" for trunk in trunks.values()):
        raise ValueError("all branches use the same EMA trunk role")
    if any(trunk.source_seed != seed for seed, trunk in trunks.items()):
        raise ValueError("each branch seed label must match its producer seed")
    jobs: list[RunRequest] = []
    for seed, trunk in sorted(trunks.items()):
        for arm in ARMS:
            configuration = regime.model_copy(deep=True)
            configuration.id = f"decision-memory-{arm.name}"
            configuration.agent.decision_memory = arm.encoder
            if arm.summaries:
                configuration.agent.decision_memory_summaries = arm.summaries
            job = TrainingJob(
                regime=configuration,
                experiment=Path(__file__).stem,
                pretrained=trunk,
            ).job(
                repo=Path(__file__).resolve().parents[1],
                revision=revision,
                max_hours=max_hours,
                max_usd=max_usd,
            )
            jobs.extend(job.runs(seed))
    return jobs


def eval_jobs(state: ExperimentState) -> list[Job]:
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
                title="Head-to-head against control",
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    census_parser = operations.add_parser("census")
    census_parser.add_argument("--out", type=Path, required=True)
    probe_parser = operations.add_parser("probe")
    probe_parser.add_argument("--out", type=Path, required=True)
    probe_parser.add_argument("--epochs", type=int, default=8)
    probe_parser.add_argument("--max-games", type=int, default=120)
    probe_parser.add_argument("--wall-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.operation == "census":
        from manabot.verify.decision_memory_capacity import run_census

        run_census(args.out)
        return
    if args.operation == "probe":
        from manabot.backends.torch.decision_memory_probe import run_probe

        run_probe(
            args.out,
            epochs=args.epochs,
            max_games=args.max_games,
            wall_seconds=args.wall_seconds,
        )
        return


if __name__ == "__main__":
    main()
