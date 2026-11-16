"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report owns the design and software prerequisites. This declaration cannot
launch until those prerequisites are delivered; the existing recent-event option
must not silently stand in for a decision-record encoder. No rentals or games run
on import. The implementation step will replace the explicit launch stop with
recipes from manabot.recipes.builders and replay-checked evaluation jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from etude.jobs import ExperimentState
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.spec import Job, RunRequest


@dataclass(frozen=True)
class Arm:
    """One encoder/summary choice over the shared viewer decision log."""

    name: str
    encoder: Literal["none", "transformer", "gru"]
    summaries: int


# CPU/probe candidates only; K selection precedes the three paid arms.
ARMS = (
    Arm("no-history", "none", 0),
    Arm("transformer-k1", "transformer", 1),
    Arm("transformer-k4", "transformer", 4),
    Arm("gru-k1", "gru", 1),
    Arm("gru-k4", "gru", 4),
)
SEEDS = (39101, 39102, 39103)
DISCLOSURES = ("open", "closed")
# Inclusive of calibration, training, evaluation, startup and failed attempts.
MAX_USD = 50


@dataclass(frozen=True)
class Quantiles:
    samples: int
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: int


@dataclass(frozen=True)
class Census:
    games: int
    decisions: int
    capped: int
    step_digest: int
    prefix_records: Quantiles
    terminal_records: Quantiles
    prefix_details: Quantiles
    terminal_details: Quantiles
    passes_before_coalescing: int
    passes_after_coalescing: int
    kinds: dict[str, int]
    record_capacity: int
    detail_capacity: int
    prefix_references: Quantiles
    terminal_references: Quantiles
    prefix_members: Quantiles
    terminal_members: Quantiles
    reference_capacity: int
    member_capacity: int
    padded_bytes_per_observation: int


def census(path: Path) -> Census:
    """Analyze the dated seeded-workload substitution, without policy buffers.

    Input is bench_steps --decision-memory-census NDJSON. The original step
    digest admits the regenerated workload; this is not retained-command replay.
    Capacities round observed maxima upward and remain subject to calibration.
    """
    import numpy as np
    from pydantic import BaseModel, ConfigDict

    class Terminal(BaseModel):
        model_config = ConfigDict(extra="forbid")
        viewer: int
        records: int
        details: int
        references: int
        members: int
        pass_records: int
        pass_events: int
        kinds: dict[str, int]

    class Game(BaseModel):
        model_config = ConfigDict(extra="forbid")
        seed: int
        steps: int
        capped: bool
        prefixes: list[tuple[int, int, int, int, int]]
        terminal: list[Terminal]

    class Summary(BaseModel):
        model_config = ConfigDict(extra="forbid")
        games: int
        capped: int
        steps: int
        elapsed_s: float
        steps_per_s: float
        digest: int

    def quantiles(values: list[int]) -> Quantiles:
        if not values:
            raise ValueError("census has no samples")
        p50, p90, p95, p99 = np.percentile(values, (50, 90, 95, 99))
        return Quantiles(
            len(values), float(p50), float(p90), float(p95), float(p99), max(values)
        )

    games: list[Game] = []
    summary: Summary | None = None
    for line in path.read_text().splitlines():
        if '"seed":' in line:
            games.append(Game.model_validate_json(line))
        else:
            if summary is not None:
                raise ValueError("duplicate census summary")
            summary = Summary.model_validate_json(line)
    if (
        summary is None
        or summary.games != 512
        or sorted(game.seed for game in games) != list(range(512))
    ):
        raise ValueError("census requires all original seeds 0..511 exactly once")
    if summary.digest != 17_459_788_495_858_550_277 or summary.steps != 102_209:
        raise ValueError(
            "regenerated workload differs from the retained step reference"
        )
    if sum(game.steps for game in games) != summary.steps:
        raise ValueError("census decisions disagree with the step receipt")
    prefixes = [prefix for game in games for prefix in game.prefixes]
    terminal = [row for game in games for row in game.terminal]
    if len(prefixes) != summary.steps or len(terminal) != 1024:
        raise ValueError("census is missing prefixes or terminal viewers")
    records = quantiles([row.records for row in terminal])
    details = quantiles([row.details for row in terminal])
    references = quantiles([row.references for row in terminal])
    members = quantiles([row.members for row in terminal])
    reference_capacity = 1 << (max(1, references.maximum) - 1).bit_length()
    member_capacity = 1 << (max(1, members.maximum) - 1).bit_length()
    record_capacity = 1 << (max(1, records.maximum) - 1).bit_length()
    detail_capacity = 1 << (max(1, details.maximum) - 1).bit_length()
    kinds: dict[str, int] = {}
    for row in terminal:
        for kind, count in row.kinds.items():
            kinds[kind] = kinds.get(kind, 0) + count
    return Census(
        summary.games,
        summary.steps,
        summary.capped,
        summary.digest,
        quantiles([row[1] for row in prefixes]),
        records,
        quantiles([row[2] for row in prefixes]),
        details,
        sum(row.pass_events for row in terminal),
        sum(row.pass_records for row in terminal),
        kinds,
        record_capacity,
        detail_capacity,
        quantiles([row[3] for row in prefixes]),
        references,
        quantiles([row[4] for row in prefixes]),
        members,
        reference_capacity,
        member_capacity,
        (
            record_capacity * 74
            + detail_capacity * 6
            + reference_capacity * 14
            + member_capacity * 2
        )
        * 4,
    )


def training_jobs(revision: str) -> list[RunRequest]:
    """Stop before allocation until CUDA admission and calibration are complete."""
    raise RuntimeError(
        "Opponent-memory CUDA peaks, worker/update and checkpoint-reload admission "
        "remain unmeasured at 2/2/2 execution limits. Calibrate a common learner-"
        "decision horizon and wire jobs before allocating training. "
        "See experiments/opponent_memory.md."
    )


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


def main() -> None:
    import argparse
    from dataclasses import asdict
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("census", "probe"))
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-games", type=int, default=120)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.operation == "probe":
        if args.source is not None:
            parser.error("probe takes no source path; it collects fresh native games")
        from manabot.backends.torch.decision_memory_probe import run_probe

        run_probe(
            args.out,
            epochs=args.epochs,
            max_games=args.max_games,
            wall_seconds=args.wall_seconds,
        )
        return
    if args.source is None:
        parser.error("census requires a source path")
    summary = census(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
