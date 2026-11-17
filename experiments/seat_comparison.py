"""One-seat versus two-seat self-play from scratch: a method comparison across seeds (MTG-165).

The question (2026-10-22): is score against the heuristic the same whether a
learner's decisions come from one seat or both? The arms differ only in
`learning_seats` and in `games_per_stream`, which keeps their learner decisions
per update about equal: two-seat learns from one whole game per stream per
update, one-seat from two.

This file is the experiment `seat_comparison` for Etude, and the small template
for comparing method arms: `SeatConfig` holds the fields the arms differ in,
`train(config)` is one run, `training_jobs()` declares one job per arm with the
same seeds, and `report()` draws the arms on the update clock with the one-seat
arm as the dotted baseline.

    uv run etude launch experiments/seat_comparison.py --revision COMMIT --rent l4 \\
        --max-usd 60
    uv run etude report experiments/seat_comparison.py --source r2://manabot/jobs --open

There is no `eval_jobs(state)` yet. The two-deck heuristic benchmark
(`manabot.eval.benchmark`) reads the retired executor's training record, so it
cannot score these runs' checkpoints; `report()` names the `plays` table that
benchmark writes, and draws training until an evaluation delivers it.

v1 (four seeds per arm to about 64M learner decisions) ran under the retired
executor at revision f4e1822faf46bbc9cdc43909293f8cbc8de4a300, on padded
forwards and matched on a learner-decision clock the current loop does not
declare. Its protocol is experiments/protocols/seat-comparison-v1.md and its
results are arena rows under `experiments/data/seat-comparison-v1`.
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.jobs import TrainingJob
from etude.report import Curve, Matrix, Ranking, Report, Score
from etude.train.tracking import track
from pydantic import BaseModel, ConfigDict, Field

from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, move_learning
from manabot.sim.matchup import selected_match
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play
from manabot.training.panels import ML, PERF

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
SEEDS = (64001, 64002, 64003, 64004)
# About 27k learner decisions per update in both arms (L4 calibration of v1):
# 2,700 updates passes 64M.
UPDATES = 2_700
CHECKPOINT_UPDATES = 20
Seats = Literal["one", "both"]
ARMS: tuple[Seats, ...] = ("one", "both")


def job_name(seats: Seats) -> str:
    return f"seat-comparison-{seats}"


BASELINE = job_name("one")


class SeatConfig(BaseModel):
    """One arm at one horizon; `learning_seats` is the treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_seats: Seats
    updates: int = Field(default=UPDATES, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cuda"
    streams: int = Field(default=256, ge=4)


def settings(config: SeatConfig) -> SelfPlay:
    """Width 64, depth 2 damped-policy self-play on Allies/Lessons, complete games.

    Damping follows the absolute update, so a shorter run is a prefix of a
    longer one. Only `learning_seats` and `games_per_stream` differ by arm.
    """
    return SelfPlay(
        match=selected_match(),
        model=ModelConfig(
            world=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        learning=move_learning(),
        streams=config.streams,
        transitions=64,
        games_per_stream=1 if config.learning_seats == "both" else 2,
        learning_seats=config.learning_seats,
        batch_closure="complete-games",
        matchup_curriculum="cross-balanced",
        device=config.device,
        bucketed_shapes=config.device == "cuda",
        live_tokens=config.device == "cuda",
        checkpoint_every=CHECKPOINT_UPDATES,
    )


def train(config: SeatConfig) -> None:
    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str,
    updates: int = UPDATES,
    hours: float = 6.0,
    usd_per_run: float = 7.5,
) -> list[TrainingJob]:
    """One job per arm, every arm on the same seeds."""
    return [
        TrainingJob(
            name=job_name(seats),
            train=train,
            config=SeatConfig(learning_seats=seats, updates=updates),
            smoke=SeatConfig(learning_seats=seats, updates=1, device="cpu", streams=4),
            seeds=SEEDS,
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
        for seats in ARMS
    ]


def report() -> Report:
    """Score against the heuristic at equal updates, by matchup and seat.

    `plays` is the two-deck benchmark's table (`manabot.eval.benchmark`): a
    row per leg and player, `score` 1 for a win and 0.5 for a draw.
    """
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
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
