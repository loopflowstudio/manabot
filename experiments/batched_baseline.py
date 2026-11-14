"""The batched GPU baseline from the size and batching study.

Width 64, depth 2, 64 streams with a cross-balanced Allies/Lessons curriculum,
its learning-rate schedule mapped onto 2,048 four-stream iterations
(experiments/reports/longer-learning-017.md).

This file is the experiment `batched_baseline` for Etude, and the template for
a single GPU recipe: `train(config)` is the run and `training_jobs()` declares
the one job, named for the file as its stored runs are.

    uv run etude train experiments/batched_baseline.py --config '{"device": "cpu"}'
    uv run etude launch experiments/batched_baseline.py --revision COMMIT --rent l4
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.jobs import TrainingJob
from etude.train.tracking import track
from pydantic import BaseModel, Field

from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, move_learning
from manabot.sim.matchup import selected_match
from manabot.training.learning import IterationSchedule
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem


class BatchedConfig(BaseModel):
    updates: int = Field(default=128, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cuda"


def train(config: BatchedConfig) -> None:
    inputs = SelfPlay(
        match=selected_match(),
        model=ModelConfig(
            world=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        learning=move_learning(IterationSchedule(steps=128, reference_steps=2048)),
        streams=64,
        transitions=64,
        matchup_curriculum="cross-balanced",
        device=config.device,
        checkpoint_every=64,
    )
    with track(clocks=CLOCKS, config=config) as run:
        self_play(inputs, run, updates=config.updates)


def training_jobs(
    revision: str = "HEAD", hours: float = 2.0, usd_per_run: float = 2.0
) -> list[TrainingJob]:
    """The baseline's one job and seed."""
    return [
        TrainingJob(
            name=EXPERIMENT,
            train=train,
            config=BatchedConfig(),
            smoke=BatchedConfig(updates=1, device="cpu"),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
    ]
