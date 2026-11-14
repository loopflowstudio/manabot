"""The batched GPU baseline from the size and batching study.

Width 64, depth 2, 64 streams with a cross-balanced Allies/Lessons curriculum,
its learning-rate schedule mapped onto 2,048 four-stream iterations
(experiments/reports/longer-learning-017.md).

    uv run etude train experiments/batched_baseline.py
"""

from typing import Literal

from etude.train.tracking import track
from pydantic import BaseModel, Field

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, move_learning
from manabot.sim.matchup import selected_match
from manabot.training.learning import IterationSchedule
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play


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
