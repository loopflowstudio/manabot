"""Ordinary damped-policy self-play on the selected Academy matchup.

`settings` builds the learning inputs; `train` declares an Etude run and calls
the loop. Evaluation is a separate Etude pass over its immutable checkpoints.
"""

from pathlib import Path
from typing import Literal

from etude.train.tracking import track
from pydantic import BaseModel, ConfigDict, Field

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK
from manabot.sim.matchup import selected_match
from manabot.training.learning import DampedPolicyLearning, IterationSchedule
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play
from manabot.verify.learning_regression import BaselineReport, report_recipe

BASELINE_STEPS = 128
BASELINE_DEALS = tuple(range(1_915_100_000, 1_915_100_024))


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    updates: int = Field(default=BASELINE_STEPS, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cuda"
    learning_seats: Literal["one", "both"] = "both"
    padded_shapes: bool = False


def settings(config: BaselineConfig) -> SelfPlay:
    """The recipe horizon fixes its rates; a later launch target runs past it at the final rates."""
    return SelfPlay(
        match=selected_match(),
        model=ModelConfig(
            hidden_dim=64,
            attention_layers=2,
            num_attention_heads=4,
            world=PACK,
            value_kind="scalar",
            value_aggregation="masked_mean",
            policy_film=True,
        ),
        streams=4,
        transitions=64,
        learning_seats=config.learning_seats,
        learning=DampedPolicyLearning(
            gradient="damped_policy",
            iteration_schedule=IterationSchedule(steps=config.updates),
        ),
        device=config.device,
        bucketed_shapes=config.device == "cuda" and not config.padded_shapes,
        live_tokens=config.device == "cuda" and not config.padded_shapes,
        checkpoint_every=max(1, config.updates // 2),
    )


def train(config: BaselineConfig) -> None:
    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def evaluate(
    config: BaselineConfig, directory: Path, *, seed: int, seconds: float = 1680
) -> BaselineReport:
    """Evaluate the initial and endpoint policies on the nightly paired-deal cohort."""
    return report_recipe(
        directory,
        settings(config),
        seed=seed,
        updates=config.updates,
        deals=BASELINE_DEALS,
        seconds=seconds,
    )
