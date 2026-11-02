"""The default-width move-learning baseline that value-model studies vary."""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, move_checkpoints
from manabot.sim.matchup import selected_match
from manabot.training.regime import TrainingRegime
import managym


def value_model_baseline() -> TrainingRegime:
    """Default-width depth-1 move learning: two one-update checkpoints."""
    return TrainingRegime(
        id="value-historical-mean-1",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(world=PACK),
        stages=move_checkpoints(2, updates=1, seconds=30),
        wall_seconds=60,
    )
