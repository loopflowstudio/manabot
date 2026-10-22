"""The default-width move-learning baseline that value-model studies vary."""

from manabot.arena.match import selected_match
from manabot.infra.hypers import AgentSpec
from manabot.training.builders import PACK, move_checkpoints
from manabot.training.models import TrainingRegime
import managym


def value_model_baseline() -> TrainingRegime:
    """Default-width depth-1 move learning: two one-update checkpoints."""
    return TrainingRegime(
        id="value-historical-mean-1",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(semantic_pack=PACK),
        stages=move_checkpoints(2, updates=1, seconds=30),
        wall_seconds=60,
    )
