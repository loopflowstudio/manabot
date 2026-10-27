"""Direct PPO self-play: the default model on the selected Allies/Lessons match.

`uv run manabot train --regime-name direct_self_play --seed 197 --out .runs/direct-pilot`
trains it for 900 seconds; the ablation and compound studies derive their arms
from it.
"""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, ppo_checkpoints
from manabot.sim.matchup import selected_match
from manabot.training.regime import Learning, TrainingRegime
import managym


def direct_self_play() -> TrainingRegime:
    """Two two-update PPO checkpoints with every Ataraxos treatment enabled."""
    return TrainingRegime(
        id="direct-self-play",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(semantic_pack=PACK),
        stages=ppo_checkpoints(2, Learning()),
        wall_seconds=900,
    )
