"""Small Ataraxos move learning and its scalar-value control (docs/ataraxos.md)."""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, move_checkpoints, with_value_output
from manabot.sim.matchup import selected_match
from manabot.training.regime import TrainingRegime
import managym


def ataraxos_move() -> TrainingRegime:
    """Width-16 two-head move learning with a win/draw/loss value: two checkpoints."""
    return TrainingRegime(
        id="ataraxos-move",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(
            semantic_pack=PACK,
            hidden_dim=16,
            num_attention_heads=2,
            value_kind="categorical_wdl",
        ),
        stages=move_checkpoints(2, updates=1, seconds=80),
        wall_seconds=180,
    )


def ataraxos_move_scalar() -> TrainingRegime:
    """`ataraxos_move` with a scalar value head, the representation control."""
    return with_value_output(
        ataraxos_move(), id="ataraxos-move-scalar", output="scalar"
    )
