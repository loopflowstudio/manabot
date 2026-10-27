"""The CPU Ataraxos Mini recipe and the screens that shorten or split it.

`ataraxos_mini` is the source the early learning-rate screen varies; the mirror
screen and the sustained baseline run it with their own stage plans.
"""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import GIB, PACK, move_checkpoints, move_stage
from manabot.sim.matchup import selected_match
from manabot.training.regime import TrainingRegime
import managym


def ataraxos_mini() -> TrainingRegime:
    """Masked-mean width-64 depth-2 move learning: one 1,500-update CPU stage."""
    return TrainingRegime(
        id="early-learning-rate-mini-source",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(
            semantic_pack=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        stages=[
            move_stage("self-play", updates=1500, seconds=10800, memory_bytes=15 * GIB)
        ],
        wall_seconds=10800,
    )


def mirror_screen_baseline() -> TrainingRegime:
    """`ataraxos_mini` shortened to a 20-update, 12-stream recoverable midpoint."""
    regime = ataraxos_mini()
    regime.id = "mirror-screen-baseline"
    regime.stages = [
        move_stage(
            "midpoint",
            updates=20,
            streams=12,
            transitions=32,
            seconds=600,
            memory_bytes=15 * GIB,
        )
    ]
    regime.wall_seconds = 660
    regime.schedule_clock = "iteration_fraction"
    regime.recovery_max_microsteps = 1_000_000
    return TrainingRegime.model_validate(regime.model_dump())


def sustained_baseline_source() -> TrainingRegime:
    """`ataraxos_mini` as two cumulative 620-update checkpoints."""
    regime = ataraxos_mini()
    regime.id = "actor-critic"
    regime.stages = move_checkpoints(
        2, updates=620, seconds=3590, memory_bytes=15 * GIB
    )
    regime.wall_seconds = 7200
    return TrainingRegime.model_validate(regime.model_dump())
