"""Small `TrainingRegime` records for tests of the retained regime schema.

Regimes are records now: nothing here trains. Each builder is the shape one
group of tests reads, on the selected Allies/Lessons match.
"""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import (
    GIB,
    PACK,
    move_checkpoints,
    move_learning,
    ppo_checkpoints,
    with_value_output,
)
from manabot.sim.matchup import selected_match
from manabot.training.learning import IterationSchedule, Learning
from manabot.training.regime import Execution, TrainingRegime, TrainSelfPlay
import managym


def direct_self_play() -> TrainingRegime:
    """Two two-update PPO checkpoints of the default model."""
    return TrainingRegime(
        id="direct-self-play",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(world=PACK),
        stages=ppo_checkpoints(2, Learning()),
        wall_seconds=900,
    )


def ataraxos_move() -> TrainingRegime:
    """Width-16 two-head move learning with a win/draw/loss value: two checkpoints."""
    return TrainingRegime(
        id="ataraxos-move",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(
            world=PACK,
            hidden_dim=16,
            num_attention_heads=2,
            value_kind="categorical_wdl",
        ),
        stages=move_checkpoints(2, updates=1, seconds=80),
        wall_seconds=180,
    )


def ataraxos_move_scalar() -> TrainingRegime:
    """`ataraxos_move` with a scalar value head."""
    return with_value_output(
        ataraxos_move(), id="ataraxos-move-scalar", output="scalar"
    )


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


def batched_baseline() -> TrainingRegime:
    """Width 64, depth 2, 64 streams on CUDA with a padded forward layout."""
    return TrainingRegime(
        id="batching-grid-small-s64",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(
            world=PACK, value_aggregation="masked_mean", attention_layers=2
        ),
        stages=[
            TrainSelfPlay(
                id="self-play",
                operation="train_self_play",
                updates=128,
                streams=64,
                transitions=64,
                matchup_curriculum="cross-balanced",
                learning=move_learning(
                    IterationSchedule(steps=128, reference_steps=2048)
                ),
                execution=Execution(
                    device="cuda",
                    wall_seconds=1740,
                    memory_bytes=48 * GIB,
                    bucketed_shapes=False,
                    live_tokens=False,
                ),
            )
        ],
        wall_seconds=1770,
    )
