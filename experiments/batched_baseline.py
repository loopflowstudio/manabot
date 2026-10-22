"""The batched GPU baseline from the size and batching study, and its scale ladder."""

from manabot.arena.match import selected_match
from manabot.infra.hypers import AgentSpec
from manabot.training.builders import GIB, PACK, move_learning
from manabot.training.models import (
    Execution,
    IterationSchedule,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def batched_baseline() -> TrainingRegime:
    """Width 64, depth 2, 64 streams on CUDA with a cross-balanced curriculum."""
    return TrainingRegime(
        id="batching-grid-small-s64",
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=AgentSpec(
            semantic_pack=PACK, value_aggregation="masked_mean", attention_layers=2
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
                    device="cuda", wall_seconds=1740, memory_bytes=48 * GIB
                ),
            )
        ],
        wall_seconds=1770,
    )


def scale_ladder() -> TrainingRegime:
    """`batched_baseline` at 256 streams, sized by a money and wall-time allocation.

    `updates` is an upper bound: the allocation pauses training at its deadline.
    Without an iteration schedule the learning-rate and tau curves follow
    absolute iterations, so a prefix of the run is the same recipe at a smaller
    budget (experiments/scale-ladder.md).
    """
    regime = batched_baseline()
    regime.id = "scale-ladder-allies-lessons-v1"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 100_000
    stage.streams = 256
    stage.learning = move_learning()
    stage.execution = Execution(
        device="cuda", wall_seconds=400_000, memory_bytes=40 * GIB
    )
    regime.wall_seconds = 400_000
    regime.schedule_clock = "iteration_fraction"
    return TrainingRegime.model_validate(regime.model_dump())
