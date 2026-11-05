"""The batched GPU baseline from the size and batching study, and its scale ladder."""

from typing import Literal

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import GIB, PACK, move_learning
from manabot.sim.matchup import selected_match
from manabot.training.regime import (
    BatchClosure,
    Execution,
    IterationSchedule,
    TrainingRegime,
    TrainSelfPlay,
)
import managym

# Rows per stream per update. Past the opening updates, a floor of 76 realized 106
# rows per stream in calibration (experiments/protocols/closure-lambda-v1.md).
CLOSURE_QUOTA = 106
CLOSURE_COMPLETE_FLOOR = 76


def batched_baseline() -> TrainingRegime:
    """Width 64, depth 2, 64 streams on CUDA with a cross-balanced curriculum."""
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
    stage.learning_seats = "one"
    stage.batch_closure = "exact-quota"
    stage.learning = move_learning()
    stage.execution = Execution(
        device="cuda", wall_seconds=400_000, memory_bytes=40 * GIB
    )
    regime.wall_seconds = 400_000
    regime.schedule_clock = "iteration_fraction"
    return TrainingRegime.model_validate(regime.model_dump())


def seat_comparison(seats: Literal["one", "both"], updates: int) -> TrainingRegime:
    """`scale_ladder` from scratch with complete games, matched per learner decision.

    Two-seat learns from one whole game per stream per update; one-seat from two,
    so both arms take about the same learner decisions per update. The absolute
    iteration LR/tau schedule and EMA therefore sit at about the same position at
    equal learner decisions, and Ataraxos minibatches of `streams` rows give the
    same optimizer steps per learner decision. Only `learning_seats` and
    `games_per_stream` differ between arms; recorded counters verify the rest.
    """
    regime = scale_ladder()
    regime.id = f"seat-comparison-{seats}-v1"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    stage.learning_seats = seats
    stage.batch_closure = "complete-games"
    stage.games_per_stream = 1 if seats == "both" else 2
    return TrainingRegime.model_validate(regime.model_dump())


def scale_ladder_both() -> TrainingRegime:
    """Current-world two-seat treatment; frozen w6 attempts retain their own recipe."""
    regime = scale_ladder()
    regime.id = "scale-ladder-allies-lessons-both-v2"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "both"
    stage.batch_closure = "complete-games"
    return TrainingRegime.model_validate(regime.model_dump())


def closure_comparison(closure: BatchClosure, updates: int) -> TrainingRegime:
    """One-seat `seat_comparison` differing only in when each batch closes.

    The complete arm's floor sits below the exact arm's quota by the calibrated
    mean overshoot of finishing the game in progress, so both arms take about the
    same rows per update and sit at the same point of the absolute iteration
    schedules at equal learner decisions. Game length moves with the policy, so
    recorded counters verify the match in each run.
    """
    regime = seat_comparison("one", updates)
    regime.id = f"batch-closure-{closure}-v1"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.games_per_stream = None
    stage.batch_closure = closure
    stage.transitions = (
        CLOSURE_QUOTA if closure == "exact-quota" else CLOSURE_COMPLETE_FLOOR
    )
    return TrainingRegime.model_validate(regime.model_dump())


def lambda_sweep(
    policy_lambda: float, value_lambda: float, updates: int
) -> TrainingRegime:
    """Two-seat complete-game `seat_comparison` differing only in its traces."""
    regime = seat_comparison("both", updates)
    regime.id = f"lambda-sweep-p{policy_lambda:g}-v{value_lambda:g}-v1"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning.policy_lambda = policy_lambda
    stage.learning.value_lambda = value_lambda
    return TrainingRegime.model_validate(regime.model_dump())
