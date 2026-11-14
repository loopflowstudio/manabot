"""Small saved model configuration shared by admission and inference checks."""

from experiments.direct_self_play import direct_self_play
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym


def recipe() -> TrainingRegime:
    value = direct_self_play()
    value.world = managym.WORLD_VERSION
    value.id = f"test-{value.world}-{value.id}"
    value.stages = value.stages[:1]
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 3
    stage.transitions = 32
    stage.streams = 2
    # This fixture exercises exact-quota tails across updates; whole-game
    # both-seat recovery has its own collector and game-error coverage.
    stage.learning_seats = "one"
    stage.batch_closure = "exact-quota"
    stage.learning.epochs = 1
    stage.learning.ema = 0.9
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    value.recovery_max_microsteps = 10000
    value.schedule_clock = "iteration_fraction"
    return value
