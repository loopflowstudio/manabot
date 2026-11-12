"""Two models in one run: train a learner against a fixed reference, export both.

The smallest regime function with more than one model. `reference` is created
and never trained; `learner` trains against it frozen, and both are exported
through ordinary checkpoint admission.
"""

from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.recipes.builders import move_learning
from manabot.training.decks import UniformDecks
from manabot.training.regime import DampedPolicyLearning, Execution, Learning
from manabot.training.run import Run
from manabot.training.stages import self_play


def two_models(
    run: Run,
    *,
    decks: MatchConfig | UniformDecks,
    updates: int,
    learning: Learning | DampedPolicyLearning = move_learning(),
    reference: ModelConfig = ModelConfig(hidden_dim=32),
    learner: ModelConfig = ModelConfig(hidden_dim=64),
    streams: int = 4,
    transitions: int = 256,
    execution: Execution = Execution(),
) -> None:
    """Train `learner` against an untrained `reference` held frozen."""
    observation = ObservationConfig()
    fixed = run.model("reference", reference, observation)
    trained = run.model("learner", learner, observation)
    self_play(
        run,
        trained,
        opponent=fixed,
        decks=decks,
        updates=updates,
        learning=learning,
        streams=streams,
        transitions=transitions,
        execution=execution,
    )
    run.export(trained)
    run.export(fixed)
