"""The learned belief sampler (docs/search-and-beliefs.md#belief-sampler).

Trains a small frozen policy on the Gray Ogre matchup, collects its histories
and fits the belief sampler.
"""

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import gray_ogre_match
from manabot.training.regime import (
    CollectBelief,
    Execution,
    Learning,
    TrainBelief,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def belief_sampler() -> TrainingRegime:
    """Train a frozen policy, collect its histories and fit the belief sampler."""
    return TrainingRegime(
        id="frozen-policy-belief-pilot",
        world=managym.WORLD_VERSION,
        match=gray_ogre_match(),
        agent=ModelConfig(hidden_dim=16, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=16,
                learning=Learning(epochs=1, minibatches=1),
                execution=Execution(wall_seconds=60),
            ),
            CollectBelief(
                id="histories",
                operation="collect_belief",
                policy="policy",
                games=3,
                max_steps=1000,
                execution=Execution(wall_seconds=60),
            ),
            TrainBelief(
                id="belief",
                operation="train_belief",
                dataset="histories",
                steps=4,
                batch_size=8,
                hidden_size=16,
                history_dropout=0.0,
                evaluation_samples=4,
                execution=Execution(wall_seconds=60),
            ),
        ],
        wall_seconds=180,
    )
