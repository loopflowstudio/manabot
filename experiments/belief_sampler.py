"""The learned belief sampler (docs/belief-sampler.md) and local search over it.

Both train a small frozen policy on the Gray Ogre matchup, collect its histories
and fit the belief sampler; `learned_belief_local_search` then searches with
sampled worlds and distils the labels into a student.
"""

from manabot.infra.config import ModelConfig
from manabot.training.builders import gray_ogre_match
from manabot.training.models import (
    CollectBelief,
    CollectLocalUpdate,
    Execution,
    Learning,
    LocalSearchConfig,
    Operation,
    TrainBelief,
    TrainingRegime,
    TrainSelfPlay,
    TrainSupervised,
)
import managym


def _belief_stages() -> list[Operation]:
    return [
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
    ]


def belief_sampler() -> TrainingRegime:
    """Train a frozen policy, collect its histories and fit the belief sampler."""
    return TrainingRegime(
        id="frozen-policy-belief-pilot",
        world=managym.WORLD_VERSION,
        match=gray_ogre_match(),
        agent=ModelConfig(hidden_dim=16, num_attention_heads=2),
        stages=_belief_stages(),
        wall_seconds=180,
    )


def learned_belief_local_search() -> TrainingRegime:
    """`belief_sampler` followed by belief-sampled local search and a student."""
    regime = belief_sampler()
    regime.id = "learned-belief-local-search-proof"
    regime.stages = [
        *_belief_stages(),
        CollectLocalUpdate(
            id="labels",
            operation="collect_local_update",
            policy="policy",
            games=8,
            max_steps=1000,
            search=LocalSearchConfig(depth=1, decision_seconds=5, sampling="learned"),
            sampler="belief",
            execution=Execution(wall_seconds=60),
        ),
        TrainSupervised(
            id="student",
            operation="train_supervised",
            target="local_soft",
            datasets=["labels"],
            initial="policy",
            epochs=1,
            batch_size=64,
            execution=Execution(wall_seconds=30),
        ),
    ]
    return TrainingRegime.model_validate(regime.model_dump())
