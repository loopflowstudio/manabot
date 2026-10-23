"""Local policy search from a frozen policy (docs/local-policy-search.md)."""

from typing import Literal

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import gray_ogre_match
from manabot.training.regime import (
    CollectLocalUpdate,
    Learning,
    LocalSearchConfig,
    TrainingRegime,
    TrainSelfPlay,
    TrainSupervised,
)
import managym


def _student(
    id: str,
    target: Literal["local_soft", "local_argmax", "local_allocation"],
    datasets: list[str],
    initial: str,
    epochs: int,
) -> TrainSupervised:
    return TrainSupervised(
        id=id,
        operation="train_supervised",
        target=target,
        datasets=datasets,
        initial=initial,
        epochs=epochs,
        batch_size=64,
    )


def frozen_policy_local_update() -> TrainingRegime:
    """Local-update labels from a frozen policy, three targets, then a second round."""
    return TrainingRegime(
        id="frozen-policy-local-update",
        world=managym.WORLD_VERSION,
        match=gray_ogre_match(),
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=32,
                learning=Learning(value_lambda=1.0, epochs=1, minibatches=1),
            ),
            CollectLocalUpdate(
                id="labels",
                operation="collect_local_update",
                policy="policy",
                games=2,
                search=LocalSearchConfig(
                    worlds=2, full_probability=0.5, decision_seconds=5
                ),
            ),
            _student("local-soft", "local_soft", ["labels"], "policy", 2),
            _student("local-argmax", "local_argmax", ["labels"], "policy", 2),
            _student("local-allocation", "local_allocation", ["labels"], "policy", 2),
            CollectLocalUpdate(
                id="round-two",
                operation="collect_local_update",
                policy="local-soft",
                games=2,
                search=LocalSearchConfig(
                    depth=1, full_probability=1.0, decision_seconds=5
                ),
            ),
            _student(
                "cumulative", "local_soft", ["labels", "round-two"], "local-soft", 1
            ),
        ],
        wall_seconds=180,
    )
