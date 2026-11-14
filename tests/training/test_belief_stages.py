"""Frozen policy → private whole-game dataset → sampler execution contracts."""

import json

import pytest

from manabot.infra.config import MatchConfig, ModelConfig
from manabot.training.learning import Learning
from manabot.training.regime import (
    CollectBelief,
    TrainBelief,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def belief_recipe() -> TrainingRegime:
    return TrainingRegime(
        id="belief-stage-proof",
        world=managym.WORLD_VERSION,
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2),
        wall_seconds=120,
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=4,
                learning=Learning(epochs=1, minibatches=1),
            ),
            CollectBelief(
                id="histories",
                operation="collect_belief",
                policy="policy",
                games=3,
                max_steps=1000,
            ),
            TrainBelief(
                id="belief",
                operation="train_belief",
                dataset="histories",
                steps=2,
                batch_size=4,
                hidden_size=8,
                evaluation_samples=2,
            ),
        ],
    )


def test_belief_dependencies_reject_forward_and_wrong_artifacts() -> None:
    recipe = belief_recipe()
    collection = recipe.stages[1]
    assert isinstance(collection, CollectBelief)
    collection.policy = "belief"
    with pytest.raises(ValueError, match="earlier policy"):
        TrainingRegime.model_validate(recipe.model_dump())
    collection.policy = "policy"
    collection.weights = "ema"
    with pytest.raises(ValueError, match="EMA"):
        TrainingRegime.model_validate(recipe.model_dump())
    collection.weights = "raw"
    fit = recipe.stages[2]
    assert isinstance(fit, TrainBelief)
    fit.dataset = "policy"
    with pytest.raises(ValueError, match="earlier belief collection"):
        TrainingRegime.model_validate(recipe.model_dump())


def test_removed_compound_policies_fail_validation_clearly() -> None:
    """Saved compound regimes never load as flat policies or unknown stages."""
    ordinary = belief_recipe().model_dump(mode="json")
    assert "compound_decisions" not in ordinary["agent"]
    explicit_off = json.loads(json.dumps(ordinary))
    explicit_off["agent"]["compound_decisions"] = False
    assert TrainingRegime.model_validate(explicit_off) == TrainingRegime.model_validate(
        ordinary
    )
    compound_agent = json.loads(json.dumps(ordinary))
    compound_agent["agent"]["compound_decisions"] = True
    with pytest.raises(ValueError, match="no longer executable.*ba3cd58de"):
        TrainingRegime.model_validate(compound_agent)
    compound_stage = json.loads(json.dumps(ordinary))
    compound_stage["stages"][0] = {"id": "policy", "operation": "train_compound"}
    with pytest.raises(ValueError, match="no longer executable.*ba3cd58de"):
        TrainingRegime.model_validate(compound_stage)
    with pytest.raises(ValueError, match="no longer executable"):
        ModelConfig.model_validate({"compound_decisions": True})
    assert ModelConfig.model_validate({"compound_decisions": False}) == ModelConfig()
