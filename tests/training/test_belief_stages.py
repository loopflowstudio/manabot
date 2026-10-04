"""Frozen policy → private whole-game dataset → sampler execution contracts."""

import json
from pathlib import Path
from typing import Literal

import pytest

from manabot.belief.sampling_data import read_dataset
from manabot.belief.sampling_fit import load_belief_sampler
from manabot.infra.hypers import AgentHypers, MatchHypers
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import (
    AtaraxosMoveLearning,
    CollectBelief,
    Execution,
    Learning,
    TrainBelief,
    TrainCompound,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym


def belief_recipe() -> TrainingRegime:
    return TrainingRegime(
        id="belief-stage-proof",
        world=managym.WORLD_VERSION,
        match=MatchHypers(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=AgentHypers(hidden_dim=8, num_attention_heads=2),
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


@pytest.mark.parametrize(
    "gradient,weights",
    [
        ("ppo", "raw"),
        ("ataraxos_move", "raw"),
        ("ataraxos_move", "ema"),
        ("compound", "raw"),
    ],
)
def test_frozen_belief_stages_run_and_reload(
    tmp_path: Path,
    gradient: Literal["ppo", "ataraxos_move", "compound"],
    weights: Literal["raw", "ema"],
) -> None:
    recipe = belief_recipe()
    policy_stage = recipe.stages[0]
    collection_stage = recipe.stages[1]
    assert isinstance(policy_stage, TrainSelfPlay)
    assert isinstance(collection_stage, CollectBelief)
    if gradient == "ataraxos_move":
        recipe.agent.value_kind = "categorical_wdl"
        policy_stage.learning = AtaraxosMoveLearning(
            gradient="ataraxos_move", min_advantage=0, advantage_quantile=0
        )
    collection_stage.weights = weights
    if gradient == "compound":
        recipe.agent.compound_decisions = True
        recipe.stages[0] = TrainCompound(
            id="policy",
            operation="train_compound",
            games_per_update=1,
            learning=Learning(epochs=1, minibatches=1),
        )
        recipe = TrainingRegime.model_validate(recipe.model_dump())
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 421, tmp_path / "run", store)
        assert run.status == "completed"
        policy, collection, fit = run.stages
        assert collection.games == 3
        assert collection.inputs["policy/raw"] == policy.artifacts["raw"]
        assert collection.inputs[f"policy/{weights}"] == policy.artifacts[weights]
        loaded, _ = load_checkpoint_agent(policy.artifacts[weights]["path"])
        assert loaded.hypers.value_kind == recipe.agent.value_kind
        assert loaded.hypers.compound_decisions == recipe.agent.compound_decisions
        assert fit.inputs["histories/dataset"] == collection.artifacts["dataset"]
        dataset = read_dataset(Path(collection.artifacts["dataset"]["path"]))
        assert {game.split for game in dataset.games} == {"train", "validation", "test"}
        assert len({game.game_id for game in dataset.games}) == 3
        assert dataset.policy_identity == policy.artifacts[weights]["sha256"]
        model = load_belief_sampler(
            Path(fit.artifacts["sampler"]["path"]),
            expected_dataset_identity=dataset.identity,
            expected_policy_identity=dataset.policy_identity,
            expected_schema_identity=dataset.schema.identity,
            expected_world_identity=dataset.world_identity,
            expected_checkpoint_identity=fit.artifacts["sampler"]["sha256"],
        )
        assert model.schema == dataset.schema
        assert fit.optimizer_exposures == 8
        assert fit.learning_seconds > 0
        assert collection.collection_seconds > 0
        assert fit.cumulative_seconds > collection.cumulative_seconds
        assert run.selected_artifact == policy.artifacts["raw"]
        assert store.training_run(run.id).stages[-1].artifacts == fit.artifacts


def test_compound_belief_recipe_rejects_missing_ema_and_flat_training() -> None:
    recipe = belief_recipe()
    recipe.agent.compound_decisions = True
    with pytest.raises(ValueError, match="compound stages"):
        TrainingRegime.model_validate(recipe.model_dump())
    recipe.stages[0] = TrainCompound(id="policy", operation="train_compound")
    collection = recipe.stages[1]
    assert isinstance(collection, CollectBelief)
    collection.weights = "ema"
    with pytest.raises(ValueError, match="EMA policy artifact"):
        TrainingRegime.model_validate(recipe.model_dump())
    collection.weights = "raw"
    recipe.stages.append(TrainSelfPlay(id="flat", operation="train_self_play"))
    with pytest.raises(ValueError, match="compound policy stages"):
        TrainingRegime.model_validate(recipe.model_dump())


def test_failed_belief_collection_keeps_frozen_policy(tmp_path: Path) -> None:
    recipe = belief_recipe()
    collection = recipe.stages[1]
    assert isinstance(collection, CollectBelief)
    collection.execution = Execution(wall_seconds=1e-9)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(TimeoutError):
            execute_regime(recipe, 421, tmp_path / "run", store)
        payload = json.loads((tmp_path / "run/run.json").read_text())
        assert payload["status"] == "interrupted"
        assert payload["stages"][0]["artifacts"]["raw"]["sha256"]
        assert payload["stages"][1]["status"] == "interrupted"
        assert not payload["stages"][1]["artifacts"]
