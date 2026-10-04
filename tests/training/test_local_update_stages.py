"""Real-game local targets, same-root distillation and admitted policy reloads."""

import json
from pathlib import Path

import numpy as np
import pytest

from manabot.belief.likelihood import RulesProviderGap
from manabot.infra.hypers import AgentHypers, MatchHypers
from manabot.sim.distill import LOCAL_RECEIPT_KEY, LOCAL_TARGET_KEY, load_shards
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.local_update import LocalSearchConfig
from manabot.sim.search_supervised import _validate_dataset
from manabot.training.execution import execute_regime
from manabot.training.models import (
    CollectLocalUpdate,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
    TrainSupervised,
)
from manabot.verify.store import VerifyStore
import managym
from managym.possible_worlds import PossibleWorldSpace


def local_recipe() -> TrainingRegime:
    return TrainingRegime(
        id="local-update-proof",
        world=managym.WORLD_VERSION,
        match=MatchHypers(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=AgentHypers(hidden_dim=8, num_attention_heads=2),
        wall_seconds=90,
        stages=[
            TrainSelfPlay(
                id="policy",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=4,
                learning=Learning(epochs=1, minibatches=1),
            ),
            CollectLocalUpdate(
                id="labels",
                operation="collect_local_update",
                policy="policy",
                games=2,
                max_steps=1000,
                search=LocalSearchConfig(
                    depth=1, decision_seconds=5, sampling="compatible_prior"
                ),
            ),
            *[
                TrainSupervised(
                    id=target.replace("_", "-"),
                    operation="train_supervised",
                    initial="policy",
                    datasets=["labels"],
                    target=target,
                    epochs=1,
                    batch_size=64,
                )
                for target in ("local_soft", "local_argmax", "local_allocation")
            ],
        ],
    )


def test_local_stage_dependencies() -> None:
    recipe = local_recipe().model_dump()
    recipe["stages"][1]["policy"] = "missing"
    with pytest.raises(ValueError, match="earlier policy"):
        TrainingRegime.model_validate(recipe)
    recipe = local_recipe().model_dump()
    recipe["stages"][2]["target"] = "visit_distribution"
    with pytest.raises(ValueError, match="semantics"):
        TrainingRegime.model_validate(recipe)


def test_complete_games_and_three_distillation_controls(tmp_path: Path) -> None:
    with VerifyStore(tmp_path / "verify.sqlite") as store:
        run = execute_regime(local_recipe(), 712, tmp_path / "run", store)
    assert run.status == "completed"
    labels = run.stages[1]
    assert labels.games == 2
    paths = [
        item["path"]
        for key, item in labels.artifacts.items()
        if key.startswith("game-")
    ]
    dataset = load_shards(paths)
    assert set(dataset["game_index"]) == {0, 1}
    assert set(dataset["seat"]) == {0, 1}
    assert np.all(dataset["winner"] >= 0)
    for row in dataset[LOCAL_RECEIPT_KEY]:
        receipt = json.loads(str(row))
        assert receipt["policy_sha256"] == run.stages[0].artifacts["raw"]["sha256"]
    original_targets = dataset[LOCAL_TARGET_KEY].copy()
    for kind in ("local_soft", "local_argmax", "local_allocation"):
        targets = _validate_dataset(
            dataset, policy_target_kind=kind, value_target_kind="terminal_outcome"
        )
        assert targets is not None
        np.testing.assert_allclose(targets.sum(axis=1), 1, atol=1e-7)
        for index, encoded in enumerate(dataset[LOCAL_RECEIPT_KEY]):
            receipt = json.loads(str(encoded))
            count = len(receipt["target"])
            if kind == "local_soft":
                expected = receipt["target"]
            elif kind == "local_argmax":
                expected = np.eye(count)[np.argmax(receipt["values"])]
            else:
                counts = np.asarray(receipt["allocation_counts"])
                expected = counts / counts.sum()
            np.testing.assert_allclose(targets[index, :count], expected)
        np.testing.assert_array_equal(dataset[LOCAL_TARGET_KEY], original_targets)
    for record in run.stages[2:]:
        model, _ = load_checkpoint_agent(record.artifacts["raw"]["path"])
        assert model.world_binding["world"] == managym.WORLD_VERSION
        assert record.optimizer_exposures > 0
    corrupt = {key: value.copy() for key, value in dataset.items()}
    corrupt[LOCAL_TARGET_KEY][0] = 0
    with pytest.raises(ValueError, match="differs"):
        _validate_dataset(
            corrupt,
            policy_target_kind="local_soft",
            value_target_kind="terminal_outcome",
        )


def test_exact_history_gap_retains_failed_attempt(tmp_path: Path) -> None:

    recipe = local_recipe()
    collection = recipe.stages[1]
    assert isinstance(collection, CollectLocalUpdate)
    collection.search = collection.search.model_copy(update={"sampling": "belief"})
    with VerifyStore(tmp_path / "failed.sqlite") as store:
        with pytest.raises(RulesProviderGap, match="exact local posterior unavailable"):
            execute_regime(recipe, 712, tmp_path / "failed", store)
    manifest = json.loads((tmp_path / "failed" / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["seconds"] > 0
    assert list((tmp_path / "failed").glob("*.receipts.jsonl"))


def test_learned_regime_requires_same_frozen_policy_and_sampler() -> None:
    recipe = json.loads(
        Path("experiments/regimes/learned-belief-local-search.json").read_text()
    )
    TrainingRegime.model_validate(recipe)
    recipe["stages"][3]["sampler"] = "histories"
    with pytest.raises(ValueError, match="train_belief"):
        TrainingRegime.model_validate(recipe)
    recipe["stages"][3]["sampler"] = None
    with pytest.raises(ValueError, match="sampler stage"):
        TrainingRegime.model_validate(recipe)


def test_learned_pipeline_never_enumerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def reject_enumeration(*args: object, **kwargs: object) -> None:
        raise AssertionError("learned pipeline attempted exact support enumeration")

    monkeypatch.setattr(PossibleWorldSpace, "from_engine", reject_enumeration)
    recipe = TrainingRegime.model_validate_json(
        Path("experiments/regimes/learned-belief-local-search.json").read_text()
    )
    with VerifyStore(tmp_path / "learned.sqlite") as store:
        run = execute_regime(recipe, 718, tmp_path / "learned", store)
    assert run.status == "completed"
    assert run.stages[1].games == 3
    assert run.stages[3].games == 8
    assert run.stages[4].optimizer_exposures > 0
    for name, artifact in run.stages[3].artifacts.items():
        if not name.startswith("game-"):
            continue
        dataset = load_shards([artifact["path"]])
        for encoded in dataset[LOCAL_RECEIPT_KEY]:
            receipt = json.loads(str(encoded))
            assert (
                receipt["learned_belief"]["artifact"]["sha256"]
                == run.stages[2].artifacts["sampler"]["sha256"]
            )
            assert receipt["policy_sha256"] == run.stages[0].artifacts["raw"]["sha256"]
    agent, _ = load_checkpoint_agent(run.stages[4].artifacts["raw"]["path"])
    assert agent.world_binding["world"] == managym.WORLD_VERSION
