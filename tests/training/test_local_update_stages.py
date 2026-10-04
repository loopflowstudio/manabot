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
