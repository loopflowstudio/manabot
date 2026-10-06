"""Fixed-weight published fixtures: no optimizer or scientific cohort runs."""

import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.belief.sampling_data import read_dataset
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentSpec, MatchHypers
from manabot.model.agent import Agent
from manabot.sim.distill import LOCAL_TARGET_KEY, load_shards, save_bc_checkpoint
from manabot.sim.local_update import LocalSearchConfig
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import (
    ArtifactReference,
    CollectBelief,
    CollectLocalUpdate,
    ImportPolicy,
    Learning,
    ProducerCost,
    StageRecord,
    TrainCompound,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
    TrainSupervised,
)
from manabot.verify.store import VerifyStore
import managym


def reference(path: Path) -> ArtifactReference:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def published_fixture(
    tmp_path: Path, weights: Literal["raw", "ema"] = "raw", *, compound: bool = False
) -> TrainingRegime:
    """Synthetic producer evidence explicitly records fixed untrained weights."""
    torch.set_num_threads(1)
    recipe = TrainingRegime(
        id="fixture-producer",
        world=managym.WORLD_VERSION,
        match=MatchHypers(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=AgentSpec(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="producer",
                operation="train_self_play",
                updates=1,
                learning=Learning(ema=0.1),
            )
        ],
    )
    if compound:
        recipe.agent.compound_decisions = True
        recipe.stages = [TrainCompound(id="producer", operation="train_compound")]
    source = TrainingRun(
        id="synthetic-untrained-producer",
        regime=recipe,
        regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
        seed=112,
        seed_streams={},
        status="completed",
        seconds=12.5,
        identities={"source_commit": "fixture", "training_source_sha256": "fixture"},
    )
    space = ObservationSpace(recipe.observation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(112)
        agent = Agent(space, recipe.agent)
    path = tmp_path / "published.pt"
    save_bc_checkpoint(
        agent,
        space,
        path,
        player_configs=Match(recipe.match).to_rust(),
        extra={
            "run_id": source.id,
            "stage_id": "producer",
            "weights": weights,
            "regime_digest": source.regime_digest,
            "value_semantic": "signed_outcome",
            "averaging": {
                "clock": "collect-update-iteration",
                "iteration": 1,
                "rate": 0.1,
            }
            if weights == "ema"
            else None,
        },
    )
    source.stages = [
        StageRecord(
            id="producer",
            status="completed",
            cumulative_seconds=12.0,
            artifacts={weights: reference(path)},
        )
    ]
    export = tmp_path / "producer.json"
    atomic_json(export, source.model_dump(mode="json"))
    return recipe.model_copy(
        update={
            "id": "reuse",
            "stages": [
                ImportPolicy(
                    id="policy",
                    operation="import_policy",
                    source_run=reference(export),
                    source_stage="producer",
                    checkpoint=reference(path),
                    weights=weights,
                )
            ],
        }
    )


@pytest.mark.parametrize(
    "weights,compound", [("raw", False), ("ema", False), ("raw", True)]
)
def test_saved_policy_consumers_and_costs(
    tmp_path: Path, weights: Literal["raw", "ema"], compound: bool
) -> None:
    recipe = published_fixture(tmp_path, weights, compound=compound)
    imported = recipe.stages[0]
    assert isinstance(imported, ImportPolicy)
    recipe.stages.extend(
        [
            CollectBelief(
                id="histories",
                operation="collect_belief",
                policy="policy",
                weights=weights,
                games=3,
                max_steps=1000,
            ),
            CollectLocalUpdate(
                id="labels",
                operation="collect_local_update",
                policy="policy",
                weights=weights,
                games=2,
                max_steps=1000,
                search=LocalSearchConfig(depth=1, decision_seconds=5),
            ),
        ]
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 112, tmp_path / "run", store)
        assert run.status == "completed", run.error
        assert store.training_run(run.id) == run
    admission = run.stages[0]
    assert admission.producer_cost is not None
    assert admission.producer_cost.cumulative_seconds == 12.0
    assert admission.producer_cost.weights == weights
    assert admission.seconds > 0 and admission.export_seconds > 0
    assert admission.learning_seconds == admission.collection_seconds == 0
    assert all(s.optimizer_exposures == 0 for s in run.stages)
    assert run.stages[1].games == 3
    assert run.stages[2].games == 2
    assert ("raw" in admission.artifacts) == (weights == "raw")
    assert (run.selected_artifact is None) == (weights == "ema")
    dataset = read_dataset(Path(run.stages[1].artifacts["dataset"]["path"]))
    assert dataset.policy_identity == imported.checkpoint["sha256"]
    shards = [
        a["path"]
        for name, a in run.stages[2].artifacts.items()
        if name.startswith("game-")
    ]
    assert load_shards(shards)[LOCAL_TARGET_KEY].shape[0] > 0
    # The import is self-contained after the original publication is unavailable.
    Path(imported.checkpoint["path"]).unlink()
    assert (
        file_sha256(admission.artifacts[weights]["path"])
        == imported.checkpoint["sha256"]
    )
    assert (
        file_sha256(admission.artifacts["source_run"]["path"])
        == imported.source_run["sha256"]
    )


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing", "FileNotFoundError"),
        ("checkpoint_hash", "hash/size"),
        ("receipt_hash", "hash/size"),
        ("model", "model/observation/world"),
        ("observation", "model/observation/world"),
        ("setup", "setup differs"),
        ("weight", "weight artifact"),
        ("metadata", "metadata mismatch"),
        ("cost", "cost is unavailable"),
        ("provenance", "source provenance"),
        ("incomplete", "completed producer"),
        ("ema_identity", "averaging identity"),
    ],
)
def test_rejects_and_persists_failed_admission(
    tmp_path: Path, fault: str, match: str
) -> None:
    recipe = published_fixture(tmp_path, "ema" if fault == "ema_identity" else "raw")
    stage = recipe.stages[0]
    assert isinstance(stage, ImportPolicy)
    path = Path(stage.checkpoint["path"])
    export = Path(stage.source_run["path"])
    source = TrainingRun.model_validate_json(export.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "checkpoint_hash":
        path.write_bytes(b"tampered")
    elif fault == "receipt_hash":
        export.write_text("{}")
    elif fault == "model":
        recipe.agent.hidden_dim = 16
    elif fault == "observation":
        recipe.observation.max_cards_per_player += 1
    elif fault == "setup":
        recipe.match.hero_deck = {"Mountain": 16}
    elif fault == "weight":
        stage.weights = "ema"
    elif fault in {"metadata", "ema_identity"}:
        payload = torch.load(path, weights_only=False)
        if fault == "metadata":
            payload["bc"]["run_id"] = "different-producer"
        else:
            payload["bc"]["averaging"]["rate"] = 0.2
        torch.save(payload, path)
        stage.checkpoint = reference(path)
        source.stages[0].artifacts[stage.weights] = reference(path)
    elif fault == "cost":
        source.stages[0].cumulative_seconds = None
    elif fault == "provenance":
        source.identities = {}
    elif fault == "incomplete":
        source.stages[0].status = "failed"
    if fault in {"metadata", "ema_identity", "cost", "provenance", "incomplete"}:
        atomic_json(export, source.model_dump(mode="json"))
        stage.source_run = reference(export)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises((ValueError, FileNotFoundError)):
            execute_regime(recipe, 112, tmp_path / "run", store)
        run = TrainingRun.model_validate_json((tmp_path / "run/run.json").read_text())
        assert run.status == "failed"
        assert match in (run.error or "")
        assert store.training_run(run.id) == run
    assert run.stages[0].status == "failed"
    assert run.stages[0].inputs["source_run"] == stage.source_run
    assert not run.stages[0].artifacts


def test_declarative_weight_and_continuation_contracts(tmp_path: Path) -> None:
    recipe = published_fixture(tmp_path, "ema")
    labels = CollectLocalUpdate(
        id="labels",
        operation="collect_local_update",
        policy="policy",
        weights="ema",
        games=2,
    )
    fit = TrainSupervised(
        id="student",
        operation="train_supervised",
        initial="policy",
        initial_weights="ema",
        datasets=["labels"],
        target="local_soft",
    )
    recipe.stages.extend([labels, fit])
    assert TrainingRegime.model_validate_json(recipe.model_dump_json()) == recipe
    fit.initial_weights = "raw"
    with pytest.raises(ValueError, match="initial weights"):
        TrainingRegime.model_validate(recipe.model_dump())
    fit.initial_weights = "ema"
    labels.weights = "raw"
    with pytest.raises(ValueError, match="weights differ"):
        TrainingRegime.model_validate(recipe.model_dump())


def test_pre_import_supervised_export_preserves_original_identity(
    tmp_path: Path,
) -> None:
    """Pre-ETU-112 supervised receipts have no initial_weights/producer_cost fields."""
    recipe = published_fixture(tmp_path)
    stage = recipe.stages[0]
    assert isinstance(stage, ImportPolicy)
    export = Path(stage.source_run["path"])
    # Construct the historical wire shape before binding its checkpoint/export.
    source = json.loads(export.read_text())
    source["regime"]["stages"] = [
        {
            "id": "teacher",
            "operation": "collect_search",
            "games": 2,
            "execution": source["regime"]["stages"][0]["execution"],
            "policy": "uniform-prior-determinized-puct",
            "target": "visit_distribution",
            "simulations": 64,
            "worlds": 4,
            "max_steps": 2000,
        },
        {
            "id": "producer",
            "operation": "train_supervised",
            "trainer": "search_supervised",
            "optimizer": "adam",
            "trainable": "policy",
            "target": "visit_distribution",
            "datasets": ["teacher"],
            "initial": None,
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.001,
            "execution": source["regime"]["stages"][0]["execution"],
        },
    ]
    source["stages"][0].pop("producer_cost")
    source["regime_digest"] = canonical_sha256(source["regime"])
    path = Path(stage.checkpoint["path"])
    payload = torch.load(path, weights_only=False)
    payload["bc"]["regime_digest"] = source["regime_digest"]
    payload["bc"]["value_semantic"] = "win_logit"
    torch.save(payload, path)
    stage.checkpoint = reference(path)
    source["stages"][0]["artifacts"]["raw"] = reference(path)
    atomic_json(export, source)
    stage.source_run = reference(export)
    frozen_receipt, frozen_checkpoint = export.read_bytes(), path.read_bytes()
    parsed = TrainingRun.model_validate(source)
    assert parsed.regime.stages[1].initial_weights == "raw"
    assert (
        canonical_sha256(parsed.regime.model_dump(mode="json"))
        == source["regime_digest"]
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 112, tmp_path / "run", store)
    assert run.status == "completed", run.error
    assert (
        export.read_bytes() == frozen_receipt and path.read_bytes() == frozen_checkpoint
    )


def test_original_wire_digest_precedes_default_normalization(tmp_path: Path) -> None:
    recipe = published_fixture(tmp_path)
    stage = recipe.stages[0]
    assert isinstance(stage, ImportPolicy)
    export = Path(stage.source_run["path"])
    source = json.loads(export.read_text())
    # Explicitly serialized historical off contract is normalized away today.
    source["regime"]["observation"]["policy_history_version"] = 0
    source["regime_digest"] = canonical_sha256(source["regime"])
    path = Path(stage.checkpoint["path"])
    payload = torch.load(path, weights_only=False)
    payload["bc"]["regime_digest"] = source["regime_digest"]
    torch.save(payload, path)
    stage.checkpoint = reference(path)
    source["stages"][0]["artifacts"]["raw"] = reference(path)
    atomic_json(export, source)
    stage.source_run = reference(export)
    assert (
        canonical_sha256(
            TrainingRun.model_validate(source).regime.model_dump(mode="json")
        )
        != source["regime_digest"]
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 112, tmp_path / "run", store)
    assert run.status == "completed", run.error


def test_nested_producer_cost_cannot_be_lost(tmp_path: Path) -> None:
    recipe = published_fixture(tmp_path)
    stage = recipe.stages[0]
    assert isinstance(stage, ImportPolicy)
    export = Path(stage.source_run["path"])
    source = TrainingRun.model_validate_json(export.read_text())
    # No training: a synthetic A-import -> B-producer receipt exercises accounting.
    ancestor = stage.model_copy(update={"id": "ancestor"})
    source.regime.stages.insert(0, ancestor)
    source.regime_digest = canonical_sha256(source.regime.model_dump(mode="json"))
    source.stages.insert(
        0,
        StageRecord(
            id="ancestor",
            status="completed",
            producer_cost=ProducerCost(
                run_id="ancestor-run",
                stage_id="ancestor-policy",
                weights="raw",
                cumulative_seconds=100,
            ),
        ),
    )
    atomic_json(export, source.model_dump(mode="json"))
    stage.source_run = reference(export)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="nested published producer costs"):
            execute_regime(recipe, 112, tmp_path / "run", store)
    run = TrainingRun.model_validate_json((tmp_path / "run/run.json").read_text())
    assert run.status == "failed"
    assert "nested published producer costs" in (run.error or "")
    assert run.stages[0].producer_cost is None
