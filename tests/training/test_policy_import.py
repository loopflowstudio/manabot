"""Fixed-weight published fixtures: no optimizer or scientific cohort runs."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.training.learning import Learning
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    CollectBelief,
    ImportPolicy,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def reference(path: Path) -> ArtifactReference:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def published_fixture(
    tmp_path: Path, weights: Literal["raw", "ema"] = "raw"
) -> TrainingRegime:
    """Synthetic producer evidence explicitly records fixed untrained weights."""
    torch.set_num_threads(1)
    recipe = TrainingRegime(
        id="fixture-producer",
        world=managym.WORLD_VERSION,
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="producer",
                operation="train_self_play",
                updates=1,
                learning=Learning(ema=0.1),
            )
        ],
    )
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
        agent = ActorCritic(space, recipe.agent)
    path = tmp_path / "published.pt"
    save_checkpoint(
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


def test_declarative_weight_contracts(tmp_path: Path) -> None:
    recipe = published_fixture(tmp_path, "ema")
    histories = CollectBelief(
        id="histories",
        operation="collect_belief",
        policy="policy",
        weights="ema",
        games=3,
    )
    recipe.stages.append(histories)
    assert TrainingRegime.model_validate_json(recipe.model_dump_json()) == recipe
    histories.weights = "raw"
    with pytest.raises(ValueError, match="weights differ"):
        TrainingRegime.model_validate(recipe.model_dump())
