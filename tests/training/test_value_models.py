"""The value factorial freezes one learning rule and independent output heads."""

import json
from pathlib import Path

from pydantic import TypeAdapter
import pytest

from experiments.runners.run_value_models import smoke_plan
from experiments.runners.training_protocol import ResolvedStudy
from manabot.arena.models import PlayerRegistration, canonical_sha256, file_sha256
from manabot.training.models import (
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)


def test_value_factorial_protocol() -> None:
    plan = smoke_plan()
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert len(set(plan.protocol.regime_digests)) == 8
    recipes = [TrainingRegime.model_validate(row) for row in plan.recipes]
    assert {
        (r.agent.value_aggregation, r.agent.attention_layers, r.agent.value_kind)
        for r in recipes
    } == {
        (aggregation, depth, kind)
        for aggregation, depth in (
            ("historical_mean", 1),
            ("masked_mean", 1),
            ("value_token", 1),
            ("value_token", 2),
        )
        for kind in ("scalar", "categorical_wdl")
    }
    for recipe in recipes:
        assert recipe.agent.hidden_dim == 64
        assert recipe.agent.num_attention_heads == 4
        assert recipe.wall_seconds == 60
        for stage in recipe.stages:
            assert isinstance(stage, TrainSelfPlay)
            assert stage.learning.gradient == "ataraxos_move"
            assert stage.streams * stage.transitions * stage.updates == 256
            player_id = TypeAdapter(
                PlayerRegistration.model_fields["player_id"].rebuild_annotation()
            )
            for variant in ("raw", "ema"):
                player_id.validate_python(f"{recipe.id}-1061-{stage.id}-{variant}")


def test_recovery_preserves_source_and_requires_unchanged_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.runners import run_value_models as runner

    source = tmp_path / "failed"
    source.mkdir()
    plan = smoke_plan()
    recipe = TrainingRegime.model_validate(plan.recipes[0])
    checkpoint = source / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    run = TrainingRun(
        id="retained",
        regime=recipe,
        regime_digest=canonical_sha256(recipe.model_dump(mode="json")),
        seed=1061,
        seed_streams={},
        identities={},
        status="completed",
        stages=[
            StageRecord(
                id="policy-0",
                status="completed",
                artifacts={
                    "raw": {
                        "path": str(checkpoint),
                        "sha256": file_sha256(checkpoint),
                        "bytes": 7,
                    }
                },
            )
        ],
    )
    path = source / "run.json"
    path.write_text(run.model_dump_json())
    study = dict(
        study="value-models",
        status="failed",
        seconds=36.0,
        runs=[dict(path=str(path), sha256=file_sha256(path))],
        comparisons=[],
        measurements=[],
        protocol_sha256=canonical_sha256(plan.protocol.model_dump(mode="json")),
    )
    (source / "study.json").write_text(json.dumps(study))
    (source / "protocol.json").write_text(plan.protocol.model_dump_json())
    (source / "recipes.json").write_text(json.dumps(plan.recipes))
    (source / "resolved-plan.json").write_text(plan.model_dump_json())
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    called: list[bool] = []

    def resume_only(
        study: str, out: Path, retained: ResolvedStudy, *, resume: bool
    ) -> None:
        assert study == "value-models" and resume
        assert retained == plan
        assert json.loads((out / "study.json").read_text())["seconds"] == 36.0
        called.append(True)

    monkeypatch.setattr(runner, "run_study", resume_only)
    runner.recover_evaluation(source, tmp_path / "recovered")
    assert called == [True]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    provenance = json.loads((tmp_path / "recovered/recovery.json").read_text())
    assert provenance["remaining_seconds"] == 864
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint digest"):
        runner.recover_evaluation(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
