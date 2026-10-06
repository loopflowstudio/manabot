"""Recipe and receipt fixtures; no optimizer, games or benchmark timing runs."""

import json
from pathlib import Path
import sys
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.runners import distributed_workloads as workloads
from manabot.arena.models import canonical_sha256
from manabot.sim.net_opponent import RolloutBatch
from manabot.training.models import (
    Learning,
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore


@pytest.fixture
def recipe(tmp_path: Path) -> Path:
    regime = workloads._recipe(None, 110)
    regime.id = "representative-fixture"
    regime.agent.hidden_dim = 64
    regime.agent.num_attention_heads = 4
    regime.agent.value_kind = "scalar"
    regime.agent.value_aggregation = "masked_mean"
    for stage in regime.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.streams = 6
        stage.transitions = 32
        stage.updates = 3
    path = tmp_path / "recipe.json"
    path.write_text(regime.model_dump_json(indent=2))
    return path


@pytest.fixture(autouse=True)
def avoid_global_thread_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workloads.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(workloads.torch, "set_num_interop_threads", lambda _: None)


@pytest.mark.parametrize("exposures", [0, 96])
@pytest.mark.parametrize("status", ["completed", "failed"])
def test_train_propagates_recipe_and_reports_actual_work(
    recipe: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exposures: int,
    status: Literal["completed", "failed"],
) -> None:
    expected = TrainingRegime.model_validate_json(recipe.read_bytes())
    out = tmp_path / "workload"

    def execute(
        regime: TrainingRegime, seed: int, path: Path, store: VerifyStore
    ) -> TrainingRun:
        assert regime == expected
        assert seed == 10831
        assert path == out / "run"
        return TrainingRun(
            id="fixture",
            regime=regime,
            seed=seed,
            seed_streams={},
            identities={},
            regime_digest=canonical_sha256(regime.model_dump(mode="json")),
            status=status,
            seconds=4,
            stages=[
                StageRecord(
                    id=regime.stages[0].id,
                    status=status,
                    learner_transitions=192,
                    optimizer_exposures=exposures,
                    collection_seconds=2,
                    learning_seconds=1,
                    export_seconds=0.1,
                    diagnostics=[
                        {
                            "optimizer_exposures": exposures,
                            **(
                                {"skipped": "empty advantage filter"}
                                if not exposures
                                else {}
                            ),
                        }
                    ],
                )
            ],
        )

    monkeypatch.setattr(workloads, "execute_regime", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--mode",
            "train",
            "--out",
            str(out),
            "--seconds",
            "1",
            "--recipe",
            str(recipe),
            "--seed",
            "10831",
        ],
    )
    if status == "failed":
        with pytest.raises(RuntimeError, match="training status: failed"):
            workloads.main()
    else:
        workloads.main()
    result = json.loads((out / "measurement.json").read_text())
    assert result["status"] == status
    assert result["learner_transitions"] == 192
    assert (
        result["completed_iterations"] == 1
    )  # recipe requests six, receipt records one
    assert result["optimizer_exposures"] == exposures
    assert result["empty_filter_skips"] == (1 if exposures == 0 else 0)
    assert result["learning_seconds"] == 1  # nonzero even when no optimizer ran
    assert (
        TrainingRegime.model_validate_json((out / "recipe.json").read_bytes())
        == expected
    )
    identity = json.loads((out / "identity.json").read_text())
    assert identity["regime_digest"] == canonical_sha256(
        expected.model_dump(mode="json")
    )
    assert identity["memory_limit_enforced"] is False
    assert identity["requested_memory_bytes_by_stage"]["policy-0"] == 32 * 1024**3
    assert len(identity["training_source_sha256"]) == 64
    assert (out / "training-result.json").is_file()


def test_inference_uses_recipe_model_and_collection_geometry(
    recipe: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    constructor = MagicMock(return_value=agent)
    agent.eval.return_value = agent
    collector = MagicMock()
    values = np.zeros((32, 6), dtype=np.float32)
    collector.collect.return_value = RolloutBatch(
        obs={"fixture": np.zeros((32, 6, 7), dtype=np.float32)},
        actions=values,
        logprobs=values,
        rewards=values,
        dones=values,
        values=values,
        next_obs={},
        next_done=values[-1],
        probabilities=values,
    )
    collection = MagicMock(return_value=collector)
    monkeypatch.setattr(workloads, "Agent", constructor)
    monkeypatch.setattr(workloads, "SeatRoutedCollector", collection)
    clock = iter([0.0, 0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(workloads.time, "perf_counter", lambda: next(clock))
    out = tmp_path / "inference"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--mode",
            "inference",
            "--out",
            str(out),
            "--seconds",
            "1",
            "--recipe",
            str(recipe),
            "--seed",
            "10831",
        ],
    )
    workloads.main()
    assert constructor.call_args.args[1].hidden_dim == 64
    assert constructor.call_args.args[1].value_aggregation == "masked_mean"
    assert collection.call_args.kwargs["num_envs"] == 6
    assert collection.call_args.kwargs["seed"] == 10831
    assert collector.collect.call_args.args[1] == 32
    assert agent.call_count == 2  # warmup plus one timed fixture forward
    assert agent.call_args.args[0]["fixture"].shape == (192, 7)
    result = json.loads((out / "measurement.json").read_text())
    assert result["units"] == 192
    assert result["units_per_second"] == 96


@pytest.mark.parametrize(
    "change, message",
    [
        ({"world": "wrong"}, "world differs"),
        ({"wall_seconds": 241}, "supervisor timeout"),
        ({"surprise": True}, "Extra inputs"),
        ({"stages": []}, "at least 1"),
    ],
)
def test_invalid_recipe_rejected(
    recipe: Path, change: dict[str, object], message: str
) -> None:
    payload = json.loads(recipe.read_text())
    payload.update(change)
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        workloads._recipe(recipe, 240)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("threads", 2, "one CPU thread"),
        ("device", "mps", "cpu"),
        ("wall_seconds", 100, "stage wall_seconds"),
    ],
)
def test_unsupported_execution_rejected(
    recipe: Path, field: str, value: str | int, message: str
) -> None:
    payload = json.loads(recipe.read_text())
    payload["stages"][0]["execution"][field] = value
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        workloads._recipe(recipe, 110)


def test_unsupported_stage_rejected(recipe: Path) -> None:
    payload = json.loads(recipe.read_text())
    payload["stages"] = [{"id": "search", "operation": "collect_search", "games": 2}]
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-play stages"):
        workloads._recipe(recipe, 110)


def test_smoke_defaults_remain_tiny() -> None:
    regime = workloads._recipe(None, 110)
    assert regime.agent.hidden_dim == 16
    assert [
        (s.streams, s.transitions, s.updates)
        for s in regime.stages
        if isinstance(s, TrainSelfPlay)
    ] == [(4, 16, 1), (4, 16, 1)]


def test_ppo_recipe_and_original_bytes_preserved(recipe: Path) -> None:
    regime = TrainingRegime.model_validate_json(recipe.read_bytes())
    for stage in regime.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.learning = Learning(min_advantage=0.1)
    raw = regime.model_dump_json(indent=4).encode() + b"\n"
    recipe.write_bytes(raw)
    assert workloads._recipe(recipe, 110) == regime
    assert recipe.read_bytes() == raw


def test_malformed_json_rejected(recipe: Path) -> None:
    recipe.write_text("{broken")
    with pytest.raises(ValueError, match="Invalid JSON"):
        workloads._recipe(recipe, 110)


def test_ema_behavior_rejected_instead_of_using_current_self(recipe: Path) -> None:
    regime = TrainingRegime.model_validate_json(recipe.read_bytes())
    for stage in regime.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.behavior = "ema-self"
    recipe.write_text(regime.model_dump_json())
    with pytest.raises(ValueError, match="current-self"):
        workloads._recipe(recipe, 110)
