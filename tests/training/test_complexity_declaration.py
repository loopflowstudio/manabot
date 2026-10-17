"""Freeze the same learner across three worlds; no scientific execution."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.runners.environment_complexity import calibrate, regime, roster


def test_pool_recipes_only_change_declared_environment() -> None:
    recipes = [regime(pool, updates=100, seconds=600) for pool in ("l1", "l2", "l3")]
    common = []
    for recipe in recipes:
        value = recipe.model_dump()
        value.pop("id")
        value.pop("match")
        common.append(value)
        assert recipe.stages[0].matchup_curriculum == "cross-balanced"
        assert recipe.stages[0].streams * recipe.stages[0].transitions == 256
    assert common[0] == common[1] == common[2]
    for pool, recipe in zip(("l1", "l2", "l3"), recipes, strict=True):
        assert recipe.match == roster(pool).match()
        assert recipe.match.content_pack == recipe.agent.semantic_pack


def test_calibration_rejects_unowned_local_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ETUDE_RUN", raising=False)
    with pytest.raises(ValueError, match="Etude-owned"):
        calibrate()


def test_gpu_pool_recipes_share_64_stream_learning_contract() -> None:
    """GPU preparation does not silently change historical recipes or horizons."""
    common: list[dict[str, object]] = []
    for pool in ("l1", "l2", "l3"):
        recipe = regime(pool, updates=100, seconds=600, streams=64, device="cuda")
        stage = recipe.stages[0]
        assert stage.streams == 64
        assert stage.transitions == 64
        assert stage.execution.device == "cuda"
        assert stage.streams * stage.transitions == 4096
        value = recipe.model_dump()
        value.pop("id")
        value.pop("match")
        common.append(value)
    assert common[0] == common[1] == common[2]
    historical = regime("l1", updates=100, seconds=600)
    assert historical.stages[0].streams == 4
    assert historical.stages[0].execution.device == "cpu"


def test_gpu_calibration_requires_cuda_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setenv("ETUDE_DEADLINE", "2099-01-01T00:00:00+00:00")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA; no CPU fallback"):
        calibrate(target="gpu")


def test_mini_calibration_remains_remote_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "fixture-only")
    monkeypatch.setenv("ETUDE_DEADLINE", "2099-01-01T00:00:00+00:00")
    monkeypatch.setattr("socket.gethostname", lambda: "local-fixture")
    with pytest.raises(ValueError, match="Mini-only"):
        calibrate()


def test_gpu_calibration_selects_declared_recipe_and_retains_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No learning runs: inspect the real boundary at the executor call."""
    for key, value in {
        "ETUDE_RUN": "fixture-only",
        "ETUDE_DEADLINE": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "ETUDE_SEED": "12250",
        "ETUDE_OUTPUT_DIR": str(tmp_path),
        "ETUDE_TRACK_DIR": str(tmp_path / "tracking"),
        "ETUDE_STOP_FILE": str(tmp_path / "stop"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    executor = Mock(side_effect=RuntimeError("fixture stop before learning"))
    monkeypatch.setattr(
        "experiments.runners.environment_complexity.execute_regime", executor
    )
    with pytest.raises(RuntimeError, match="fixture stop"):
        calibrate(target="gpu")
    expected = regime("l1", updates=100, seconds=1200, streams=64, device="cuda")
    assert executor.call_args.args[0] == expected
    assert executor.call_args.args[1] == 12250
    receipt = json.loads((tmp_path / "calibration.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["streams"] == 64
    assert receipt["device"] == "cuda"
    assert receipt["pools"] == []
    assert not (tmp_path / "etude-result.json").exists()
    assert os.environ["ETUDE_RUN"] == "fixture-only"
    assert os.environ["ETUDE_TRACK_DIR"] == str(tmp_path / "tracking")
