"""Freeze the same learner across three worlds; no scientific execution."""

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
