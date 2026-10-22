"""Real CUDA optimizer/export checks run only where CUDA is available."""

import json
from pathlib import Path

import pytest
import torch

from experiments.direct_self_play import direct_self_play
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime
from manabot.verify.store import VerifyStore
import managym


def test_unavailable_cuda_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = direct_self_play().model_dump()
    for stage in data["stages"]:
        stage["execution"]["device"] = "cuda"
    regime = TrainingRegime.model_validate(data)
    regime.world = managym.WORLD_VERSION
    regime.id = f"test-{regime.world}-{regime.id}"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="CPU fallback is forbidden"):
            execute_regime(regime, 197, tmp_path / "run", store)
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert run["status"] == "failed"
    assert not run["selected_artifact"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_cuda_continuation_exports_cpu_loadable_raw_and_ema(tmp_path: Path) -> None:
    data = direct_self_play().model_dump()
    for stage in data["stages"]:
        stage["execution"]["device"] = "cuda"
        stage["learning"]["ema"] = 0.9
    regime = TrainingRegime.model_validate(data)
    regime.world = managym.WORLD_VERSION
    regime.id = f"test-{regime.world}-{regime.id}"
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 197, tmp_path / "run", store)
    assert run.status == "completed"
    for stage in run.stages:
        assert stage.actual_device.startswith("cuda")
        assert stage.optimizer_exposures > 0
        for name in ("raw", "ema"):
            model, _ = load_checkpoint(stage.artifacts[name]["path"])
            assert next(model.parameters()).device.type == "cpu"
