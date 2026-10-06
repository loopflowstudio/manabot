"""Real CUDA optimizer/export checks run only where CUDA is available."""

import json
from pathlib import Path

import pytest
import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime
from manabot.verify.store import VerifyStore
from tests.remote.test_compile import ROOT


def test_unavailable_cuda_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = json.loads((ROOT / "experiments/regimes/direct-self-play.json").read_text())
    for stage in data["stages"]:
        stage["execution"]["device"] = "cuda"
    regime = TrainingRegime.model_validate(data)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="CPU fallback is forbidden"):
            execute_regime(regime, 197, tmp_path / "run", store)
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert run["status"] == "failed"
    assert not run["selected_artifact"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_cuda_continuation_exports_cpu_loadable_raw_and_ema(tmp_path: Path) -> None:
    data = json.loads((ROOT / "experiments/regimes/direct-self-play.json").read_text())
    for stage in data["stages"]:
        stage["execution"]["device"] = "cuda"
        stage["learning"]["ema"] = 0.9
    regime = TrainingRegime.model_validate(data)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 197, tmp_path / "run", store)
    assert run.status == "completed"
    for stage in run.stages:
        assert stage.actual_device.startswith("cuda")
        assert stage.optimizer_exposures > 0
        for name in ("raw", "ema"):
            model, _ = load_checkpoint_agent(stage.artifacts[name]["path"])
            assert next(model.parameters()).device.type == "cpu"
