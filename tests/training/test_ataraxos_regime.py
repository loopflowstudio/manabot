"""Bounded real-engine recipe execution, admission and iteration contracts."""

from pathlib import Path

import pytest
import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]


def _recipe(name: str = "ataraxos-move") -> TrainingRegime:
    return TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{name}.json").read_text()
    )


def test_iteration_schedule_and_invalid_config() -> None:
    learning = AtaraxosMoveLearning(gradient="ataraxos_move")
    assert learning.rates(1) == (1e-4, 0.05)
    rate, tau = learning.rates(10000)
    assert rate == pytest.approx(0.5 / 10000**1.1)
    assert tau == pytest.approx(0.05 / 10000**0.3)
    assert learning.rates(10**9)[0] == 5e-6
    with pytest.raises(ValueError, match="positive"):
        learning.rates(0)
    with pytest.raises(ValueError):
        AtaraxosMoveLearning.model_validate({"gradient": "ataraxos_move", "epochs": 4})
    bad = _recipe().model_dump()
    bad["stages"][0]["learning"] = {"gradient": "ppo"}
    with pytest.raises(ValueError, match="categorical"):
        validate_regime(bad)
    bad = _recipe("ataraxos-move-scalar").model_dump()
    bad["stages"][1]["learning"] = {"gradient": "ppo", "ema": 0.999}
    with pytest.raises(ValueError, match="gradient"):
        validate_regime(bad)


@pytest.mark.parametrize("name", ["ataraxos-move", "ataraxos-move-scalar"])
def test_real_collection_update_export_reload(name: str, tmp_path: Path) -> None:
    recipe = _recipe(name)
    recipe.agent.hidden_dim = 8
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, AtaraxosMoveLearning)
        stage.transitions = 4
        stage.learning.min_advantage = 0
        stage.learning.advantage_quantile = 0
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 601, tmp_path / "run", store)
    assert run.status == "completed"
    for iteration, stage in enumerate(run.stages, 1):
        assert stage.learner_transitions == 16
        assert stage.optimizer_exposures == 16
        assert stage.collection_seconds > 0
        assert stage.learning_seconds > 0
        assert stage.export_seconds > 0
        assert stage.diagnostics[0]["iteration"] == iteration
        assert stage.diagnostics[0]["tau"] == pytest.approx(0.05 / iteration**0.3)
        for variant in ("raw", "ema"):
            loaded, _ = load_checkpoint_agent(stage.artifacts[variant]["path"])
            assert loaded.hypers.value_kind == recipe.agent.value_kind
            assert loaded.hypers.semantic_pack == "ur-lessons-vs-gw-allies"
            assert all(
                torch.isfinite(parameter).all() for parameter in loaded.parameters()
            )
    assert run.selected_artifact == run.stages[-1].artifacts["raw"]
