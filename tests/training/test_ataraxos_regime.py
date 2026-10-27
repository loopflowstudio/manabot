"""Bounded real-engine recipe execution, admission and iteration contracts."""

<<<<<<< HEAD
from pathlib import Path

import pytest
import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime, TrainSelfPlay
=======
from collections.abc import Callable
from pathlib import Path

from etude.train.read import read_all
import pytest
import torch

from experiments.ataraxos_move import ataraxos_move, ataraxos_move_scalar
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.regime import DampedPolicyLearning, TrainingRegime, TrainSelfPlay
from manabot.training.update_metrics import read_updates, stage_rows
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.verify.store import VerifyStore
import managym

ROOT = Path(__file__).resolve().parents[2]


<<<<<<< HEAD
def _recipe(name: str = "ataraxos-move") -> TrainingRegime:
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{name}.json").read_text()
    )
=======
def _recipe(build: Callable[[], TrainingRegime] = ataraxos_move) -> TrainingRegime:
    recipe = build()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    recipe.world = managym.WORLD_VERSION
    recipe.id = f"test-{recipe.world}-{recipe.id}"
    return recipe


def test_iteration_schedule_and_invalid_config() -> None:
<<<<<<< HEAD
    learning = AtaraxosMoveLearning(gradient="ataraxos_move")
=======
    learning = DampedPolicyLearning(gradient="damped_policy")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    assert learning.rates(1) == (1e-4, 0.05)
    rate, tau = learning.rates(10000)
    assert rate == pytest.approx(0.5 / 10000**1.1)
    assert tau == pytest.approx(0.05 / 10000**0.3)
    assert learning.rates(10**9)[0] == 5e-6
    with pytest.raises(ValueError, match="positive"):
        learning.rates(0)
    with pytest.raises(ValueError):
<<<<<<< HEAD
        AtaraxosMoveLearning.model_validate({"gradient": "ataraxos_move", "epochs": 4})
=======
        DampedPolicyLearning.model_validate({"gradient": "damped_policy", "epochs": 4})
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    bad = _recipe().model_dump()
    bad["stages"][0]["learning"] = {"gradient": "ppo"}
    with pytest.raises(ValueError, match="categorical"):
        validate_regime(bad)
<<<<<<< HEAD
    bad = _recipe("ataraxos-move-scalar").model_dump()
=======
    bad = _recipe(ataraxos_move_scalar).model_dump()
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    bad["stages"][1]["learning"] = {"gradient": "ppo", "ema": 0.999}
    with pytest.raises(ValueError, match="gradient"):
        validate_regime(bad)


<<<<<<< HEAD
@pytest.mark.parametrize("name", ["ataraxos-move", "ataraxos-move-scalar"])
def test_real_collection_update_export_reload(name: str, tmp_path: Path) -> None:
    recipe = _recipe(name)
    recipe.agent.hidden_dim = 8
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, AtaraxosMoveLearning)
=======
@pytest.mark.parametrize("build", [ataraxos_move, ataraxos_move_scalar])
def test_real_collection_update_export_reload(
    build: Callable[[], TrainingRegime], tmp_path: Path
) -> None:
    recipe = _recipe(build)
    recipe.agent.hidden_dim = 8
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, DampedPolicyLearning)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        stage.transitions = 4
        stage.learning.min_advantage = 0
        stage.learning.advantage_quantile = 0
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 601, tmp_path / "run", store)
    assert run.status == "completed"
<<<<<<< HEAD
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
=======
    updates = stage_rows(run, read_updates(tmp_path, [run]))
    for iteration, stage in enumerate(run.stages, 1):
        rows = updates[stage.id]
        assert stage.learner_transitions == sum(d["rows"] for d in rows)
        assert stage.learner_transitions >= 16
        assert stage.optimizer_exposures == stage.learner_transitions
        assert stage.collection_seconds > 0
        assert stage.learning_seconds > 0
        assert stage.export_seconds > 0
        assert rows[0]["iteration"] == iteration
        assert rows[0]["tau"] == pytest.approx(0.05 / iteration**0.3)
        for variant in ("raw", "ema"):
            loaded, _ = load_checkpoint(stage.artifacts[variant]["path"])
            assert loaded.config.value_kind == recipe.agent.value_kind
            assert loaded.config.semantic_pack == "ur-lessons-vs-gw-allies"
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            assert all(
                torch.isfinite(parameter).all() for parameter in loaded.parameters()
            )
    assert run.selected_artifact == run.stages[-1].artifacts["raw"]
<<<<<<< HEAD
=======
    tracked = read_all(tmp_path / "run" / recipe.stages[0].id / "etude")
    assert len(tracked) == 1
    assert tracked[0].state is not None and tracked[0].state.state == "finished"
    assert tracked[0].incomplete == 0
    assert len(tracked[0].checkpoints) == len(recipe.stages) + 1
    assert [c.at["learner_transitions"] for c in tracked[0].checkpoints] == [
        0,
        run.stages[0].learner_transitions,
        sum(s.learner_transitions for s in run.stages),
    ]
    assert [c.at["updates"] for c in tracked[0].checkpoints] == [0, 1, 2]


def test_sdk_retains_initialization_and_exact_update_milestones(tmp_path: Path) -> None:
    recipe = _recipe(ataraxos_move_scalar)
    recipe.agent.hidden_dim = 8
    recipe.stages = recipe.stages[:1]
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 3
    stage.transitions = 4
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(
            recipe, 602, tmp_path / "run", store, checkpoint_updates=(1, 2)
        )
    assert run.status == "completed"
    tracked = read_all(tmp_path / "run" / stage.id / "etude")[0]
    assert [c.at["updates"] for c in tracked.checkpoints] == [0, 1, 2, 3]
    assert [c.at["learner_transitions"] for c in tracked.checkpoints] == [
        0,
        *(c.learner_transitions for c in run.monitoring_checkpoints),
        sum(s.learner_transitions for s in run.stages),
    ]
    expected = [
        run.stages[0].artifacts["initial_raw"],
        *(c.artifact for c in run.monitoring_checkpoints),
        run.stages[0].artifacts["raw"],
    ]
    assert all(a is not None for a in expected)
    assert [c.artifact.sha256 for c in tracked.checkpoints] == [
        a["sha256"] for a in expected if a is not None
    ]
    assert tracked.state is not None and tracked.state.state == "finished"
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
