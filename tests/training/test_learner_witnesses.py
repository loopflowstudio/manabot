"""Numerical diagnostics preserve learning and retain the failing boundary."""

from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch

from experiments.runners.inference_preparation import checkpoint_witnesses
from experiments.runners.learner_witnesses import learner_witnesses
from experiments.runners.training_profile import cases
from manabot.model.agent import Agent
from manabot.training import objectives
from manabot.training.execution import execute_regime
from manabot.training.health import NumericalHealth, OptimizerHealth
from manabot.training.models import AtaraxosMoveLearning, TrainSelfPlay
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import recipe


def test_diagnostic_preserves_training_and_repeats_exactly(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    config = recipe()
    stage = config.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.transitions = 8
    stage.learning = AtaraxosMoveLearning(
        gradient="ataraxos_move", min_advantage=0, advantage_quantile=0, ema=0.9
    )
    witnesses = []
    traces = []
    original_forward = Agent.forward_distribution
    original_update = objectives.update_move_iteration
    for repeat in range(3):
        root = tmp_path / str(repeat)
        root.mkdir()
        context = learner_witnesses(root) if repeat else nullcontext()
        with context, VerifyStore(root / "training.sqlite") as store:
            run = execute_regime(config, 10349, root / "run", store)
        assert run.status == "completed"
        witnesses.append(checkpoint_witnesses(root / "run"))
        assert Agent.forward_distribution is original_forward
        assert objectives.update_move_iteration is original_update
        if repeat:
            traces.append((root / "learner-witnesses.jsonl").read_bytes())
    assert witnesses[0] == witnesses[1] == witnesses[2]
    assert traces[0] == traces[1]
    rows = [json.loads(line) for line in traces[0].splitlines()]
    assert {row["kind"] for row in rows} >= {
        "update_before",
        "forward_before",
        "forward_after",
        "targets",
        "backward_before",
        "adam_before",
        "adam_after",
        "update_after",
    }
    assert sum(row["kind"] == "adam_after" for row in rows) > 0


def test_large_movement_is_saved_once_and_patches_restore_on_error(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    health = NumericalHealth(sampled_steps=0, optimizer_steps=0, rejected_steps=0)
    observer = OptimizerHealth(model, optimizer, health, 1)
    original = OptimizerHealth.step
    with pytest.raises(RuntimeError, match="injected"):
        with learner_witnesses(tmp_path):
            for _ in range(2):
                observer.step(model(torch.ones(1, 1)).square().sum(), 1.0)
            raise RuntimeError("injected")
    assert OptimizerHealth.step is original
    saved = torch.load(tmp_path / "large-movement.pt", weights_only=True)
    assert saved["step"] == 1
    assert any(
        not torch.equal(saved["before"][key], value)
        for key, value in saved["after"].items()
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "learner-witnesses.jsonl").read_text().splitlines()
    ]
    assert sum(row["kind"] == "large_movement_saved" for row in rows) == 1


def test_numerical_group_has_repeated_controls_and_no_timing_cells() -> None:
    planned = cases("numerical-64")
    assert len(planned) == 6
    assert all(c.diagnostic for c in planned)
    assert [(c.proof, c.profile, c.updates) for c in planned] == [
        (True, False, 3),
        (True, False, 3),
        (True, True, 3),
        (True, True, 3),
        (False, False, 16),
        (False, False, 16),
    ]
