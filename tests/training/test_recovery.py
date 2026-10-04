"""Real native games cross interruption boundaries without changing learning."""

from pathlib import Path
import subprocess
import sys
from typing import Any, Literal

import numpy as np
import pytest
import torch

from manabot.training import execution
from manabot.training.models import (
    AtaraxosMoveLearning,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.recovery import attempt_lock, load_update
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]


def recipe() -> TrainingRegime:
    value = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text()
    )
    value.stages = value.stages[:1]
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 3
    stage.transitions = 32
    stage.streams = 2
    stage.learning.epochs = 1
    stage.learning.ema = 0.9
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    value.recovery_max_microsteps = 10000
    value.schedule_clock = "iteration_fraction"
    return value


def assert_state_equal(left: Any, right: Any) -> None:
    """Narrow recursive torch/NumPy serialization values at the test boundary."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def interrupt(
    value: TrainingRegime,
    out: Path,
    store: VerifyStore,
    monkeypatch: pytest.MonkeyPatch,
) -> TrainingRun:
    export = execution.export_training_run

    def fail_after_commit(
        run_id: str, owner: VerifyStore, path: str | Path
    ) -> TrainingRun:
        run = export(run_id, owner, path)
        if run.status == "running" and run.recovery_artifact is not None:
            raise KeyboardInterrupt("injected after durable update")
        return run

    with monkeypatch.context() as patch:
        patch.setattr(execution, "export_training_run", fail_after_commit)
        with pytest.raises(KeyboardInterrupt):
            execution.execute_regime(value, 197, out, store)
    row = store.con.execute(
        "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    return store.training_run(row[0])


@pytest.mark.parametrize("gradient", ["ppo", "ataraxos_move"])
def test_real_interruption_restores_complete_learning_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gradient: Literal["ppo", "ataraxos_move"],
) -> None:
    value = recipe()
    if gradient == "ataraxos_move":
        stage = value.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        value.agent.value_kind = "categorical_wdl"
        stage.learning = AtaraxosMoveLearning(
            gradient="ataraxos_move", ema=0.9, min_advantage=0, advantage_quantile=0
        )
    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execution.execute_regime(value, 197, tmp_path / "whole", store)
        failed = interrupt(value, tmp_path / "failed", store, monkeypatch)
        frozen = (tmp_path / "failed/run.json").read_bytes()
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=failed.id
        )
        assert store.training_run(failed.id) == failed
        assert (tmp_path / "failed/run.json").read_bytes() == frozen
        assert resumed.prior_seconds == failed.seconds
        assert resumed.prior_watchdog_seconds == failed.watchdog_seconds
        assert resumed.recovery_seconds > 0
        assert resumed.stages[0].cumulative_seconds >= resumed.prior_seconds
        a, b = load_update(whole), load_update(resumed)
        for name in (
            "learner",
            "optimizer",
            "ema",
            "minibatch_rng",
            "python_rng",
            "numpy_rng",
            "torch_rng",
        ):
            assert_state_equal(getattr(a, name), getattr(b, name))
        assert a.collector.journal == b.collector.journal
        assert_state_equal(a.collector.buffers, b.collector.buffers)
        assert_state_equal(a.collector.sampling_rng, b.collector.sampling_rng)
        assert a.iteration == b.iteration == 3
        assert whole.stages[0].diagnostics == resumed.stages[0].diagnostics
        assert whole.stages[0].games == resumed.stages[0].games
        assert whole.stages[0].games > 0
        assert whole.stages[0].learner_transitions == 192
        assert whole.stages[0].optimizer_exposures > 0
        with pytest.raises(ValueError, match="already has a recovery child"):
            execution.execute_regime(
                value, 197, tmp_path / "fork", store, resume_from=failed.id
            )


def test_recovery_rejects_seed_and_corrupt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = recipe()
    with VerifyStore(tmp_path / "training.sqlite") as store:
        failed = interrupt(value, tmp_path / "failed", store, monkeypatch)
        with pytest.raises(ValueError, match="recipe or seed"):
            execution.execute_regime(
                value, 198, tmp_path / "wrong-seed", store, resume_from=failed.id
            )
        assert failed.recovery_artifact is not None
        Path(failed.recovery_artifact["path"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="digest mismatch"):
            execution.execute_regime(
                value, 197, tmp_path / "corrupt", store, resume_from=failed.id
            )


def test_sleep_inclusive_watchdog_retains_cost_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = recipe()
    with VerifyStore(tmp_path / "training.sqlite") as store:
        failed = interrupt(value, tmp_path / "failed", store, monkeypatch)
        clock = execution.watchdog_seconds
        calls = 0

        def suspended_clock() -> float:
            nonlocal calls
            calls += 1
            return clock() + (10000 if calls > 1 else 0)

        monkeypatch.setattr(execution, "watchdog_seconds", suspended_clock)
        with pytest.raises(TimeoutError):
            execution.execute_regime(
                value, 197, tmp_path / "sleep", store, resume_from=failed.id
            )
        row = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        stopped = store.training_run(row[0])
        assert stopped.watchdog_seconds >= 10000
        assert stopped.prior_watchdog_seconds == failed.watchdog_seconds
        assert stopped.recovery_artifact == failed.recovery_artifact


def test_recovery_rejects_partial_support() -> None:
    value = recipe()
    value.schedule_clock = "run_elapsed_budget"
    with pytest.raises(ValueError, match="iteration_fraction"):
        execution.validate_regime(value)


def test_abrupt_exit_recovers_without_rewriting_last_export(tmp_path: Path) -> None:
    """os._exit bypasses exception handlers and models a lost training process."""
    value = recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(value.model_dump_json())
    database = tmp_path / "training.sqlite"
    failed_out = tmp_path / "abrupt"
    script = """
import os
import sys
from pathlib import Path
from manabot.training import execution
from manabot.training.models import TrainingRegime
from manabot.verify.store import VerifyStore
export = execution.export_training_run
def terminate_after_commit(run_id, store, out):
    run = export(run_id, store, out)
    if run.status == 'running' and run.recovery_artifact is not None:
        os._exit(86)
    return run
execution.export_training_run = terminate_after_commit
recipe = TrainingRegime.model_validate_json(Path(sys.argv[1]).read_text())
with VerifyStore(Path(sys.argv[2])) as store:
    execution.execute_regime(recipe, 197, Path(sys.argv[3]), store)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(recipe_path),
            str(database),
            str(failed_out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 86, result.stderr
    evidence = (failed_out / "run.json").read_bytes()
    with VerifyStore(database) as store:
        parent_id = store.con.execute("SELECT id FROM training_runs").fetchone()[0]
        assert store.training_run(parent_id).status == "running"
        resumed = execution.execute_regime(
            value, 197, tmp_path / "continued", store, resume_from=parent_id
        )
        parent = store.training_run(parent_id)
        assert parent.status == "interrupted"
        assert parent.unobserved_seconds > 0
        assert resumed.prior_seconds == parent.seconds
        assert (failed_out / "run.json").read_bytes() == evidence
        whole = execution.execute_regime(value, 197, tmp_path / "whole", store)
        assert_state_equal(load_update(whole).learner, load_update(resumed).learner)
        assert_state_equal(load_update(whole).optimizer, load_update(resumed).optimizer)


def test_live_lease_rejects_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = recipe()
    with VerifyStore(tmp_path / "training.sqlite") as store:
        parent = interrupt(value, tmp_path / "parent", store, monkeypatch)
        assert parent.recovery_lock_path is not None
        with attempt_lock(Path(parent.recovery_lock_path), existing=True):
            with pytest.raises(ValueError, match="live writer"):
                execution.execute_regime(
                    value, 197, tmp_path / "child", store, resume_from=parent.id
                )
