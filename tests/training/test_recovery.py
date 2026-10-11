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
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.recovery import attempt_lock, load_update
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]


def assert_learning_diagnostics_equal(left: StageRecord, right: StageRecord) -> None:
    """Recovery preserves learning facts, while elapsed cost and host samples vary."""
    for a, b in zip(left.diagnostics, right.diagnostics, strict=True):
        assert {k: v for k, v in a.items() if k != "coordinates"} == {
            k: v for k, v in b.items() if k != "coordinates"
        }
        for name in (
            "updates",
            "environment_decisions",
            "learner_transitions",
            "optimizer_exposures",
            "games",
        ):
            assert a["coordinates"][name] == b["coordinates"][name]


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


@pytest.fixture(autouse=True)
def export_after_every_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests inject failures through the export, so it must follow each save."""
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0.0)
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))


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
        if (
            run.status == "running"
            and run.recovery_artifact is not None
            and run.stages[-1].diagnostics
        ):
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
        assert_learning_diagnostics_equal(whole.stages[0], resumed.stages[0])
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


@pytest.mark.parametrize("boundary", ["update", "completed"])
def test_abrupt_exit_recovers_without_rewriting_last_export(
    tmp_path: Path, boundary: str
) -> None:
    """os._exit bypasses exception handlers and models a lost training process."""
    value = multistage_recipe()
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
execution.PROGRESS_EXPORT_SECONDS = 0.0
execution.PROGRESS_EXPORT_SHARE = float('inf')
export = execution.export_training_run
def terminate_after_commit(run_id, store, out):
    run = export(run_id, store, out)
    if run.status == 'running' and run.recovery_artifact is not None:
        name = Path(run.recovery_artifact['path']).name
        if name.startswith('second-' + sys.argv[4]):
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
            boundary,
        ],
        capture_output=True,
        text=True,
        timeout=120,
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


def multistage_recipe(*, fresh: bool = False) -> TrainingRegime:
    value = recipe()
    first = value.stages[0]
    assert isinstance(first, TrainSelfPlay)
    first.updates = 2
    second = first.model_copy(deep=True)
    second.id = "second"
    second.initial = None if fresh else first.id
    value.stages.append(second)
    return value


@pytest.mark.parametrize(
    "boundary", ["start", "before_complete", "completed", "second_update", "terminal"]
)
@pytest.mark.parametrize("fresh", [False, True])
def test_multistage_boundaries_preserve_state_and_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, fresh: bool
) -> None:
    value = multistage_recipe(fresh=fresh)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execution.execute_regime(value, 197, tmp_path / "whole", store)
        export = execution.export_training_run
        save = execution.save_update
        fired = False

        def fail_at_boundary(
            run_id: str, owner: VerifyStore, path: str | Path
        ) -> TrainingRun:
            nonlocal fired
            run = export(run_id, owner, path)
            if fired or run.status != "running" or run.recovery_artifact is None:
                return run
            name = Path(run.recovery_artifact["path"]).name
            first = value.stages[0].id
            matches = {
                "start": name == f"{first}-start.pt",
                "completed": name == f"{first}-completed.pt",
                "second_update": name.startswith("second-update-"),
                "terminal": name == "second-completed.pt",
            }
            if matches.get(boundary, False):
                fired = True
                raise KeyboardInterrupt("boundary injection")
            return run

        def fail_before_completion(*args: Any, **kwargs: Any) -> None:
            nonlocal fired
            if (
                boundary == "before_complete"
                and not fired
                and str(args[0]).endswith("-completed.pt")
            ):
                fired = True
                raise KeyboardInterrupt("before completion publication")
            save(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(execution, "export_training_run", fail_at_boundary)
            patch.setattr(execution, "save_update", fail_before_completion)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(value, 197, tmp_path / "failed", store)
        parent_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        parent = store.training_run(parent_id)
        completed = [item for item in parent.stages if item.status == "completed"]
        frozen = {
            item["path"]: Path(item["path"]).read_bytes()
            for row in completed
            for item in row.artifacts.values()
        }
        child = execution.execute_regime(
            value, 197, tmp_path / "child", store, resume_from=parent.id
        )
        assert child.status == "completed"
        assert store.training_run(parent.id) == parent
        assert child.prior_seconds == parent.prior_seconds + parent.seconds
        assert child.stages[: len(completed)] == completed
        for path, data in frozen.items():
            assert Path(path).read_bytes() == data
        for row in completed:
            assert not list((tmp_path / "child").glob(f"{row.id}-*.pt"))
        a, b = load_update(whole), load_update(child)
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
        assert a.iteration == b.iteration
        assert a.collector.journal == b.collector.journal
        assert_state_equal(a.collector.buffers, b.collector.buffers)
        for left, right in zip(whole.stages, child.stages, strict=True):
            assert_learning_diagnostics_equal(left, right)
            assert left.learner_transitions == right.learner_transitions
            assert left.environment_decisions == right.environment_decisions
            assert left.games == right.games
        with pytest.raises(ValueError, match="stopped failed, interrupted or paused"):
            execution.execute_regime(
                value, 197, tmp_path / "terminal", store, resume_from=child.id
            )


def test_setup_failure_retains_stage_lineage_and_admission_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = multistage_recipe()
    with VerifyStore(tmp_path / "training.sqlite") as store:
        parent = interrupt(value, tmp_path / "parent", store, monkeypatch)
        fingerprints = execution._runtime_identities
        with monkeypatch.context() as patch:
            patch.setattr(
                execution, "_runtime_identities", lambda *args: {"incompatible": True}
            )
            with pytest.raises(ValueError, match="runtime/source"):
                execution.execute_regime(
                    value, 197, tmp_path / "incompatible", store, resume_from=parent.id
                )
        assert execution._runtime_identities is fingerprints
        export = execution.export_training_run
        fired = False

        def fail_setup(
            run_id: str, owner: VerifyStore, path: str | Path
        ) -> TrainingRun:
            nonlocal fired
            run = export(run_id, owner, path)
            if not fired and run.status == "running":
                fired = True
                raise KeyboardInterrupt("setup failure")
            return run

        with monkeypatch.context() as patch:
            patch.setattr(execution, "export_training_run", fail_setup)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(
                    value, 197, tmp_path / "setup", store, resume_from=parent.id
                )
        child_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        child = store.training_run(child_id)
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=child.id
        )
        assert resumed.status == "completed"
        assert resumed.prior_seconds == parent.seconds + child.seconds
        assert resumed.stages[0].watchdog_seconds >= parent.stages[0].watchdog_seconds
        assert store.training_run(parent.id) == parent
        assert store.training_run(child.id) == child


def test_stage_budgets_do_not_charge_previous_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = multistage_recipe()
    for stage in value.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 1
        stage.execution.wall_seconds = 100
    clock = 0.0
    update = execution.update_iteration

    def timed_update(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal clock
        result = update(*args, **kwargs)
        clock += 60
        return result

    monkeypatch.setattr(execution, "watchdog_seconds", lambda: clock)
    monkeypatch.setattr(execution, "update_iteration", timed_update)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execution.execute_regime(value, 197, tmp_path / "run", store)
        assert run.status == "completed"
        assert run.watchdog_seconds == 120
        assert [stage.watchdog_seconds for stage in run.stages] == [60, 60]


def test_completed_artifact_corruption_rejects_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = multistage_recipe()
    export = execution.export_training_run
    fired = False

    def fail_completed(
        run_id: str, owner: VerifyStore, path: str | Path
    ) -> TrainingRun:
        nonlocal fired
        run = export(run_id, owner, path)
        if (
            not fired
            and run.status == "running"
            and run.stages
            and run.stages[-1].status == "completed"
        ):
            fired = True
            raise KeyboardInterrupt("completed boundary")
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        with monkeypatch.context() as patch:
            patch.setattr(execution, "export_training_run", fail_completed)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(value, 197, tmp_path / "parent", store)
        parent_id = store.con.execute("SELECT id FROM training_runs").fetchone()[0]
        parent = store.training_run(parent_id)
        path = Path(parent.stages[0].artifacts["raw"]["path"])
        original = path.read_bytes()
        path.write_bytes(b"stale bytes")
        with pytest.raises(ValueError, match="stage artifact digest"):
            execution.execute_regime(
                value, 197, tmp_path / "rejected", store, resume_from=parent.id
            )
        path.write_bytes(original)
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=parent.id
        )
        assert resumed.status == "completed"
        assert resumed.stages[0] == parent.stages[0]


def test_death_immediately_after_recovery_claim_keeps_snapshot_admissible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = recipe()
    with VerifyStore(tmp_path / "training.sqlite") as store:
        parent = interrupt(value, tmp_path / "parent", store, monkeypatch)
        claim = store.claim_training_recovery

        def die_after_claim(parent_id: str, run: TrainingRun) -> None:
            claim(parent_id, run)
            raise KeyboardInterrupt("death before setup handler")

        with monkeypatch.context() as patch:
            patch.setattr(store, "claim_training_recovery", die_after_claim)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(
                    value, 197, tmp_path / "claimed", store, resume_from=parent.id
                )
        child_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        assert store.training_run(child_id).status == "running"
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=child_id
        )
        child = store.training_run(child_id)
        assert child.status == "interrupted"
        assert child.unobserved_seconds > 0
        assert resumed.prior_seconds == parent.seconds + child.seconds
        assert resumed.stages[0].watchdog_seconds >= parent.stages[0].watchdog_seconds
        assert resumed.status == "completed"


def test_sparse_recovery_replays_unsaved_updates_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = recipe()
    value.recovery_every_updates = 3
    export = execution.export_training_run

    def fail_after_second_update(
        run_id: str, owner: VerifyStore, path: str | Path
    ) -> TrainingRun:
        run = export(run_id, owner, path)
        if (
            run.status == "running"
            and run.stages
            and len(run.stages[-1].diagnostics) == 2
        ):
            raise KeyboardInterrupt("unsnapshotted progress")
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execution.execute_regime(value, 197, tmp_path / "whole", store)
        with monkeypatch.context() as patch:
            patch.setattr(execution, "export_training_run", fail_after_second_update)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(value, 197, tmp_path / "failed", store)
        row = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        failed = store.training_run(row[0])
        assert len(failed.stages[0].diagnostics) == 2
        assert load_update(failed).iteration == 1
        resumed = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=failed.id
        )
        assert resumed.status == "completed"
        assert resumed.prior_seconds == failed.seconds
        a, b = load_update(whole), load_update(resumed)
        for name in ("learner", "optimizer", "ema", "torch_rng", "minibatch_rng"):
            assert_state_equal(getattr(a, name), getattr(b, name))
        assert a.collector.journal == b.collector.journal
        assert_learning_diagnostics_equal(whole.stages[0], resumed.stages[0])
        assert not list((tmp_path / "whole").glob("*update-00000002.pt"))
