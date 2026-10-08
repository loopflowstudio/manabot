"""Exact step milestones survive continuation, recovery and evaluator restarts."""

from pathlib import Path

import pytest

from manabot.training import execution
from manabot.training.checkpoint_queue import (
    CheckpointQueue,
    MonitoringBudget,
    checkpoints,
)
from manabot.training.models import CheckpointCadence, TrainingRun, TrainSelfPlay
from manabot.verify.store import VerifyStore
from tests.training import (
    test_checkpoint_queue as queue_fixtures,
    test_learning_state as portable,
)
from tests.training.test_monitoring import _recipe

producer = portable.producer
fake_process = queue_fixtures.fake_process


def test_cadence_defaults_and_legacy_identity() -> None:
    assert CheckpointCadence().model_dump() == {"checkpoint_updates": 1000}
    assert CheckpointCadence(checkpoint_seconds=60).model_dump() == {
        "checkpoint_seconds": 60.0
    }
    with pytest.raises(ValueError, match="not both"):
        CheckpointCadence(checkpoint_updates=10, checkpoint_seconds=60)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_step_interval_is_a_positive_integer(interval: object) -> None:
    with pytest.raises(ValueError):
        CheckpointCadence.model_validate({"checkpoint_updates": interval})


def test_real_iterations_export_exact_milestones_and_one_endpoint(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 6
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        run = execution.execute_regime(
            recipe, 197, tmp_path / "run", store, checkpoint_updates=2
        )
    assert run.status == "completed", run.error
    assert [c.updates for c in run.monitoring_checkpoints] == [2, 4]
    assert [c.coordinates.updates for c in checkpoints(run, include_initial=True)] == [
        0,
        2,
        4,
        6,
    ]
    assert all(
        c.coordinates.learner_transitions == c.coordinates.updates * 32
        for c in checkpoints(run)
    )


def test_continuation_anchors_after_inherited_step(
    tmp_path: Path, producer: TrainingRun
) -> None:
    recipe = portable.continuation(producer, tmp_path / "producer/run.json", 7)
    with VerifyStore(tmp_path / "segments.sqlite") as store:
        run = execution.execute_regime(
            recipe, 197, tmp_path / "segment", store, checkpoint_updates=3
        )
    assert run.status == "completed", run.error
    assert run.updates_through() == 7
    assert len(run.stages[0].diagnostics) == 5
    assert [c.coordinates.updates for c in checkpoints(run, include_initial=True)] == [
        3,
        6,
        7,
    ]
    assert run.stages[0].diagnostics[0]["iteration"] == 3


def test_recovery_preserves_prior_exports_and_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 6
    recipe.recovery_max_microsteps = 10000
    export = execution.export_training_run

    def interrupt(run_id: str, store: VerifyStore, out: str | Path) -> TrainingRun:
        run = export(run_id, store, out)
        if run.status == "running" and run.updates_through() == 3:
            raise KeyboardInterrupt("after committed step 3")
        return run

    with VerifyStore(tmp_path / "runs.sqlite") as store:
        with monkeypatch.context() as patch:
            patch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0)
            patch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))
            patch.setattr(execution, "export_training_run", interrupt)
            with pytest.raises(KeyboardInterrupt):
                execution.execute_regime(
                    recipe, 197, tmp_path / "first", store, checkpoint_updates=2
                )
        parent = TrainingRun.model_validate_json(
            (tmp_path / "first/run.json").read_text()
        )
        original = parent.monitoring_checkpoints[0].model_dump()
        with pytest.raises(ValueError, match="cadence differs"):
            execution.execute_regime(
                recipe,
                197,
                tmp_path / "wrong",
                store,
                resume_from=parent.id,
                checkpoint_updates=3,
            )
        resumed = execution.execute_regime(
            recipe,
            197,
            tmp_path / "resumed",
            store,
            resume_from=parent.id,
            checkpoint_updates=2,
        )
    assert resumed.status == "completed", resumed.error
    assert resumed.monitoring_checkpoints[0].model_dump() == original
    assert [c.coordinates.updates for c in checkpoints(resumed)] == [2, 4, 6]


@pytest.mark.usefixtures("fake_process")
def test_queue_deduplicates_recovered_artifacts_and_restart(tmp_path: Path) -> None:
    source = tmp_path / "first.json"
    run = queue_fixtures.run_fixture(source)
    run.monitoring_checkpoint_updates = 1
    source.write_text(run.model_dump_json())
    recovered = run.model_copy(
        deep=True, update={"id": "recovered", "parent_run_id": run.id}
    )
    second = tmp_path / "recovered.json"
    second.write_text(recovered.model_dump_json())
    out, lease = tmp_path / "monitor", tmp_path / "owner.lock"
    config = MonitoringBudget(seconds=10, attempt_seconds=5)
    queue = CheckpointQueue(out, config, lease=lease)
    try:
        queue.tick([source, second], launch=False)
        assert queue.pending == 1
        queue.tick([source, second])
        assert len(queue.attempts) == 1
    finally:
        queue.close()
    restarted = CheckpointQueue(out, config, lease=lease)
    try:
        restarted.tick([second, source])
        assert restarted.pending == 0
        assert len(restarted.attempts) == 1
    finally:
        restarted.close()


def test_step_admission_rejects_ambiguous_stage_clocks() -> None:
    recipe = _recipe()
    recipe.stages.append(recipe.stages[0].model_copy(update={"id": "second"}))
    with pytest.raises(ValueError, match="one step-target self-play stage"):
        CheckpointCadence().admit_regime(recipe)
    CheckpointCadence(checkpoint_seconds=60).admit_regime(recipe)
