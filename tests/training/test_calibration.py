"""Bounded complete-loop receipts include real native play and failed attempts."""

from pathlib import Path

import pytest

from experiments.runners import calibrate_training
from manabot.training import execution
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore


def test_complete_loop_retains_observed_checkpoints(tmp_path: Path) -> None:
    out = tmp_path / "calibration"
    report = calibrate_training.calibrate(out)
    assert report.status == "completed"
    assert report.complete_replayed_games == report.scheduled_evaluation_games == 8
    assert 0 < report.replay_seconds < report.evaluation_including_replay_seconds
    assert report.elapsed_seconds >= report.evaluation_including_replay_seconds
    assert len(report.runs) == 1
    run = report.runs[0]
    assert all(s.actual_device == "cpu" and s.actual_threads == 1 for s in run.stages)
    assert sum(s.learner_transitions for s in run.stages) == 512
    assert sum(s.environment_decisions for s in run.stages) >= 512
    assert sum(s.games for s in run.stages) > 0
    assert all(s.optimizer_exposures > 0 for s in run.stages)
    costs = [s.cumulative_seconds for s in run.stages]
    assert all(cost is not None for cost in costs)
    assert costs[0] < costs[1]
    for stage in run.stages:
        assert stage.collection_seconds > 0
        assert stage.learning_seconds > 0
        assert stage.export_seconds > 0
        assert stage.sampled_peak_rss_bytes > 0
        assert Path(stage.artifacts["raw"]["path"]).exists()
    saved = (out / "calibration.json").read_bytes()
    with pytest.raises(FileExistsError):
        calibrate_training.calibrate(out)
    assert (out / "calibration.json").read_bytes() == saved


def test_training_failure_remains_in_canonical_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = execution.export_training_run

    def stop_after_setup(
        run_id: str, store: VerifyStore, out: str | Path
    ) -> TrainingRun:
        run = export(run_id, store, out)
        if run.status == "running":
            raise RuntimeError("injected calibration interruption")
        return run

    monkeypatch.setattr(execution, "export_training_run", stop_after_setup)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected calibration"):
        calibrate_training.calibrate(out)
    report = calibrate_training.CalibrationReport.model_validate_json(
        (out / "calibration.json").read_text()
    )
    assert report.status == "failed"
    assert len(report.runs) == 1
    assert report.runs[0].status == "failed"
    assert report.runs[0].run_seconds > 0
    assert report.error is not None
    with VerifyStore(out / "training.sqlite") as store:
        run = store.training_run(report.runs[0].run_id)
        assert run.status == "failed" and run.seed == 197
    assert (out / "study.json").exists()


@pytest.mark.parametrize("device", ["mps", "cuda", "cpu:0"])
def test_unsupported_devices_never_launch(tmp_path: Path, device: str) -> None:
    out = tmp_path / device.replace(":", "-")
    with pytest.raises(ValueError, match="CPU only"):
        calibrate_training.calibrate(out, device)
    assert not out.exists()
