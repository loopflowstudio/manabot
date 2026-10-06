"""Bounded complete-loop receipts include real native play and failed attempts."""

from pathlib import Path

import pytest

from experiments.runners import calibrate_training
from manabot.training import execution
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
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
    with VerifyStore(out / "training.sqlite") as store:
        canonical = store.training_run(run.run_id)
    probe = calibrate_training._inference_probe(canonical)
    assert probe.architecture.identity == canonical.identities["architecture"]
    assert probe.initialization_seconds > 0 and probe.cold_load_seconds > 0
    assert probe.first_forward_seconds > 0 and probe.steady_forward_seconds > 0
    assert probe.sampled_peak_rss_bytes > 0
    assert probe.batch_size == 4 and probe.measured_forwards == 10
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


def test_capacity_plan_reuses_ladder_and_freezes_small_workload() -> None:
    plan = calibrate_training.capacity_plan()
    assert plan.allocation_seconds == 900
    assert plan.protocol.process_seconds == 780
    assert plan.protocol.purpose == "workflow-smoke"
    recipes = [TrainingRegime.model_validate(r) for r in plan.recipes]
    assert [(r.agent.hidden_dim, r.agent.attention_layers) for r in recipes] == [
        (64, 1),
        (64, 2),
        (128, 2),
    ]
    for recipe in recipes:
        assert recipe.agent.num_attention_heads == 4
        assert recipe.agent.semantic_pack == "ur-lessons-vs-gw-allies"
        assert recipe.agent.value_aggregation == "historical_mean"
        for stage in recipe.stages:
            assert isinstance(stage, TrainSelfPlay)
            assert stage.streams * stage.transitions * stage.updates == 256
            assert stage.execution.threads == 1
    with pytest.raises(ValueError, match="CPU only"):
        calibrate_training.capacity_plan("mps")


def test_probe_failure_is_not_hidden_by_completed_study(tmp_path: Path) -> None:
    (tmp_path / "study.json").write_text('{"status":"completed","comparisons":[]}')
    report = calibrate_training.CalibrationReport(
        status="failed",
        error="probe deadline exceeded",
        host_before=calibrate_training.HostSample(
            load_average=(1, 1, 1),
            process_rss_bytes=1,
            available_memory_bytes=1,
            torch_threads=1,
        ),
    )
    calibrate_training._collect_records(tmp_path, report)
    assert report.status == "failed"
    assert report.error == "probe deadline exceeded"
