"""Fake-clock queue contracts: no training or games."""

import json
from pathlib import Path

import pytest

from manabot.arena.models import canonical_sha256
from manabot.training import checkpoint_queue as queue
from manabot.training.execution import atomic_json
from manabot.training.models import MonitoringCheckpoint, StageRecord, TrainingRun
from manabot.training.monitor_evaluation import ArenaRow, _summarize
from manabot.training.presets import ataraxos_mtg_v1
from tests.training.test_monitor_evaluation import _manifest, _rows


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class Process:
    pid = 99999999
    code: int | None = None

    def poll(self) -> int | None:
        return self.code

    def wait(self) -> int:
        return self.code if self.code is not None else -9


@pytest.fixture
def fake_process(monkeypatch: pytest.MonkeyPatch) -> list[Process]:
    processes: list[Process] = []

    def start(*args: object, **kwargs: object) -> Process:
        process = Process()
        processes.append(process)
        return process

    def kill(pid: int, sig: int) -> None:
        assert pid == 99999999

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    monkeypatch.setattr(queue.os, "killpg", kill)
    return processes


def run_fixture(path: Path) -> TrainingRun:
    regime = ataraxos_mtg_v1().regime()
    run = TrainingRun(
        id="fixture",
        regime=regime,
        seed=1,
        seed_streams={},
        identities={},
        regime_digest=canonical_sha256(regime.model_dump(mode="json")),
        status="running",
        stages=[
            StageRecord(
                id=regime.stages[0].id,
                status="completed",
                cumulative_seconds=1,
                diagnostics=[{}],
                artifacts={"raw": {"path": "first.pt", "sha256": "a" * 64, "bytes": 1}},
            ),
            StageRecord(id=regime.stages[1].id),
        ],
    )
    atomic_json(path, run.model_dump(mode="json"))
    return run


def test_prompt_discovery_failure_budget_restart_and_duplicates(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run = run_fixture(source)
    clock = Clock()
    config = queue.MonitoringBudget(seconds=10, attempt_seconds=5)
    out, lease = tmp_path / "monitor", tmp_path / "machine.lock"
    monitor = queue.CheckpointQueue(out, config, clock=clock, lease=lease)
    monitor.tick([source, source])
    assert len(fake_process) == 1  # Stage checkpoint while learner is still running.
    assert monitor.active is not None
    clock.now = 2
    fake_process[0].code = 1
    monitor.tick([source])
    assert monitor.attempts[0].status == "failed"
    assert monitor.charged_seconds == 2
    run.status = "failed"
    run.monitoring_checkpoints.append(
        MonitoringCheckpoint(
            ordinal=0,
            stage_id=run.stages[1].id,
            updates=2,
            training_seconds=2,
            artifact={"path": "second.pt", "sha256": "b" * 64, "bytes": 1},
        )
    )
    atomic_json(source, run.model_dump(mode="json"))
    monitor.tick([source])
    assert len(fake_process) == 2  # Completed export survives learner failure.
    clock.now = 7
    monitor.tick([source])
    assert monitor.attempts[1].error == "monitoring allowance exhausted"
    assert monitor.charged_seconds == 7
    monitor.close()
    restarted = queue.CheckpointQueue(out, config, clock=clock, lease=lease)
    restarted.tick([source])
    assert len(fake_process) == 2  # Neither failure is retried.
    assert restarted.charged_seconds == 7
    run.monitoring_checkpoints.append(
        MonitoringCheckpoint(
            ordinal=1,
            stage_id=run.stages[1].id,
            updates=3,
            training_seconds=3,
            artifact={"path": "third.pt", "sha256": "c" * 64, "bytes": 1},
        )
    )
    atomic_json(source, run.model_dump(mode="json"))
    restarted.tick([source])
    assert (
        restarted.pending == 1 and restarted.process is None
    )  # Cannot fit full allowance.
    restarted.close()


def test_orphan_charge_lease_and_immutable_binding(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run = run_fixture(source)
    out, lease = tmp_path / "monitor", tmp_path / "machine.lock"
    config = queue.MonitoringBudget(seconds=20, attempt_seconds=5)
    monitor = queue.CheckpointQueue(out, config, lease=lease)
    monitor.tick([source])
    with pytest.raises(BlockingIOError):
        queue.CheckpointQueue(tmp_path / "other", config, lease=lease)
    monitor.lock.close()  # Simulate dead owner and exited worker; durable status is running.
    restarted = queue.CheckpointQueue(out, config, lease=lease)
    assert restarted.attempts[0].status == "interrupted"
    assert restarted.charged_seconds == 5
    run.stages[0].artifacts["raw"]["sha256"] = "b" * 64
    atomic_json(source, run.model_dump(mode="json"))
    with pytest.raises(ValueError, match="artifact or coordinates changed"):
        restarted.tick([source])
    restarted.close()
    with pytest.raises(ValueError, match="allocation/protocol changed"):
        queue.CheckpointQueue(
            out, queue.MonitoringBudget(seconds=21, attempt_seconds=5), lease=lease
        )


def test_completed_result_projects_and_interruption_suppresses_rate(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run_fixture(source)
    config = queue.MonitoringBudget(seconds=10, attempt_seconds=5)
    clock = Clock()
    monitor = queue.CheckpointQueue(
        tmp_path / "monitor", config, clock=clock, lease=tmp_path / "lock"
    )
    monitor.tick([source])
    attempt = monitor.attempts[0]
    manifest = _manifest()
    manifest.rows = [ArenaRow.model_validate(r) for r in _rows(manifest)]
    _summarize(manifest)
    directory = monitor._directory(attempt) / "evaluation"
    directory.mkdir()
    atomic_json(directory / "monitor.json", manifest.model_dump(mode="json"))
    monitor.tick([source], launch=False)
    dashboard = json.loads(
        next((tmp_path / "monitor").glob("run-*/dashboard.json")).read_text()
    )
    assert dashboard["rows"][0]["monitor/score/mean"] == 0.5
    monitor.close()
    dashboard = json.loads(
        next((tmp_path / "monitor").glob("run-*/dashboard.json")).read_text()
    )
    assert dashboard["rows"][0]["availability/monitor_rates"] is False
    assert "monitor/score/mean" not in dashboard["rows"][0]


def test_relocated_checkpoint_keeps_producer_binding(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run = run_fixture(source)
    run.stages[0].artifacts["initial_raw"] = {
        "path": "initial.pt",
        "sha256": "c" * 64,
        "bytes": 1,
    }
    atomic_json(source, run.model_dump(mode="json"))

    original = source.read_bytes()

    def relocate(reference: queue.ArtifactReference) -> queue.ArtifactReference:
        return {**reference, "path": str(tmp_path / reference["sha256"])}

    monitor = queue.CheckpointQueue(
        tmp_path / "monitor",
        queue.MonitoringBudget(seconds=10, attempt_seconds=5, include_initial=True),
        lease=tmp_path / "lease",
        resolve_artifact=relocate,
    )
    try:
        monitor.tick([source])
        job_path = next((tmp_path / "monitor").glob("attempt-*/job.json"))
        job = queue.EvaluationJob.model_validate_json(job_path.read_text())
        assert job.checkpoint.coordinates.updates == 0
        assert job.checkpoint.artifact["path"] == str(tmp_path / ("c" * 64))
        assert job.run.stages[0].artifacts["initial_raw"]["path"] == "initial.pt"
        assert source.read_bytes() == original
    finally:
        monitor.close()


def test_completed_worker_survives_delayed_parent_poll(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run_fixture(source)
    clock = Clock()
    monitor = queue.CheckpointQueue(
        tmp_path / "monitor",
        queue.MonitoringBudget(seconds=10, attempt_seconds=5),
        clock=clock,
        lease=tmp_path / "lease",
    )
    try:
        monitor.tick([source])
        result = _manifest()
        result.rows = [ArenaRow.model_validate(r) for r in _rows(result)]
        _summarize(result)
        directory = monitor._directory(monitor.attempts[0]) / "evaluation"
        directory.mkdir()
        atomic_json(directory / "monitor.json", result.model_dump(mode="json"))
        fake_process[0].code = 0
        clock.now = 8
        monitor.tick([source], launch=False)
        assert monitor.attempts[0].status == "completed"
        # Parent observation lag remains conservatively charged, never hidden.
        assert monitor.charged_seconds == 8
    finally:
        monitor.close()
