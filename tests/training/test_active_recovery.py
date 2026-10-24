"""Real current-game replay, safe pause and conservative downtime accounting."""

import gzip
from pathlib import Path
import socket

import pytest
import torch

from manabot.arena.monitor import MonitorProtocol
from manabot.experiment.config import Baseline, Experiment
from manabot.experiment.evaluation_queue import CheckpointQueue, MonitoringBudget
from manabot.experiment.execution import ExperimentSchedule, Hardware, HardwareInventory
from manabot.experiment.runner import run_experiment
from manabot.training import execution
from manabot.training.records import TrainingRun
from manabot.training.regime import RecoveryPolicy, TrainSelfPlay
from manabot.training.state import recovery
from manabot.training.state.recovery import TrainingPaused, load_update
from manabot.verify.store import VerifyStore
from tests.training import test_recovery as proof
from tests.training.test_recovery import (
    assert_learning_metrics_equal,
    assert_state_equal,
    recipe,
)


def test_safe_pause_restores_current_games_and_exact_learning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = recipe()
    value.recovery_max_microsteps = None
    value.recovery = RecoveryPolicy(checkpoint_updates=2)
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 12
    pause = tmp_path / "pause.request"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        whole = execution.execute_regime(value, 197, tmp_path / "whole", store)
        export = execution.export_training_run

        def request_pause(
            run_id: str, owner: VerifyStore, out: str | Path
        ) -> TrainingRun:
            run = export(run_id, owner, out)
            if run.status == "running" and run.updates_through() == 2:
                pause.touch()
            return run

        with monkeypatch.context() as patch:
            # The pause request is injected through an exported update boundary.
            patch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0.0)
            patch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))
            patch.setattr(execution, "export_training_run", request_pause)
            # An arbitrarily advanced continuous clock must not spend active time.
            patch.setattr(execution, "watchdog_seconds", lambda: 1e20)
            with pytest.raises(TrainingPaused):
                execution.execute_regime(
                    value, 197, tmp_path / "paused", store, pause_path=pause
                )
        stopped = TrainingRun.model_validate_json(
            (tmp_path / "paused/run.json").read_text()
        )
        assert stopped.status == "interrupted" and stopped.updates_through() == 3
        assert load_update(stopped).iteration == 3
        pause.unlink()
        child = execution.execute_regime(
            value, 197, tmp_path / "resumed", store, resume_from=stopped.id
        )
        assert child.status == "completed" and child.updates_through() == 12
        assert child.prior_seconds == stopped.seconds
        assert child.prior_watchdog_seconds == stopped.watchdog_seconds
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
        assert a.collector.current_games == b.collector.current_games
        assert a.collector.current_games is not None
        assert a.collector.stats.games > 0
        assert (
            sum(map(len, a.collector.current_games.actions))
            < a.collector.stats.micro_steps
        )
        assert a.collector.journal == b.collector.journal == []
        assert_state_equal(a.collector.buffers, b.collector.buffers)
        assert_learning_metrics_equal(whole, child, tmp_path)
        assert (
            child.stages[0].learner_transitions == whole.stages[0].learner_transitions
        )
        assert child.stages[0].games == whole.stages[0].games
        assert child.recovery_artifact is not None
        with gzip.open(child.recovery_artifact["path"], "rb") as stream:
            compact = torch.load(stream, weights_only=False)
        assert compact.format_version == 4
        # Snapshots hold counters, not update rows: those are Etude metrics.
        assert compact.record.updates == 12 and compact.record.diagnostics == []


def test_orphan_separates_known_sleep_from_uncertain_awake_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = recipe()
    value.recovery_max_microsteps = None
    value.recovery = RecoveryPolicy()
    parent = TrainingRun(
        id="stopped",
        regime=value,
        regime_digest="fixture",
        seed=197,
        seed_streams={},
        identities={},
        status="running",
        seconds=10,
        watchdog_seconds=10,
        recovery_host=socket.gethostname(),
        last_recorded_wall_seconds=1000,
        last_recorded_active_seconds=500,
        recovery_boot_identity="same-boot",
    )
    monkeypatch.setattr(recovery.time, "time", lambda: 1107)
    monkeypatch.setattr(recovery.time, "monotonic", lambda: 507)
    monkeypatch.setattr(recovery, "boot_identity", lambda: "same-boot")
    settled = recovery.settle_orphan(parent)
    assert settled.seconds == settled.watchdog_seconds == 17
    assert settled.downtime_seconds == 100
    assert settled.unobserved_seconds == 7
    assert recovery.settle_orphan(settled) == settled
    monkeypatch.setattr(recovery, "boot_identity", lambda: "new-boot")
    rebooted = recovery.settle_orphan(parent)
    assert rebooted.unobserved_seconds == 107
    assert rebooted.seconds == 117


def test_shared_experiment_pauses_and_resumes_without_resetting_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture must not acquire a running experiment's real host lease.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "test-account")
    value = recipe()
    value.recovery_max_microsteps = None
    value.recovery = RecoveryPolicy(checkpoint_updates=1)
    value.wall_seconds = 180
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 12
    stage.execution.wall_seconds = 170
    experiment = Experiment(
        name="pause-proof",
        baseline=Baseline.capture("pause-proof", value),
        schedule=ExperimentSchedule(
            seeds=(197,),
            hardware="fixture",
            wall_seconds=400,
            process_seconds=400,
            active_runtime=True,
            monitoring=MonitoringBudget(
                seconds=200,
                attempt_seconds=75,
                active_runtime=True,
                protocol=MonitorProtocol(deal_seeds=(1911182900,), game_seconds=25),
            ),
            checkpoint_seconds=3600,
        ),
    )
    placement = HardwareInventory(
        resources=(Hardware(name="fixture", host=socket.gethostname(), cpu_threads=2),)
    )
    root = tmp_path / "experiment"
    tick = CheckpointQueue.tick

    def request_pause(
        queue: CheckpointQueue, sources: list[Path], *, launch: bool = True
    ) -> None:
        # The supervisor invokes this after starting the learner. Request at
        # startup rather than waiting for a running export: twelve tiny updates
        # can finish between its five-second polls. The worker must still reach
        # an exact recovery boundary before honoring the request.
        (root / "pause.request").touch()
        tick(queue, sources, launch=launch)

    with monkeypatch.context() as patch:
        patch.setattr(CheckpointQueue, "tick", request_pause)
        paused = run_experiment(experiment, placement, root)
    assert paused.paused and paused.status == "incomplete"
    assert paused.attempts[0].status == "interrupted"
    parent = Path(paused.attempts[0].path) / "run.json"
    before = parent.read_bytes()
    resumed = run_experiment(experiment, placement, root, resume=True, recover=(0,))
    assert resumed.status == "completed" and not resumed.paused
    assert len(resumed.attempts) == 2
    assert parent.read_bytes() == before
    assert resumed.attempts[1].allowance_seconds < paused.attempts[0].allowance_seconds
    assert resumed.process_seconds > paused.process_seconds
    assert resumed.elapsed_seconds > paused.elapsed_seconds
    run = TrainingRun.model_validate_json(
        (Path(resumed.attempts[1].path) / "run.json").read_text()
    )
    assert run.updates_through() == 12


def test_current_game_recovery_after_abrupt_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse the actual os._exit/store-settlement proof with compact native state."""
    value = proof.multistage_recipe()
    value.recovery_max_microsteps = None
    value.recovery = RecoveryPolicy(checkpoint_updates=1)
    monkeypatch.setattr(proof, "multistage_recipe", lambda: value)
    proof.test_abrupt_exit_recovers_without_rewriting_last_export(tmp_path, "update")
