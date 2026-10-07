"""Step-target pause/export and explicit same-host recovery with real CPU learning."""

from collections.abc import Callable
from pathlib import Path
import time

import pytest

from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training import execution
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorProtocol, stage_checkpoint
from manabot.training.recovery import load_update
from manabot.verify.store import VerifyStore
from tests.remote.test_job_spec import job_spec
from tests.training.test_recovery import (
    assert_learning_diagnostics_equal,
    assert_state_equal,
    recipe,
)


class AllocationClock:
    """Advance wall authority independently of measured process/performance time."""

    def __init__(self) -> None:
        self.now = time.time()

    monotonic = staticmethod(time.monotonic)
    perf_counter = staticmethod(time.perf_counter)
    process_time = staticmethod(time.process_time)
    sleep = staticmethod(time.sleep)

    def time(self) -> float:
        return self.now


def test_pause_export_and_resume_preserve_target_and_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = AllocationClock()
    monkeypatch.setattr(execution, "time", clock)
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0.0)
    monkeypatch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))
    value = recipe()
    # Historical recipe watchdogs must not become independent allocation clocks.
    value.wall_seconds = 0.002
    value.stages[0].execution.wall_seconds = 0.001
    original = value.model_dump_json()
    allocation = job_spec().admit(clock.time())
    export = execution.export_training_run

    def pause_after_update(
        run_id: str, owner: VerifyStore, out: str | Path
    ) -> TrainingRun:
        run = export(run_id, owner, out)
        if run.status == "running" and run.stages and run.stages[0].diagnostics:
            clock.now = allocation.pause_at
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        with monkeypatch.context() as patch:
            patch.setattr(execution, "export_training_run", pause_after_update)
            paused = execution.execute_regime(
                value, 197, tmp_path / "paused", store, allocation=allocation
            )
        assert paused.status == "paused"
        assert paused.stages[0].status == "paused"
        assert paused.updates_through() == 1
        assert set(paused.stages[0].artifacts) == {"raw", "ema", "optimizer"}
        assert paused.selected_artifact is None
        assert paused.allocation_deadline == allocation.deadline
        assert paused.recovery_artifact is not None
        second = job_spec(1).admit(clock.time())
        resumed = execution.execute_regime(
            value,
            197,
            tmp_path / "resumed",
            store,
            allocation=second,
            resume_from=paused.id,
        )
        control = execution.execute_regime(
            value, 197, tmp_path / "control", store, allocation=second
        )
        assert resumed.status == control.status == "completed"
        assert resumed.updates_through() == control.updates_through() == 3
        assert resumed.regime_digest == paused.regime_digest == control.regime_digest
        assert_learning_diagnostics_equal(resumed.stages[0], control.stages[0])
        left, right = load_update(resumed), load_update(control)
        assert_state_equal(left.learner, right.learner)
        assert_state_equal(left.optimizer, right.optimizer)
        assert_state_equal(left.ema, right.ema)
        assert value.model_dump_json() == original


def test_reserves_exhausted_refuses_before_training(tmp_path: Path) -> None:
    allocation = job_spec().admit(time.time() - 1800)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(ValueError, match="reserves exhausted"):
            execution.execute_regime(
                recipe(), 197, tmp_path / "never-started", store, allocation=allocation
            )
    assert not (tmp_path / "never-started").exists()


def test_collection_cutoff_leaves_checkpoint_reserve_without_claiming_cuda_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    clock = AllocationClock()
    monkeypatch.setattr(execution, "time", clock)
    allocation = job_spec().admit(clock.time())
    original_collect = SeatRoutedCollector.collect
    value = recipe()
    value.recovery_max_microsteps = None

    def expire_collection(
        collector: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        assert deadline_monotonic is not None
        remaining = deadline_monotonic - time.perf_counter()
        assert remaining == pytest.approx(allocation.pause_at - clock.time(), abs=0.1)
        clock.now = allocation.pause_at
        return original_collect(
            collector,
            agent,
            num_steps,
            deadline_monotonic=time.perf_counter() - 1,
            check=check,
        )

    monkeypatch.setattr(SeatRoutedCollector, "collect", expire_collection)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        paused = execution.execute_regime(
            value, 197, tmp_path / "paused", store, allocation=allocation
        )
        assert paused.status == "paused" and paused.updates_through() == 0
        assert set(paused.stages[0].artifacts) == {"raw", "ema", "optimizer"}
        assert paused.recovery_artifact is None
        point = stage_checkpoint(paused, paused.stages[0].id)
        assert point is not None
        budget = MonitoringBudget(
            seconds=100,
            attempt_seconds=10,
            terminal_protocols=(MonitorProtocol(deal_seeds=(1,)),),
        )
        assert budget.protocols_for(paused, point) == [budget.protocol]
        with pytest.raises(ValueError, match="no committed recovery"):
            execution.execute_regime(
                value,
                197,
                tmp_path / "forbidden",
                store,
                resume_from=paused.id,
                allocation=job_spec().admit(clock.time()),
            )


def test_export_reserve_exhaustion_is_not_a_completed_or_paused_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = AllocationClock()
    monkeypatch.setattr(execution, "time", clock)
    allocation = job_spec().admit(clock.time())
    value = recipe()
    value.stages[0].updates = 1
    value.recovery_max_microsteps = None
    load = execution.load_checkpoint_agent

    def slow_admission(path: str) -> tuple[Agent, ObservationSpace]:
        result = load(path)
        clock.now = allocation.checkpoint_deadline
        return result

    monkeypatch.setattr(execution, "load_checkpoint_agent", slow_admission)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(TimeoutError, match="wall deadline"):
            execution.execute_regime(
                value, 197, tmp_path / "exhausted", store, allocation=allocation
            )
        row = store.con.execute("SELECT id FROM training_runs").fetchone()
        retained = store.training_run(row[0])
        assert retained.status == "interrupted"
        assert retained.stages[0].status == "interrupted"
        assert retained.selected_artifact is None
        assert "raw" in retained.stages[0].artifacts
