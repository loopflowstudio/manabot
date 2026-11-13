"""Active-time endpoints count completed learner work, excluding slow exports."""

from collections.abc import Callable
from pathlib import Path
import time
from typing import ParamSpec, TypeVar

import pytest

from experiments.ataraxos_move import ataraxos_move_scalar
from manabot.experiment.checkpoint_evaluation import stage_checkpoint
from manabot.training import execution
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
from tests.training.test_ataraxos_regime import _recipe

P = ParamSpec("P")
R = TypeVar("R")


class Clock:
    def __init__(self) -> None:
        # Keep native collection deadlines on the real epoch, but charge only
        # injected durations so host speed cannot change the expected update count.
        self.epoch = time.perf_counter()
        self.offset = 0.0

    def perf_counter(self) -> float:
        return self.epoch + self.offset

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)

    def charge(self, call: Callable[P, R], seconds: float) -> Callable[P, R]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            result = call(*args, **kwargs)
            self.offset += seconds
            return result

        return wrapped


def recipe(updates: int = 10) -> TrainingRegime:
    result = _recipe(ataraxos_move_scalar)
    result.stages = result.stages[:1]
    result.agent.hidden_dim = 8
    stage = result.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.transitions = 4
    stage.updates = updates
    stage.active_seconds = 15
    stage.execution.wall_seconds = 900
    result.wall_seconds = 1000
    return TrainingRegime.model_validate(result.model_dump())


@pytest.mark.parametrize("updates,expected", [(10, "completed"), (1, "failed")])
def test_endpoint_excludes_exports_and_rejects_early_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates: int, expected: str
) -> None:
    clock = Clock()
    monkeypatch.setattr(execution, "time", clock)
    monkeypatch.setattr(execution, "collect", clock.charge(execution.collect, 2))
    monkeypatch.setattr(
        execution,
        "update_damped_iteration",
        clock.charge(execution.update_damped_iteration, 10),
    )
    monkeypatch.setattr(
        execution, "save_checkpoint", clock.charge(execution.save_checkpoint, 100)
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        if expected == "failed":
            with pytest.raises(RuntimeError, match="safety ceiling"):
                execution.execute_regime(
                    recipe(updates),
                    10351,
                    tmp_path / "run",
                    store,
                    checkpoint_seconds=5,
                )
            run = TrainingRun.model_validate_json(
                (tmp_path / "run/run.json").read_text()
            )
        else:
            run = execution.execute_regime(
                recipe(updates), 10351, tmp_path / "run", store, checkpoint_seconds=5
            )
        assert run == store.training_run(run.id)
    assert run.status == expected
    stage = run.stages[0]
    if expected == "completed":
        assert stage.updates == 2
        assert stage.collection_seconds + stage.learning_seconds >= 20
        assert len(run.monitoring_checkpoints) == 1
        point = run.monitoring_checkpoints[0]
        assert point.active_training_seconds is not None
        assert 10 <= point.active_training_seconds < 20
        assert point.training_seconds > 100
        terminal = stage_checkpoint(run, stage.id)
        assert terminal is not None
        assert terminal.coordinates.active_training_seconds is not None
        assert terminal.coordinates.active_training_seconds >= 20
    else:
        assert "safety ceiling" in (stage.error or "")
        assert "raw" not in stage.artifacts
        assert stage.collection_seconds + stage.learning_seconds < 15


def test_old_recipe_bytes_and_invalid_endpoint() -> None:
    old = _recipe(ataraxos_move_scalar)
    assert all("active_seconds" not in stage for stage in old.model_dump()["stages"])
    payload = recipe().model_dump()
    payload["stages"][0]["active_seconds"] = 900
    with pytest.raises(ValueError, match="watchdog reserve"):
        TrainingRegime.model_validate(payload)
    payload = recipe().model_dump()
    payload["recovery_max_microsteps"] = 10000
    payload["schedule_clock"] = "iteration_fraction"
    with pytest.raises(ValueError, match="active-time process recovery"):
        TrainingRegime.model_validate(payload)


@pytest.mark.parametrize("admitted", [True, False])
def test_initial_admission_precedes_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admitted: bool
) -> None:
    clock = Clock()
    gate = tmp_path / "initial-admission"
    out = tmp_path / "run"

    def admission_sleep(seconds: float) -> None:
        # Admission waits only after publishing the initial checkpoint. Release
        # it here so the test needs neither a supervisor thread nor real sleeps.
        run = TrainingRun.model_validate_json((out / "run.json").read_text())
        assert run.updates_through() == 0
        assert run.stages[0].learner_transitions == 0
        # A slow admission cannot consume the active training allocation.
        clock.offset += 100
        digest = run.stages[0].artifacts["initial_raw"]["sha256"]
        gate.write_text(digest if admitted else "wrong-checkpoint")

    monkeypatch.setattr(clock, "sleep", admission_sleep)
    monkeypatch.setattr(execution, "time", clock)
    monkeypatch.setattr(execution, "collect", clock.charge(execution.collect, 2))
    monkeypatch.setattr(
        execution,
        "update_damped_iteration",
        clock.charge(execution.update_damped_iteration, 10),
    )

    def train() -> TrainingRun:
        with VerifyStore(tmp_path / "training.sqlite") as store:
            return execution.execute_regime(
                recipe(),
                10351,
                out,
                store,
                checkpoint_seconds=5,
                initial_admission=gate,
            )

    if admitted:
        run = train()
        assert run.status == "completed"
        assert run.stages[0].updates == 2
    else:
        with pytest.raises(ValueError, match="admission differs"):
            train()
        run = TrainingRun.model_validate_json((out / "run.json").read_text())
        assert run.updates_through() == 0
        assert run.stages[0].learner_transitions == 0
        # Failure accounting does not charge the same wait a second time.
        assert run.stages[0].diagnostic_seconds < 10
    assert run.stages[0].admission_wait_seconds is not None
    # The fake clock's 100 s offset can round to just under 100.
    assert run.stages[0].admission_wait_seconds >= 100 - 1e-6
