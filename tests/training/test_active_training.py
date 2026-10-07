"""Active-time endpoints count completed learner work, excluding slow exports."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import ParamSpec, TypeVar

import pytest

from manabot.training import execution
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.training.monitor_evaluation import stage_checkpoint
from manabot.verify.store import VerifyStore
from tests.training.test_ataraxos_regime import _recipe

P = ParamSpec("P")
R = TypeVar("R")


class Clock:
    def __init__(self) -> None:
        self.offset = 0.0

    def perf_counter(self) -> float:
        return time.perf_counter() + self.offset

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)

    def charge(self, call: Callable[P, R], seconds: float) -> Callable[P, R]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            result = call(*args, **kwargs)
            self.offset += seconds
            return result

        return wrapped


def recipe(updates: int = 10) -> TrainingRegime:
    result = _recipe("ataraxos-move-scalar")
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
    monkeypatch.setattr(
        execution, "update_iteration", clock.charge(execution.update_iteration, 10)
    )
    monkeypatch.setattr(
        execution, "save_bc_checkpoint", clock.charge(execution.save_bc_checkpoint, 100)
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
        assert len(stage.diagnostics) == 2
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
    old = _recipe("ataraxos-move-scalar")
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
    monkeypatch.setattr(execution, "time", clock)
    monkeypatch.setattr(
        execution, "update_iteration", clock.charge(execution.update_iteration, 10)
    )
    gate = tmp_path / "initial-admission"
    out = tmp_path / "run"

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

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(train)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (out / "run.json").exists():
                run = TrainingRun.model_validate_json((out / "run.json").read_text())
                if run.stages and "initial_raw" in run.stages[0].artifacts:
                    break
            if future.done():
                future.result()
            time.sleep(0.05)
        else:
            gate.write_text("timeout")
            pytest.fail("initial checkpoint unavailable")
        assert run.updates_through() == 0
        assert run.stages[0].learner_transitions == 0
        # A slow admission cannot consume the active training allocation.
        clock.offset += 100
        digest = run.stages[0].artifacts["initial_raw"]["sha256"]
        gate.write_text(digest if admitted else "wrong-checkpoint")
        if admitted:
            finished = future.result(timeout=30)
            assert finished.status == "completed"
            assert len(finished.stages[0].diagnostics) == 2
        else:
            with pytest.raises(ValueError, match="admission differs"):
                future.result(timeout=30)
            failed = TrainingRun.model_validate_json((out / "run.json").read_text())
            assert failed.updates_through() == 0
            assert failed.stages[0].learner_transitions == 0
