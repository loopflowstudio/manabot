"""Learner-decision horizons stop and schedule by real committed rows."""

import json
from pathlib import Path

import pytest

from manabot.backends import resolve
from manabot.experiment.job import TrainingJob, run_worker
from manabot.training.execution import execute_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    DampedPolicyLearning,
    IterationSchedule,
    LearnerDecisionSchedule,
    PretrainedPolicy,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.update_metrics import read_updates, stage_rows
from manabot.verify.store import VerifyStore
from tests.training.test_etude_job import ema_regime, worker_environment
from tests.training.test_learning_state import continuation, reference


def specification(*, updates: int = 100) -> TrainingJob:
    recipe = ema_regime(updates)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert isinstance(stage.learning, DampedPolicyLearning)
    stage.learning_seats = "one"
    stage.batch_closure = "exact-quota"
    stage.transitions = 4
    stage.target_learner_transitions = 26
    stage.learning.iteration_schedule = None
    stage.learning.decision_schedule = LearnerDecisionSchedule(transitions=26)
    return TrainingJob(regime=recipe, experiment="fixture", milestones=(1,))


def test_rates_follow_decisions_independently_of_update_count() -> None:
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        decision_schedule=LearnerDecisionSchedule(transitions=1000),
    )
    assert learning.rates(1, learner_transitions=0) == (1e-4, 0.05)
    assert learning.rates(2, learner_transitions=500) == learning.rates(
        99, learner_transitions=500
    )
    assert learning.rates(20, learner_transitions=1000) == learning.rates(
        21, learner_transitions=1200
    )
    with pytest.raises(ValueError, match="completed transitions"):
        learning.rates(1)
    with pytest.raises(ValueError, match="nonnegative"):
        learning.rates(1, learner_transitions=-1)
    with pytest.raises(ValueError, match="choose an iteration"):
        DampedPolicyLearning(
            gradient="damped_policy",
            iteration_schedule=IterationSchedule(steps=100),
            decision_schedule=LearnerDecisionSchedule(transitions=1000),
        )
    legacy = DampedPolicyLearning(gradient="damped_policy")
    assert "decision_schedule" not in legacy.model_dump()


def test_worker_stops_after_crossing_decision_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = specification()
    spec.milestones = (1, 2, 3, 4)
    worker_environment(tmp_path, monkeypatch, spec)
    run = run_worker()
    assert run.status == "completed", run.error
    assert run.updates_through() == 4 < spec.updates
    assert run.learner_transitions_through() == 32
    assert [point.learner_transitions for point in run.monitoring_checkpoints] == [
        8,
        16,
        24,
    ]
    result = json.loads((tmp_path / "output/etude-result.json").read_text())
    assert result["clocks"] == {"updates": 4, "learner_transitions": 32}
    rows = stage_rows(run, read_updates(tmp_path, [run]))["fit"]
    assert [row["schedule_clock"] for row in rows] == ["learner_transitions"] * 4
    assert [row["schedule_progress"] for row in rows] == pytest.approx(
        [0, 8 / 26, 16 / 26, 24 / 26]
    )


def test_frozen_job_exposes_the_scientific_clock() -> None:
    spec = specification()
    job = spec.job(
        repo=Path(__file__).resolve().parents[2], revision="HEAD", max_hours=1
    )
    assert str(job.until) == "learner_transitions=26"
    assert spec.updates == 100


def test_update_ceiling_does_not_claim_the_decision_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = specification(updates=2)
    worker_environment(tmp_path, monkeypatch, spec)
    with pytest.raises(RuntimeError, match="learner-decision target"):
        run_worker()


def test_decision_target_keeps_complete_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = specification()
    stage = spec.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "both"
    stage.batch_closure = "complete-games"
    stage.games_per_stream = 1
    stage.target_learner_transitions = 1
    worker_environment(tmp_path, monkeypatch, spec)
    run = run_worker()
    assert run.status == "completed", run.error
    assert run.updates_through() == 1
    assert run.stages[0].games == 2
    assert run.learner_transitions_through() > 1


def execute(recipe: TrainingRegime, root: Path, label: str) -> TrainingRun:
    with VerifyStore(root / "training.sqlite") as store:
        run = execute_regime(
            recipe, 12250, root / label, store, backend=resolve("torch")
        )
    assert run.status == "completed", run.error
    return run


def test_continuation_retains_absolute_decision_clock(tmp_path: Path) -> None:
    recipe = specification().regime
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.target_learner_transitions = 16
    parent = execute(recipe, tmp_path, "parent")
    child_recipe = continuation(parent, tmp_path / "parent/run.json", 100)
    child_stage = child_recipe.stages[0]
    assert isinstance(child_stage, TrainSelfPlay)
    child_stage.target_learner_transitions = 26
    child = execute(child_recipe, tmp_path, "continued")
    assert child.updates_through() == 4
    assert child.stages[0].updates == 2
    assert child.learner_transitions_through() == 32
    rows = stage_rows(child, read_updates(tmp_path, [child]))["fit"]
    assert [row["schedule_progress"] for row in rows] == pytest.approx(
        [16 / 26, 24 / 26]
    )


def test_pretrained_producer_counts_do_not_advance_fresh_schedule(
    tmp_path: Path,
) -> None:
    recipe = specification().regime
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.target_learner_transitions = 16
    parent = execute(recipe, tmp_path, "producer")
    branch = specification().regime
    branch_stage = branch.stages[0]
    assert isinstance(branch_stage, TrainSelfPlay)
    branch_stage.pretrained = PretrainedPolicy(
        source_run=reference(tmp_path / "producer/run.json"),
        source_stage="fit",
        checkpoint=parent.stages[0].artifacts["ema"],
        weights="ema",
    )
    run = execute(branch, tmp_path, "branch")
    assert run.updates_through() == 4
    assert run.learner_transitions_through() == 32
    assert run.stages[0].learning_state_origin is None
    assert run.stages[0].producer_cost is not None
    rows = stage_rows(run, read_updates(tmp_path, [run]))["fit"]
    assert rows[0]["schedule_progress"] == 0
