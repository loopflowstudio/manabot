"""Recipe identity, horizon schedules and CLI routing without training."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from manabot import cli
from manabot.training.baseline import baseline
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.local import LOCAL_SECONDS, LOCAL_STEPS, local_smoke
from manabot.training.models import (
    DampedPolicyLearning,
    IterationSchedule,
    TrainSelfPlay,
)


def test_recipe_report_dependency_is_installed(tmp_path: Path) -> None:
    path = write_comparison_notebook(tmp_path, tmp_path / "comparison.ipynb")
    assert path.is_file()


def test_baseline_horizon_preserves_recipe_and_rate_endpoints() -> None:
    short, long = baseline(101), baseline(1001)
    a, b = short.regime.stages[0], long.regime.stages[0]
    assert isinstance(a, TrainSelfPlay) and isinstance(b, TrainSelfPlay)
    assert isinstance(a.learning, DampedPolicyLearning)
    assert isinstance(b.learning, DampedPolicyLearning)
    for i, j in ((1, 1), (26, 251), (51, 501), (101, 1001)):
        assert a.learning.rates(i) == pytest.approx(b.learning.rates(j))
    legacy = DampedPolicyLearning(gradient="damped_policy")
    assert a.learning.rates(1) == legacy.rates(1)
    assert a.learning.rates(101) == legacy.rates(10_000)
    assert "iteration_schedule" not in legacy.model_dump()
    assert short.regime.agent == long.regime.agent
    assert short.regime.match == long.regime.match
    assert a.behavior == b.behavior == "current-self"
    assert a.learning_seats == b.learning_seats == "both"
    assert short.version == long.version == "baseline-v3"
    assert a.opponent is None and a.trainable == "policy_value"
    plan = short.experiment(197).schedule
    assert plan is not None
    assert plan.monitoring.updates == (0, 101)
    assert len(plan.monitoring.protocol.deal_seeds) == 24
    plan.admit_regime(short.regime)
    # Watchdog allowance is independent of the learning schedule.
    assert baseline(101, seconds=400).regime.stages[0].learning == a.learning


def test_schedule_boundaries_and_invalid_lengths() -> None:
    assert IterationSchedule(steps=1).at(1) == 1
    assert IterationSchedule(steps=10).at(20) == 10_000
    with pytest.raises(ValueError):
        baseline(0)
    with pytest.raises(ValueError):
        baseline(True)
    with pytest.raises(ValueError):
        IterationSchedule(steps=10).at(0)


def test_cli_smoke_and_named_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            id="smoke",
            notebook="baseline.ipynb",
            stages=[SimpleNamespace(artifacts={"raw": {"path": "policy.pt"}})],
        )

    monkeypatch.setattr(cli, "run_local", run)
    monkeypatch.setattr(cli, "run_baseline", run)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["train"])
    assert result.exit_code == 0, result.output
    assert "checkpoint: policy.pt" in result.output
    assert calls == [
        dict(
            steps=LOCAL_STEPS,
            seed=197,
            out=None,
            seconds=LOCAL_SECONDS,
            learning_seats="both",
        )
    ]
    result = runner.invoke(
        cli.app,
        ["train", "--recipe", "baseline", "--steps", "123", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert calls[-1] == dict(
        steps=123, seed=197, out=tmp_path, seconds=None, learning_seats="both"
    )
    result = runner.invoke(
        cli.app, ["train", "--steps", "1", "--learning-seats", "one"]
    )
    assert result.exit_code == 0, result.output
    assert calls[-1]["learning_seats"] == "one"
    for args in (
        ["--recipe", "nope"],
        ["--preset", "local"],
        ["--recipe", "baseline", "--preset", "simple"],
        ["--recipe", "baseline", "--regime", "x"],
        ["--recipe", "baseline", "--set", "agent.hidden_dim=8"],
        ["--regime", "", "--out", str(tmp_path)],
        ["--resume-from", "policy.pt"],
        ["--learning-seats", "three"],
    ):
        assert runner.invoke(cli.app, ["train", *args]).exit_code != 0
    assert len(calls) == 3


def test_local_recipe_is_bounded_and_preserves_seat_controls() -> None:
    for seats in ("one", "both"):
        plan = local_smoke(learning_seats=seats)
        stage = plan.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.learning_seats == seats
        assert stage.updates == 2 and stage.streams == 2 and stage.transitions == 16
        assert stage.execution.device == "cpu" and plan.wall_seconds == 120
        assert isinstance(stage.learning, DampedPolicyLearning)
        assert stage.learning.iteration_schedule.steps == 2
        assert plan.agent.policy_film
