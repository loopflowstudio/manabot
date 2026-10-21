"""Recipe identity, horizon schedules and CLI routing without training."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from manabot import cli
from manabot.config.presets import DEFAULT_TRAIN_PRESET
from manabot.training.baseline import baseline
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.models import (
    AtaraxosMoveLearning,
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
    assert isinstance(a.learning, AtaraxosMoveLearning)
    assert isinstance(b.learning, AtaraxosMoveLearning)
    for i, j in ((1, 1), (26, 251), (51, 501), (101, 1001)):
        assert a.learning.rates(i) == pytest.approx(b.learning.rates(j))
    legacy = AtaraxosMoveLearning(gradient="ataraxos_move")
    assert a.learning.rates(1) == legacy.rates(1)
    assert a.learning.rates(101) == legacy.rates(10_000)
    assert "iteration_schedule" not in legacy.model_dump()
    assert short.regime.agent == long.regime.agent
    assert short.regime.match == long.regime.match
    assert a.behavior == b.behavior == "current-self"
    assert a.learning_seats == b.learning_seats == "both"
    assert short.version == long.version == "baseline-v2"
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


def test_cli_smoke_and_named_recipe_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    def smoke(preset: str, overrides: list[str]) -> None:
        calls.append((preset, overrides))

    def train(
        *, steps: int, seed: int, out: Path | None, seconds: float | None
    ) -> SimpleNamespace:
        calls.append((steps, seed, out, seconds))
        return SimpleNamespace(status="completed", notebook="baseline.ipynb")

    monkeypatch.setattr(cli, "_run_train", smoke)
    monkeypatch.setattr(cli, "run_baseline", train)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["train"]).exit_code == 0
    assert calls == [(DEFAULT_TRAIN_PRESET, [])]
    result = runner.invoke(
        cli.app,
        ["train", "--recipe", "baseline", "--steps", "123", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert calls[-1] == (123, 197, tmp_path, None)
    for args in (
        ["--recipe", "nope"],
        ["--steps", "2"],
        ["--recipe", "baseline", "--preset", "simple"],
        ["--recipe", "baseline", "--regime", "x"],
        ["--recipe", "baseline", "--set", "agent.hidden_dim=8"],
    ):
        assert runner.invoke(cli.app, ["train", *args]).exit_code != 0
    assert len(calls) == 2
