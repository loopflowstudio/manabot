"""Recipe identity, horizon schedules and CLI routing without training."""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from typer.testing import CliRunner

from manabot import cli
from manabot.recipes import local
from manabot.recipes.baseline import TrainingRecipe, baseline, baseline_test
from manabot.recipes.device import accelerator
from manabot.recipes.local import LOCAL_SECONDS, LOCAL_STEPS, local_smoke
from manabot.reports.notebook import write_comparison_notebook
from manabot.training.regime import (
    DampedPolicyLearning,
    IterationSchedule,
    TrainingRegime,
    TrainSelfPlay,
)


def test_recipe_report_dependency_is_installed(tmp_path: Path) -> None:
    path = write_comparison_notebook(tmp_path, tmp_path / "comparison.ipynb")
    assert path.is_file()


def test_baseline_horizon_preserves_recipe_and_rate_endpoints() -> None:
    short, long = baseline_test(101), baseline_test(1001)
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
    assert baseline_test(101, seconds=400).regime.stages[0].learning == a.learning


def test_schedule_boundaries_and_invalid_lengths() -> None:
    assert IterationSchedule(steps=1).at(1) == 1
    assert IterationSchedule(steps=10).at(20) == 10_000
    with pytest.raises(ValueError):
        baseline_test(0)
    with pytest.raises(ValueError):
        baseline_test(True)
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
            device=None,
        )
    ]
    result = runner.invoke(
        cli.app,
        ["train", "--recipe", "baseline", "--steps", "123", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert calls[-1] == dict(
        steps=123,
        seed=197,
        out=tmp_path,
        seconds=None,
        learning_seats="both",
        device=None,
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
        plan = local_smoke(learning_seats=seats, device="cpu")
        stage = plan.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.learning_seats == seats
        assert stage.updates == 2 and stage.streams == 2 and stage.transitions == 16
        assert stage.execution.device == "cpu" and plan.wall_seconds == 120
        assert isinstance(stage.learning, DampedPolicyLearning)
        assert stage.learning.iteration_schedule.steps == 2
        assert plan.agent.policy_film


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("steps", [1, 101])
def test_cpu_baseline_differs_only_in_device(
    device: Literal["cuda", "mps"], steps: int
) -> None:
    cpu = baseline_test(steps)
    accelerated = baseline(steps, device=device)
    cpu_data, accelerated_data = (
        cpu.regime.model_dump(),
        accelerated.regime.model_dump(),
    )
    assert cpu_data["stages"][0]["execution"].pop("device") == "cpu"
    assert accelerated_data["stages"][0]["execution"].pop("device") == device
    assert cpu_data == accelerated_data
    assert cpu.model_dump(exclude={"regime"}) == accelerated.model_dump(
        exclude={"regime"}
    )
    assert not accelerated.regime.recoverable
    schedule = accelerated.experiment(197).schedule
    assert schedule is not None and not schedule.active_runtime
    schedule.admit_regime(accelerated.regime)


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, None)],
)
@pytest.mark.parametrize("override", [None, "cpu", "cuda", "mps"])
@pytest.mark.parametrize("recipe", ["baseline", "local"])
def test_recipe_and_cli_device_selection(
    cuda: bool,
    mps: bool,
    expected: str | None,
    override: Literal["cpu", "cuda", "mps"] | None,
    recipe: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert accelerator() == expected
    selected = override if override is not None else expected
    rejected = recipe == "baseline" and selected in {None, "cpu"}
    message = "baseline requires CUDA or Apple MPS; use --recipe local for CPU"
    if rejected:
        with pytest.raises(ValueError, match=message):
            baseline(device=override)
    else:
        regime = (
            baseline(device=override).regime
            if recipe == "baseline"
            else local_smoke(device=override)
        )
        assert regime.stages[0].execution.device == (selected or "cpu")

    observed: list[str] = []
    result = SimpleNamespace(
        status="completed",
        id="fixture",
        notebook="baseline.ipynb",
        stages=[SimpleNamespace(artifacts={"raw": {"path": "policy.pt"}})],
    )

    def run_plan(self: TrainingRecipe, **kwargs: object) -> SimpleNamespace:
        observed.append(self.regime.stages[0].execution.device)
        return result

    def execute(plan: TrainingRegime, *args: object) -> SimpleNamespace:
        observed.append(plan.stages[0].execution.device)
        return result

    monkeypatch.setattr(TrainingRecipe, "run", run_plan)
    monkeypatch.setattr(local, "execute_regime", execute)
    args = ["train", "--recipe", recipe, "--out", str(tmp_path / "run")]
    if override is not None:
        args += ["--device", override]
    invoked = CliRunner().invoke(cli.app, args)
    if rejected:
        assert invoked.exit_code != 0 and str(invoked.exception) == message
        assert observed == []
    else:
        assert invoked.exit_code == 0, invoked.output
        assert observed == [selected or "cpu"]


def test_test_baseline_is_not_a_cli_recipe() -> None:
    result = CliRunner().invoke(cli.app, ["train", "--recipe", "baseline_test"])
    assert result.exit_code != 0
