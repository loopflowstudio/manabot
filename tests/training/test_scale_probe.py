"""Scale authoring preserves the scientific ladder and ordinary admission."""

from pathlib import Path
import platform
import subprocess

import pytest

from experiments.runners.calibrate_training import capacity_plan
from experiments.runners.model_capacity import regimes
from experiments.runners.scale_probe import scale, scale_recipes
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.training.recipes import with_capacity


def test_scale_is_explicit_and_bounded() -> None:
    base = TrainingRegime.model_validate(capacity_plan().recipes[0])
    original = regimes(base)
    expanded = regimes(base, include_ataraxos=True)
    assert list(expanded) == [*original, "w384-d8"]
    assert all(expanded[k] == v for k, v in original.items())
    large = scale_recipes()["w384-d8"]
    assert (
        large.agent.hidden_dim,
        large.agent.attention_layers,
        large.agent.attention_feedforward_dim,
    ) == (384, 8, 1536)
    stage = large.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert (stage.updates, stage.transitions, stage.streams) == (1, 8, 2)
    assert stage.learning.min_advantage == 0


def test_capacity_expansion_omission_and_reset() -> None:
    base = scale_recipes()["w384-d8"]
    kept = with_capacity(base, id="kept", width=64, depth=3, heads=4)
    reset = with_capacity(
        base, id="reset", width=64, depth=3, heads=4, feedforward_dim=None
    )
    assert kept.agent.attention_feedforward_dim == 1536
    assert reset.agent.attention_feedforward_dim is None
    assert "attention_feedforward_dim" not in reset.agent.model_dump()
    assert base.agent.hidden_dim == 384


def test_child_timeouts_are_retained_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts: list[float] = []

    def timeout(
        args: list[str],
        *,
        stdout: object,
        stderr: object,
        timeout: float,
        check: bool,
        env: dict[str, str],
    ) -> None:
        timeouts.append(timeout)
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        raise subprocess.TimeoutExpired(args, timeout)

    platform.platform()  # Populate platform cache before intercepting child launches.
    monkeypatch.setattr(subprocess, "run", timeout)
    out = tmp_path / "failed"
    report = scale(out)
    assert len(report.attempts) == len(timeouts) == 12
    assert all(0 < seconds <= 90 for seconds in timeouts)
    assert all(row.status == "failed" and row.result is None for row in report.attempts)
    assert (out / "scale.json").is_file()
    with pytest.raises(FileExistsError):
        scale(out)
