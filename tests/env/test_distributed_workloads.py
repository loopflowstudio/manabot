"""Run the disposable probes against real native/model APIs in integration CI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.runners import distributed_workloads as workloads
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector


def test_collection_failure_is_not_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_collection(
        self: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float,
    ) -> RolloutBatch:
        raise TimeoutError("fixture collection deadline")

    monkeypatch.setattr(SeatRoutedCollector, "collect", fail_collection)
    monkeypatch.setattr(workloads.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(workloads.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--mode",
            "inference",
            "--seconds",
            "0.01",
            "--out",
            str(tmp_path / "failed"),
        ],
    )
    with pytest.raises(TimeoutError, match="fixture collection deadline"):
        workloads.main()


@pytest.mark.parametrize("mode", ["inference", "simulator"])
def test_bounded_workload(mode: str, tmp_path: Path) -> None:
    """Collection, cleanup and timing must reach a retained measurement."""
    out = tmp_path / mode
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "experiments/runners/distributed_benchmark.py",
            "--mode",
            mode,
            "--seconds",
            "0.01",
            "--timeout",
            "110",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    stderr = out / "stderr.log"
    assert result.returncode == 0, result.stderr + (
        stderr.read_text() if stderr.exists() else ""
    )
    receipt = json.loads((out / "result.json").read_text())
    assert receipt["status"] == "completed"
    measurement = json.loads((out / "workload/measurement.json").read_text())
    assert measurement["units"] > 0
    assert measurement["units_per_second"] > 0
    assert measurement["strength_or_useful_learning_progress_measured"] is False
