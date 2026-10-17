"""Exercise deadline retention and progress publication through real processes."""

import json
from pathlib import Path
import sys
import time

import pytest
import torch

from experiments.runners.cuda_admission import diagnose
from experiments.runners.etude_progress import ProgressStore
from experiments.runners.gpu_job import Deadlines, Worker
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import assert_learning_diagnostics_equal, recipe


def test_cuda_failure_keeps_diagnostics_without_exposing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> None:
        raise RuntimeError("CUDA unknown error")

    monkeypatch.setattr(torch.cuda, "init", fail)
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-appear")
    target = tmp_path / "cuda.json"
    with pytest.raises(RuntimeError, match="CUDA unknown error"):
        diagnose(target)
    text = target.read_text()
    assert json.loads(text)["status"] == "failed"
    assert "must-not-appear" not in text


def test_expired_reserve_prevents_new_work(tmp_path: Path) -> None:
    worker = Worker(tmp_path, Deadlines(time.time() + 1, retention_seconds=2))
    with pytest.raises(TimeoutError, match="retention reserve"):
        worker.command("forbidden", ["false"], 1)
    assert not (tmp_path / "forbidden.log").exists()


def test_timeout_retains_partial_files_and_stops_child(tmp_path: Path) -> None:
    marker = tmp_path / "checkpoint.pt"
    late = tmp_path / "should-not-exist"
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({str(marker)!r}).write_bytes(b'partial checkpoint')\n"
        "print('setup completed', flush=True)\ntime.sleep(10)\n"
        f"Path({str(late)!r}).touch()\n"
    )
    worker = Worker(tmp_path, Deadlines(time.time() + 20, retention_seconds=2))
    started = time.monotonic()
    assert worker.command("training", [sys.executable, str(child)], 0.5) == 124
    assert time.monotonic() - started < 5
    assert marker.read_bytes() == b"partial checkpoint"
    assert "setup completed" in (tmp_path / "training.log").read_text()
    assert not late.exists()
    assert (
        json.loads((tmp_path / "worker-status.json").read_text())["phase"]
        == "training_interrupted"
    )


def test_attempt_paths_do_not_collide_on_reused_host(tmp_path: Path) -> None:
    for attempt in ("one", "two"):
        output = tmp_path / attempt
        output.mkdir()
        worker = Worker(output, Deadlines(time.time() + 20, retention_seconds=2))
        target = output / "study"
        assert (
            worker.command(
                "training",
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(target)!r}).mkdir()",
                ],
                5,
            )
            == 0
        )
        assert target.is_dir()


def test_progress_retains_loadable_checkpoint_without_changing_learning(
    tmp_path: Path,
) -> None:
    spec = recipe()
    torch.set_num_threads(1)
    plain, published = tmp_path / "plain", tmp_path / "published"
    plain.mkdir()
    published.mkdir()
    with VerifyStore(plain / "training.sqlite") as store:
        baseline = execute_regime(
            spec, 13561, plain / "run", store, checkpoint_updates=(1, 2)
        )
    with ProgressStore(published / "training.sqlite", published) as store:
        observed = execute_regime(
            spec, 13561, published / "run", store, checkpoint_updates=(1, 2)
        )
    assert baseline.status == observed.status == "completed"
    assert_learning_diagnostics_equal(baseline.stages[0], observed.stages[0])
    receipt = json.loads((published / "etude-result.json").read_text())
    assert receipt["clocks"]["updates"] == 3
    checkpoint = published / receipt["checkpoint"]
    assert checkpoint.is_file()
    actual, _ = load_checkpoint_agent(str(checkpoint))
    expected, _ = load_checkpoint_agent(str(plain / "run" / checkpoint.name))
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
