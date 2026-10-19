"""Exercise deadline retention and progress publication through real processes."""

import json
from pathlib import Path
import subprocess
import sys
import time
import venv

import pytest
import torch

from experiments.runners.cuda_admission import diagnose
from experiments.runners.etude_progress import ProgressStore
from experiments.runners.gpu_job import (
    Deadlines,
    Worker,
    main,
    project_interpreter,
    write_retained_manifest,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.verify.store import VerifyStore
from tests.training.test_recovery import assert_learning_diagnostics_equal, recipe


def test_manifest_separates_courier_metadata_from_downloadable_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "etude-result.json").write_text('{"clocks":{"updates":1}}')
    (tmp_path / "manifest.json").write_text("stale manifest")
    nested = tmp_path / "study"
    nested.mkdir()
    (nested / "checkpoint.pt").write_bytes(b"retained checkpoint")
    (nested / "etude-result.json").write_text("ordinary nested file")
    write_retained_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {f["path"] for f in manifest["files"]} == {
        "study/checkpoint.pt",
        "study/etude-result.json",
    }
    assert "attempt clocks/checkpoint" in manifest["courier_metadata"]
    assert (tmp_path / "etude-result.json").exists()


def test_worker_uses_project_environment_through_python_symlink(tmp_path: Path) -> None:
    environment = tmp_path / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = project_interpreter(tmp_path)
    assert Path(interpreter).is_symlink()
    result = subprocess.run(
        [interpreter, "-c", "import sys; print(sys.prefix)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == environment


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


def test_explicit_learning_limit_reaches_both_learner_and_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, tuple[list[str], float]] = {}

    def command(self: Worker, name: str, args: list[str], seconds: float) -> int:
        calls[name] = (args, seconds)
        return 0

    def ignore_signal(signum: int, handler: object) -> None:
        pass

    monkeypatch.setattr(Worker, "command", command)
    monkeypatch.setattr("experiments.runners.gpu_job.signal.signal", ignore_signal)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ETUDE_DEADLINE", "2099-01-01T00:00:00+00:00")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu_job",
            "--study",
            "batching-grid",
            "--size",
            "middle",
            "--seed",
            "13591",
            "--training-seconds",
            "2400",
            "--checkpoint-transitions",
            "131072",
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    argv, limit = calls["training"]
    assert argv[argv.index("--seconds") + 1] == "2400.0"
    assert argv[argv.index("--checkpoint-transitions") + 1] == "131072"
    assert limit == 2460


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_training_limit_is_rejected_before_worker_creation(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "not-created"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu_job",
            "--study",
            "batching-grid",
            "--size",
            "small",
            "--seed",
            "13591",
            "--training-seconds",
            value,
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert not (tmp_path / "not-created").exists()


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
