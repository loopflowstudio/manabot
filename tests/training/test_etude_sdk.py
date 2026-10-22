"""Public SDK admission: local recording, artifact retention, and offline reports.

This is infrastructure proof only. The fixture does not train a policy, launch a
job, or claim that an SDK artifact has passed manabot checkpoint admission.
"""

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

from etude.train.read import read_all
from etude.train.records import Step
from etude.train.report import page_data, write
from etude.train.run import track
import pytest

from manabot.infra.experiment import Experiment
from manabot.infra.hypers import ExperimentHypers


def test_public_sdk_records_checkpoint_and_renders_offline(tmp_path: Path) -> None:
    artifact = tmp_path / "fixture.bin"
    payload = b"mtg122-sdk-admission-no-training"
    artifact.write_bytes(payload)
    directory = tmp_path / "records"
    run = track(
        experiment="mtg122-sdk-admission",
        clocks=[Step("updates", "completed collect/update iterations")],
        directory=directory,
        strict=True,
    )
    try:
        run.log({"fixture_loss": 0.5}, updates=1)
        reference = run.checkpoint(artifact, updates=1)
    finally:
        run.finish()

    assert reference.sha256 == hashlib.sha256(payload).hexdigest()
    stored = Path(unquote(urlparse(reference.uri).path))
    artifact.unlink()
    assert stored.read_bytes() == payload
    data = read_all(directory)
    assert len(data) == 1
    assert len(data[0].checkpoints) == 1
    report = write(page_data("mtg122-sdk-admission", data), tmp_path / "report")
    assert "mtg122-sdk-admission" in report.read_text()


def test_experiment_uses_sdk_even_with_historical_wandb_disabled(
    tmp_path: Path,
) -> None:
    with Experiment(
        ExperimentHypers(exp_name="sdk-fixture", runs_dir=tmp_path, wandb=False)
    ) as experiment:
        experiment.log_update({"loss": 0.25}, updates=2, transitions=512)
    data = read_all(tmp_path / "sdk-fixture" / "etude")
    assert len(data) == 1
    assert data[0].rows == 1
    assert "loss" in data[0].metrics()
    assert data[0].incomplete == 0
    assert data[0].points("loss", "updates").x.tolist() == [2.0]
    assert data[0].points("loss", "learner_transitions").x.tolist() == [512.0]
    assert data[0].state is not None
    assert data[0].state.state == "finished"


def test_experiment_retains_failure_state(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fixture failure"):
        with Experiment(ExperimentHypers(exp_name="failed-fixture", runs_dir=tmp_path)):
            raise RuntimeError("fixture failure")
    data = read_all(tmp_path / "failed-fixture" / "etude")
    assert data[0].state is not None
    assert data[0].state.state == "failed"


def test_experiment_obeys_launcher_identity_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched = tmp_path / "launched"
    monkeypatch.setenv("ETUDE_RUN", "fixture-launched-run")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(launched))
    with Experiment(
        ExperimentHypers(exp_name="launched-fixture", runs_dir=tmp_path / "local"),
        resolved_config={"fixture": "resolved-recipe"},
    ) as experiment:
        experiment.log_update({"loss": 0.5}, updates=1, transitions=16)
    data = read_all(launched)
    assert len(data) == 1
    assert data[0].name == "fixture-launched-run"
    assert data[0].file is not None
    assert "resolved-recipe" in (data[0].file.declaration.config_json or "")
    assert not (tmp_path / "local" / "launched-fixture" / "etude").exists()
