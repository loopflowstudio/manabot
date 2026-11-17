"""Public SDK admission: local recording and artifact retention.

This is infrastructure proof only. The fixture does not train a policy, launch a
job, or claim that an SDK artifact has passed manabot checkpoint admission.
"""

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

from etude.errors import EtudeError
from etude.records import Clock
from etude.train.read import read_all
from etude.train.tracking import track
import pytest

from tests.fixtures.training_records import recording


def test_public_sdk_records_and_retains_a_checkpoint(tmp_path: Path) -> None:
    artifact = tmp_path / "fixture.bin"
    payload = b"mtg122-sdk-admission-no-training"
    artifact.write_bytes(payload)
    directory = tmp_path / "records"
    run = recording(
        experiment="mtg122-sdk-admission",
        clocks=[
            Clock(
                name="updates",
                kind="step",
                meaning="completed collect/update iterations",
            )
        ],
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


def test_training_without_assignment_refuses_before_creating_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ETUDE_EXPERIMENT", raising=False)
    monkeypatch.delenv("ETUDE_RUN", raising=False)
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    with pytest.raises(EtudeError, match="training_run_not_assigned"):
        track(clocks=["updates"])
    assert not (tmp_path / "home").exists()
