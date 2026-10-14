"""Public SDK admission: local recording, artifact retention, and offline reports.

This is infrastructure proof only. The fixture does not train a policy, launch a
job, or claim that an SDK artifact has passed manabot checkpoint admission.
"""

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

from etude.track.read import read_all
from etude.track.records import Step
from etude.track.report import page_data, write
from etude.track.run import init


def test_public_sdk_records_checkpoint_and_renders_offline(tmp_path: Path) -> None:
    artifact = tmp_path / "fixture.bin"
    payload = b"mtg122-sdk-admission-no-training"
    artifact.write_bytes(payload)
    directory = tmp_path / "records"
    run = init(
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
