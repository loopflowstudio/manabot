"""Public SDK admission: local recording and artifact retention.

This is infrastructure proof only. The fixture does not train a policy, launch a
job, or claim that an SDK artifact has passed manabot checkpoint admission.
"""

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

from etude.train.read import read_all
from etude.train.records import Step
from etude.train.run import track


def test_public_sdk_records_and_retains_a_checkpoint(tmp_path: Path) -> None:
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
