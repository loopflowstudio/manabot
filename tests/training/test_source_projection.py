"""A smaller runtime archive omits only the declared historical-data subtree."""

import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from experiments.runners.prepared_source import PreparedSource
from experiments.runners.source_projection import project_source


def test_projection_preserves_runtime_bytes_modes_and_links(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    files = {
        "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
        "content/cards.json": b'{"rules": "unchanged"}',
        "experiments/regimes/baseline.json": b'{"streams": 4}',
        "experiments/database.json": b"keep the similarly named path",
        "experiments/data/historical.json": b"old measurements",
    }
    with tarfile.open(source, "w") as archive:
        for name, body in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            entry.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(entry, io.BytesIO(body))
        link = tarfile.TarInfo("scripts/alias.sh")
        link.type = tarfile.SYMTYPE
        link.linkname = "run.sh"
        archive.addfile(link)
    destination = tmp_path / "projected.tar.gz"
    receipt = project_source(source, destination, commit="a" * 40)
    assert set(receipt.omitted_files) == {"experiments/data/historical.json"}
    assert receipt.retained_files == {
        name: hashlib.sha256(body).hexdigest()
        for name, body in files.items()
        if not name.startswith("experiments/data/")
    }
    assert receipt.source_tar_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    with tarfile.open(destination) as archive:
        assert archive.getmember("scripts/run.sh").mode == 0o755
        assert archive.getmember("scripts/alias.sh").linkname == "run.sh"
        assert "experiments/data/historical.json" not in archive.getnames()
    with pytest.raises(FileExistsError):
        project_source(source, destination, commit="a" * 40)


def test_projection_transport_requires_a_manifest() -> None:
    with pytest.raises(ValueError, match="explicit manifest digest"):
        PreparedSource(
            status="verified",
            bucket="campaign",
            key="prepared/source.tar.gz",
            commit="a" * 40,
            bytes=10,
            sha256="b" * 64,
            transport="history-excluded-gzip",
            source_tar_bytes=20,
            source_tar_sha256="c" * 64,
        )
