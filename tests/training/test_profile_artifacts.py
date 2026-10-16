"""Large model artifacts cannot hide the independently readable timing records."""

import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

import pytest

from experiments.runners.profile_artifacts import package


def test_bounded_parts_reconstruct_every_original_file(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    files = {
        "manifest.json": b"[]",
        "case/run/run.json": b'{"seconds": 42}',
        "case/model.pt": random.Random(37).randbytes(100_000),
        "case/training.sqlite": b"retained database",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    out = tmp_path / "outputs"
    package(root, out, part_bytes=4096)
    index = json.loads((out / "01-artifact-index.json").read_text())
    with tarfile.open(out / index["metadata"]["name"]) as archive:
        assert set(archive.getnames()) == {
            "training-evidence/manifest.json",
            "training-evidence/case/run/run.json",
        }
        assert (
            archive.extractfile("training-evidence/case/run/run.json").read()
            == files["case/run/run.json"]
        )
    parts = index["complete_archive_parts"]
    assert len(parts) > 1
    chunks = []
    for part in parts:
        data = (out / part["name"]).read_bytes()
        assert len(data) == part["bytes"] <= 4096
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        chunks.append(data)
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r:gz") as archive:
        assert {
            m.name.removeprefix("training-evidence/"): archive.extractfile(m).read()
            for m in archive
        } == files
    with pytest.raises(FileExistsError):
        package(root, out)


def test_packaging_rejects_recursive_output_and_external_links(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    (root / "manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="outside"):
        package(root, root / "outputs")
    external = tmp_path / "external"
    external.write_text("not study evidence")
    (root / "link.json").symlink_to(external)
    with pytest.raises(ValueError, match="regular files"):
        package(root, tmp_path / "out")
