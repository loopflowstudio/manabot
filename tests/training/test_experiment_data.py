"""Experiment data round-trips through the content-addressed store byte-for-byte."""

import io
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest

from manabot import data
from manabot.data import DataManifest, fetch_data, load_manifest, publish_data


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, bytes]:
    """An in-memory S3 bucket behind boto3, with the repository rooted at tmp_path."""
    objects: dict[str, bytes] = {}

    def put_object(**request: object) -> dict[str, str]:
        body = request["Body"]
        assert isinstance(body, io.BufferedReader)
        objects.setdefault(str(request["Key"]), body.read())
        return {}

    def get_object(**request: object) -> dict[str, object]:
        stored = objects[str(request["Key"])]
        return {"Body": io.BytesIO(stored), "ContentLength": len(stored)}

    client = MagicMock()
    client.put_object.side_effect = put_object
    client.get_object.side_effect = get_object
    session = MagicMock()
    session.client.return_value = client
    monkeypatch.setattr(boto3, "Session", MagicMock(return_value=session))
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/data.manifest.json").write_text(
        DataManifest(store="s3://bucket/manabot/", files={}).model_dump_json()
    )
    return objects


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_published_data_fetches_back_identically(
    tmp_path: Path, bucket: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "experiments/data/run"
    _write(run / "result.json", b'{"wins": 3}\n')
    _write(run / "traces/0.jsonl.gz", b"\x1f\x8b trace")
    _write(tmp_path / "experiments/data/same.json", b'{"wins": 3}\n')
    manifest_path = tmp_path / "experiments/data.manifest.json"

    published = publish_data(
        ["experiments/data/run", "experiments/data/same.json"],
        manifest_path=manifest_path,
    )
    manifest = load_manifest(manifest_path)

    assert published == [
        "experiments/data/run/result.json",
        "experiments/data/run/traces/0.jsonl.gz",
        "experiments/data/same.json",
    ]
    assert len(bucket) == 2  # equal bytes share one digest
    assert publish_data(["experiments/data"], manifest_path=manifest_path) == []

    originals = {path: (tmp_path / path).read_bytes() for path in manifest.files}
    for path in originals:
        (tmp_path / path).unlink()
    monkeypatch.chdir(tmp_path / "experiments")  # paths are relative to the cwd
    first = fetch_data(["data/run"], manifest=manifest)
    everything = fetch_data(manifest=manifest)

    assert first.fetched == published[:2]
    assert everything.fetched == ["experiments/data/same.json"]
    assert everything.present == published[:2]
    assert {path: (tmp_path / path).read_bytes() for path in originals} == originals
    assert not list(tmp_path.rglob(".fetch-*"))


def test_fetch_never_overwrites_changed_local_data(
    tmp_path: Path, bucket: dict[str, bytes]
) -> None:
    path = tmp_path / "experiments/data/result.json"
    _write(path, b"recorded")
    manifest_path = tmp_path / "experiments/data.manifest.json"
    publish_data(["experiments/data/result.json"], manifest_path=manifest_path)
    path.write_bytes(b"edited locally")

    with pytest.raises(ValueError, match="differs from experiments/data.manifest"):
        fetch_data(manifest=load_manifest(manifest_path))
    assert path.read_bytes() == b"edited locally"


def test_corrupt_store_bytes_never_reach_the_path(
    tmp_path: Path, bucket: dict[str, bytes]
) -> None:
    path = tmp_path / "experiments/data/result.json"
    _write(path, b"recorded")
    manifest_path = tmp_path / "experiments/data.manifest.json"
    publish_data(["experiments/data/result.json"], manifest_path=manifest_path)
    path.unlink()
    (key,) = bucket
    bucket[key] = b"recordeX"

    with pytest.raises(ValueError, match="differ"):
        fetch_data(manifest=load_manifest(manifest_path))
    assert not path.exists()
    assert not list(tmp_path.rglob(".fetch-*"))


def test_unknown_paths_are_errors_not_empty_fetches(
    tmp_path: Path, bucket: dict[str, bytes]
) -> None:
    manifest_path = tmp_path / "experiments/data.manifest.json"
    with pytest.raises(ValueError, match="no manifest entry"):
        fetch_data(["experiments/data/absent"], manifest=load_manifest(manifest_path))
    with pytest.raises(ValueError, match="under experiments/data"):
        publish_data(["README.md"], manifest_path=manifest_path)
