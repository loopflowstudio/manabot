"""Experiment data round-trips byte-for-byte, reading each digest from S3 at most once."""

from dataclasses import dataclass, field
import io
from pathlib import Path
import shutil
from unittest.mock import MagicMock

import boto3
import pytest

from manabot import data
from manabot.data import DataManifest, fetch_data, load_manifest, publish_data


@dataclass
class Bucket:
    """An in-memory S3 bucket that counts the object bytes it serves."""

    objects: dict[str, bytes] = field(default_factory=dict)
    bytes_read: int = 0

    def put_object(self, **request: object) -> dict[str, str]:
        body = request["Body"]
        assert isinstance(body, io.BufferedReader)
        self.objects.setdefault(str(request["Key"]), body.read())
        return {}

    def get_object(self, **request: object) -> dict[str, object]:
        stored = self.objects[str(request["Key"])]
        self.bytes_read += len(stored)
        return {"Body": io.BytesIO(stored), "ContentLength": len(stored)}


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Bucket:
    """S3 behind boto3, a repository at tmp_path/repo and a cache at tmp_path/cache."""
    bucket = Bucket()
    client = MagicMock()
    client.put_object.side_effect = bucket.put_object
    client.get_object.side_effect = bucket.get_object
    session = MagicMock()
    session.client.return_value = client
    monkeypatch.setattr(boto3, "Session", MagicMock(return_value=session))
    monkeypatch.setenv("MANABOT_DATA_CACHE", str(tmp_path / "cache"))
    _worktree(monkeypatch, tmp_path / "repo")
    return bucket


def _worktree(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    (root / "experiments").mkdir(parents=True)
    (root / "experiments/data.manifest.json").write_text(
        DataManifest(store="s3://bucket/manabot/", files={}).model_dump_json()
    )
    monkeypatch.setattr(data, "ROOT", root)
    monkeypatch.chdir(root)
    return root


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _publish(root: Path, files: dict[str, bytes]) -> DataManifest:
    for name, content in files.items():
        _write(root / "experiments/data" / name, content)
    manifest_path = root / "experiments/data.manifest.json"
    publish_data(["experiments/data"], manifest_path=manifest_path)
    return load_manifest(manifest_path)


def test_published_data_fetches_back_identically(
    tmp_path: Path, bucket: Bucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    files = {
        "run/result.json": b'{"wins": 3}\n',
        "run/traces/0.jsonl.gz": b"\x1f\x8b trace",
        "same.json": b'{"wins": 3}\n',
    }
    manifest = _publish(root, files)

    assert sorted(manifest.files) == [f"experiments/data/{name}" for name in files]
    assert len(bucket.objects) == 2  # equal bytes share one digest
    assert (
        publish_data(
            ["experiments/data"], manifest_path=root / "experiments/data.manifest.json"
        )
        == []
    )

    shutil.rmtree(root / "experiments/data")
    monkeypatch.chdir(root / "experiments")  # paths are relative to the cwd
    first = fetch_data(["data/run"], manifest=manifest)
    everything = fetch_data(manifest=manifest)

    assert first.fetched == [
        "experiments/data/run/result.json",
        "experiments/data/run/traces/0.jsonl.gz",
    ]
    assert everything.fetched == ["experiments/data/same.json"]
    assert everything.present == first.fetched
    for name, content in files.items():
        installed = root / "experiments/data" / name
        assert installed.read_bytes() == content
        installed.write_bytes(b"edited")  # a worktree file never aliases the cache
    assert sorted(
        path.read_bytes() for path in (tmp_path / "cache").iterdir()
    ) == sorted(set(files.values()))
    assert not list(tmp_path.rglob(".fetch-*"))


def test_store_reads_stay_within_one_download_per_digest_per_machine(
    tmp_path: Path, bucket: Bucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"a.json": b"a" * 1000, "b/trace.gz": b"b" * 300, "copy.json": b"a" * 1000}
    written = 1300
    manifest = _publish(tmp_path / "repo", files)
    assert bucket.bytes_read == 0  # publishing reads nothing back

    shutil.rmtree(tmp_path / "repo/experiments/data")
    shutil.rmtree(tmp_path / "cache")  # a machine that has never seen the data
    cold = fetch_data(manifest=manifest)
    _worktree(monkeypatch, tmp_path / "second")
    warm = fetch_data(manifest=manifest)

    assert (cold.downloaded, cold.downloaded_bytes, cold.from_cache) == (2, written, 0)
    assert (warm.downloaded, warm.from_cache, len(warm.fetched)) == (0, 2, 3)
    assert bucket.bytes_read == written


def test_fetch_never_overwrites_changed_local_data(
    tmp_path: Path, bucket: Bucket
) -> None:
    manifest = _publish(tmp_path / "repo", {"result.json": b"recorded"})
    path = tmp_path / "repo/experiments/data/result.json"
    path.write_bytes(b"edited locally")

    with pytest.raises(ValueError, match="differs from experiments/data.manifest"):
        fetch_data(manifest=manifest)
    assert path.read_bytes() == b"edited locally"


def test_corrupt_store_bytes_never_reach_the_cache_or_path(
    tmp_path: Path, bucket: Bucket
) -> None:
    manifest = _publish(tmp_path / "repo", {"result.json": b"recorded"})
    shutil.rmtree(tmp_path / "repo/experiments/data")
    shutil.rmtree(tmp_path / "cache")
    (key,) = bucket.objects
    bucket.objects[key] = b"recordeX"

    with pytest.raises(ValueError, match="differ"):
        fetch_data(manifest=manifest)
    assert not (tmp_path / "repo/experiments/data/result.json").exists()
    assert not [path for path in (tmp_path / "cache").iterdir()]


def test_unknown_paths_are_errors_not_empty_fetches(
    tmp_path: Path, bucket: Bucket
) -> None:
    manifest_path = tmp_path / "repo/experiments/data.manifest.json"
    with pytest.raises(ValueError, match="no manifest entry"):
        fetch_data(["experiments/data/absent"], manifest=load_manifest(manifest_path))
    with pytest.raises(ValueError, match="under experiments/data"):
        publish_data(["README.md"], manifest_path=manifest_path)
