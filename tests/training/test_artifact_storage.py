"""Offline storage checks: corrupt bytes cannot become admitted cache entries."""

import base64
import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import boto3
from botocore.exceptions import ClientError
import pytest

from manabot.infra.artifacts import ArtifactStore, StoredArtifact
from manabot.model.checkpoint import load_checkpoint


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> tuple[ArtifactStore, MagicMock]:
    client = MagicMock()
    session = MagicMock()
    session.client.return_value = client
    monkeypatch.setattr(boto3, "Session", MagicMock(return_value=session))
    client.put_object.return_value = {"VersionId": "version-1"}
    return ArtifactStore(), client


def _reference(data: bytes) -> StoredArtifact:
    digest = hashlib.sha256(data).hexdigest()
    return StoredArtifact(
        uri=f"s3://bucket/manabot/sha256/{digest}", sha256=digest, bytes=len(data)
    )


def _respond(client: MagicMock, data: bytes) -> None:
    client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(data),
        "ContentLength": len(data),
        "VersionId": "version-1",
    }


def test_publish_and_fetch_verify_bytes_and_preserve_version(
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    data = b"retained checkpoint bytes"
    path = tmp_path / "model.pt"
    path.write_bytes(data)
    expected = _reference(data)
    _respond(client, data)
    reference = store.publish(
        path, "s3://bucket/manabot/", sha256=expected.sha256, size=len(data)
    )
    assert reference.uri == expected.uri
    assert reference.version_id == "version-1"
    request = client.put_object.call_args.kwargs
    assert request["IfNoneMatch"] == "*"
    assert (
        request["ChecksumSHA256"]
        == base64.b64encode(bytes.fromhex(reference.sha256)).decode()
    )
    cached = store.fetch(reference, tmp_path / "cache")
    assert cached.read_bytes() == data
    assert client.get_object.call_args.kwargs["VersionId"] == "version-1"
    count = client.get_object.call_count
    assert store.fetch(reference, tmp_path / "cache") == cached
    assert client.get_object.call_count == count


def test_changed_source_never_uploads(
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    reference = _reference(b"original")
    source = tmp_path / "changed"
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="differ"):
        store.publish(
            source, "s3://bucket/prefix", sha256=reference.sha256, size=reference.bytes
        )
    client.put_object.assert_not_called()


def test_existing_remote_object_must_match(
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    data = b"original"
    reference = _reference(data)
    source = tmp_path / "source"
    source.write_bytes(data)
    client.put_object.side_effect = ClientError(
        {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        "PutObject",
    )
    checksum = base64.b64encode(bytes.fromhex(reference.sha256)).decode()
    client.head_object.return_value = {
        "ContentLength": len(data),
        "ChecksumSHA256": checksum,
        "VersionId": "version-1",
    }
    retried = store.publish(
        source, "s3://bucket/prefix", sha256=reference.sha256, size=reference.bytes
    )
    assert retried.sha256 == reference.sha256
    assert retried.version_id == "version-1"
    client.head_object.return_value = {
        "ContentLength": len(data),
        "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"other").digest()).decode(),
    }
    with pytest.raises(ValueError, match="differs"):
        store.publish(
            source, "s3://bucket/prefix", sha256=reference.sha256, size=reference.bytes
        )
    client.get_object.assert_not_called()


def test_bad_download_never_installs_cache(
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    reference = _reference(b"original")
    _respond(client, b"tampered")
    with pytest.raises(ValueError, match="differ"):
        store.fetch(reference, tmp_path)
    assert not (tmp_path / reference.sha256).exists()
    assert not list(tmp_path.glob("download-*"))


def test_loader_checks_cached_digest_before_deserializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference(b"original")
    (tmp_path / reference.sha256).write_bytes(b"tampered")
    monkeypatch.setenv("MANABOT_ARTIFACT_CACHE", str(tmp_path))
    loader = MagicMock()
    monkeypatch.setattr("manabot.model.checkpoint.torch.load", loader)
    with pytest.raises(ValueError, match="differ"):
        load_checkpoint(reference)
    loader.assert_not_called()


def test_r2_locations_round_trip_through_their_own_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored: dict[str, bytes] = {}
    client = MagicMock()

    def put_object(**kwargs: object) -> dict[str, object]:
        body = kwargs["Body"]
        assert isinstance(body, io.BufferedReader)
        stored[str(kwargs["Key"])] = body.read()
        return {}

    client.put_object.side_effect = put_object
    client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(stored[kwargs["Key"]]),
        "ContentLength": len(stored[kwargs["Key"]]),
    }
    monkeypatch.setattr(boto3, "client", MagicMock(return_value=client))
    monkeypatch.setattr(boto3, "Session", MagicMock(side_effect=AssertionError))
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "fixture")
    source = tmp_path / "weights.pt"
    source.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    reference = ArtifactStore().publish(
        source, "r2://manabot/", sha256=digest, size=len(b"weights")
    )
    assert reference.uri == f"r2://manabot/sha256/{digest}"
    assert reference.version_id is None
    fetched = ArtifactStore().fetch(reference, tmp_path / "cache")
    assert fetched.read_bytes() == b"weights"


def test_r2_without_credentials_names_the_missing_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
    reference = _reference(b"x").model_copy(
        update={"uri": f"r2://manabot/sha256/{_reference(b'x').sha256}"}
    )
    with pytest.raises(RuntimeError, match="R2_ACCOUNT_ID"):
        ArtifactStore().fetch(reference, tmp_path / "cache")
