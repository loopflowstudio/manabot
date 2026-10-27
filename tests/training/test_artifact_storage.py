"""Offline storage checks: corrupt bytes cannot become admitted cache entries."""

import base64
import hashlib
import io
import json
from pathlib import Path
<<<<<<< HEAD
import sys
=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from unittest.mock import MagicMock

import boto3
from botocore.exceptions import ClientError
import pytest

<<<<<<< HEAD
from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training import monitoring
from manabot.training.artifacts import ArtifactManifest, publish_run


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> tuple[S3ArtifactStore, MagicMock]:
=======
from manabot.experiment.artifacts import ArtifactManifest, publish_run
from manabot.infra.artifacts import ArtifactStore, StoredArtifact
from manabot.model.checkpoint import load_checkpoint


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> tuple[ArtifactStore, MagicMock]:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    client = MagicMock()
    session = MagicMock()
    session.client.return_value = client
    monkeypatch.setattr(boto3, "Session", MagicMock(return_value=session))
    client.put_object.return_value = {"VersionId": "version-1"}
<<<<<<< HEAD
    return S3ArtifactStore(), client
=======
    return ArtifactStore(), client
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


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
<<<<<<< HEAD
    tmp_path: Path, storage: tuple[S3ArtifactStore, MagicMock]
=======
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    tmp_path: Path, storage: tuple[S3ArtifactStore, MagicMock]
=======
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    tmp_path: Path, storage: tuple[S3ArtifactStore, MagicMock]
=======
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    _respond(client, data)
=======
    checksum = base64.b64encode(bytes.fromhex(reference.sha256)).decode()
    client.head_object.return_value = {
        "ContentLength": len(data),
        "ChecksumSHA256": checksum,
        "VersionId": "version-1",
    }
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    retried = store.publish(
        source, "s3://bucket/prefix", sha256=reference.sha256, size=reference.bytes
    )
    assert retried.sha256 == reference.sha256
    assert retried.version_id == "version-1"
<<<<<<< HEAD
    _respond(client, b"tampered")
    with pytest.raises(ValueError, match="differ"):
        store.publish(
            source, "s3://bucket/prefix", sha256=reference.sha256, size=reference.bytes
        )


def test_bad_download_never_installs_cache(
    tmp_path: Path, storage: tuple[S3ArtifactStore, MagicMock]
=======
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
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    monkeypatch.setattr("manabot.sim.flat_mc.torch.load", loader)
    with pytest.raises(ValueError, match="differ"):
        load_checkpoint_agent(reference)
=======
    monkeypatch.setattr("manabot.model.checkpoint.torch.load", loader)
    with pytest.raises(ValueError, match="differ"):
        load_checkpoint(reference)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    loader.assert_not_called()


def test_run_publication_preserves_export_and_all_roles(
<<<<<<< HEAD
    tmp_path: Path, storage: tuple[S3ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    # Retained schema fixture; no trainer, optimizer, or game execution.
    fixture = Path("experiments/study/experiment-demo/training")
=======
    tmp_path: Path, storage: tuple[ArtifactStore, MagicMock]
) -> None:
    store, client = storage
    # Retained schema fixture; no trainer, optimizer, or game execution.
    fixture = Path("tests/fixtures/experiment-demo/training")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    paths = list(fixture.glob("**/run.json"))
    assert paths
    payload = json.loads(paths[0].read_text())
    source_artifact = tmp_path / "model"
    source_artifact.write_bytes(b"artifact")
    reference = _reference(b"artifact")
    local = {
        "path": str(source_artifact),
        "sha256": reference.sha256,
        "bytes": reference.bytes,
    }
    payload["status"] = "completed"
    payload["stages"] = [
        {
            "id": payload["regime"]["stages"][0]["id"],
            "artifacts": {"raw": local, "ema": local},
        }
    ]
    payload["monitoring_checkpoints"] = []
    payload["fixed_validation"] = None
    payload["recovery_artifact"] = None
    source = tmp_path / "run.json"
    source.write_text(json.dumps(payload))
    original = source.read_bytes()
    blobs: dict[str, bytes] = {}

    def put(**kwargs: object) -> dict[str, str]:
        key, body = kwargs["Key"], kwargs["Body"]
        assert isinstance(key, str)
        assert isinstance(body, io.BufferedReader)
        blobs[key] = body.read()
        return {"VersionId": "v1"}

    def get(**kwargs: str) -> dict[str, object]:
        data = blobs[kwargs["Key"]]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    client.put_object.side_effect = put
    client.get_object.side_effect = get
    output = tmp_path / "nested" / "manifest.json"
    manifest = publish_run(source, "s3://bucket/prefix", output, store)
    assert len(manifest.artifacts) == 2
    assert (
        client.put_object.call_count == 2
    )  # shared raw/EMA blob plus exact run export
    assert source.read_bytes() == original
    assert manifest.source_run.sha256 == hashlib.sha256(original).hexdigest()
    assert ArtifactManifest.model_validate_json(output.read_text()) == manifest

    source_artifact.write_bytes(b"modified")
    count = client.put_object.call_count
    with pytest.raises(ValueError, match="differ"):
        publish_run(source, "s3://bucket/prefix", tmp_path / "other.json", store)
    assert client.put_object.call_count == count


<<<<<<< HEAD
def test_monitor_publishes_only_references_bound_to_exact_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = next(
        Path("experiments/study/experiment-demo/training").glob("**/run.json")
    )
    payload = source.read_bytes()
    manifest = ArtifactManifest(
        training_run_id=json.loads(payload)["id"],
        source_run=_reference(payload),
        artifacts={"stages/policy-0/artifacts/raw": _reference(b"model bytes")},
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    published: list[monitoring.Dashboard] = []

    def publish(
        dashboard: monitoring.Dashboard, out: Path, *, project: str, entity: str | None
    ) -> str:
        published.append(dashboard)
        return "https://wandb.invalid/run"

    monkeypatch.setattr(monitoring, "publish_dashboard", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitoring",
            "--run",
            str(source),
            "--artifact-manifest",
            str(path),
            "--out",
            str(tmp_path / "dashboard"),
            "--online",
        ],
    )
    monitoring.main()
    assert published[0].summary["artifact_storage"] == manifest.model_dump(mode="json")
    assert (
        published[0].rows
        == monitoring.training_dashboard(
            monitoring.TrainingRun.model_validate_json(payload)
        ).rows
    )
    bad = manifest.model_copy(update={"training_run_id": "other-run"})
    path.write_text(bad.model_dump_json())
    with pytest.raises(ValueError, match="another training run"):
        monitoring.main()
    assert len(published) == 1
=======
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
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
