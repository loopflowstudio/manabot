"""Reconnect retrieval resumes safely and log observation never fetches models."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from manabot.cli import app
from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact
from manabot.remote import cli, job_client
from manabot.remote.bundle import Bundle, BundleFile
from manabot.remote.jobs import Job, JobRecord, JobStatus
from manabot.remote.plan import digest
from manabot.remote.snapshots import JobManifest
from tests.remote.job_fixtures import FileStore
from tests.remote.test_jobs import specification


class Artifacts(S3ArtifactStore):
    def __init__(self) -> None:
        self.bytes: dict[str, bytes] = {}
        self.downloads: list[str] = []
        self.fail: str | None = None

    def add(self, data: bytes) -> StoredArtifact:
        sha = digest(data)
        self.bytes[sha] = data
        return StoredArtifact(
            uri=f"s3://bucket/sha256/{sha}", sha256=sha, bytes=len(data)
        )

    def _download(self, reference: StoredArtifact, target: Path) -> str | None:
        self.downloads.append(reference.sha256)
        if self.fail == reference.sha256:
            self.fail = None
            raise ConnectionError("interrupted")
        target.write_bytes(self.bytes[reference.sha256])
        return None


@dataclass(frozen=True)
class Evidence:
    spec: Job
    record: JobRecord
    artifacts: Artifacts


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Evidence:
    store = FileStore(tmp_path / "control.sqlite")
    spec = specification(store)
    artifacts = Artifacts()
    log = artifacts.add(b"training started\ntraining ended\n")
    weights = artifacts.add(b"model weights should not be downloaded for a log")
    manifest = JobManifest(
        spec_sha256=spec.identity,
        generation=1,
        complete=True,
        bundle=Bundle(
            files=tuple(
                BundleFile(
                    producer_path=f"/worker/{name}",
                    relative_path=name,
                    size=ref.bytes,
                    sha256=ref.sha256,
                )
                for name, ref in (("training.log", log), ("run/raw.pt", weights))
            )
        ),
        artifacts=(log, weights),
    )
    record = JobRecord(
        spec_sha256=spec.identity,
        pod_id="pod",
        phase="completed",
        accepted_at=1,
        heartbeat_at=2,
        generation=1,
        artifacts_complete=True,
        manifest=artifacts.add(manifest.model_dump_json().encode()),
    )
    store.create("runtime/record.json", record.model_dump_json().encode())
    monkeypatch.setattr(job_client, "S3JobStore", lambda prefix: store)
    monkeypatch.setattr(job_client, "S3ArtifactStore", lambda: artifacts)
    return Evidence(spec, record, artifacts)


def test_fetch_resumes_partial_generation_and_is_idempotent(
    evidence: Evidence, tmp_path: Path
) -> None:
    spec, record, artifacts = evidence.spec, evidence.record, evidence.artifacts
    manifest = job_client.job_manifest(spec, record, tmp_path / "cache")
    artifacts.fail = manifest.artifacts[1].sha256
    output = tmp_path / "returned"
    with pytest.raises(ConnectionError):
        job_client.fetch_job(spec, output)
    partial = output / "generation-000001"
    assert (partial / "training.log").exists()
    assert not (partial / "bundle.json").exists()
    assert not (partial / "run/raw.pt").exists()
    path = job_client.fetch_job(spec, output)
    manifest.bundle.verify(path)
    downloads = artifacts.downloads.copy()
    assert job_client.fetch_job(spec, output) == path
    assert artifacts.downloads == downloads
    assert artifacts.downloads.count(manifest.artifacts[0].sha256) == 1


def test_fetch_never_overwrites_corrupt_or_conflicting_evidence(
    evidence: Evidence, tmp_path: Path
) -> None:
    spec = evidence.spec
    output = tmp_path / "returned"
    path = job_client.fetch_job(spec, output)
    (path / "training.log").write_bytes(b"retained corrupt bytes")
    with pytest.raises(ValueError, match="size|digest"):
        job_client.fetch_job(spec, output)
    assert (path / "training.log").read_bytes() == b"retained corrupt bytes"


def test_logs_cli_only_fetches_observed_log_and_manifest(
    evidence: Evidence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, record, artifacts = evidence.spec, evidence.record, evidence.artifacts
    monkeypatch.setattr(cli, "load_job", lambda *args: spec)
    monkeypatch.setattr(
        cli,
        "job_status",
        lambda spec: JobStatus(spec=spec, record=record, provider_state="absent"),
    )
    result = CliRunner().invoke(app, ["deploy", "logs", "--job-id", spec.job_id])
    assert result.exit_code == 0, result.output
    assert result.output == "training started\ntraining ended\n"
    assert len(artifacts.downloads) == 2
    assert (
        digest(b"model weights should not be downloaded for a log")
        not in artifacts.downloads
    )
    assert (
        job_client.fetch_job_file(spec, record, "unpublished.log", tmp_path / "cache")
        is None
    )


def test_pinned_manifest_rejects_a_different_generation(
    evidence: Evidence, tmp_path: Path
) -> None:
    spec, record = evidence.spec, evidence.record
    with pytest.raises(ValueError, match="generation differs"):
        job_client.fetch_job_file(
            spec,
            record.model_copy(update={"generation": 2}),
            "training.log",
            tmp_path / "cache",
        )


def test_fetch_rejects_symlinked_transport_metadata(
    evidence: Evidence, tmp_path: Path
) -> None:
    spec = evidence.spec
    root = tmp_path / "returned/generation-000001"
    root.mkdir(parents=True)
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    (root / "job.json").symlink_to(target)
    with pytest.raises(ValueError, match="metadata is a symlink"):
        job_client.fetch_job(spec, root.parent)
    assert target.read_text() == "keep"
