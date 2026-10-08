"""A snapshot keeps the database cut and committed artifact bytes together."""

from pathlib import Path

import pytest

from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact, verify_file
from manabot.remote.plan import digest
from manabot.remote.snapshots import publish_snapshot, snapshot_evidence
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore
from tests.remote.job_fixtures import FileStore
from tests.remote.test_jobs import specification
from tests.training.test_checkpoint_queue import run_fixture


class PublicationStore(S3ArtifactStore):
    def __init__(self, version: str | None = "v1") -> None:
        self.version = version
        self.published: list[str] = []
        self.unavailable: str | None = None

    def publish(
        self, path: Path, destination: str, *, sha256: str, size: int
    ) -> StoredArtifact:
        verify_file(path, sha256, size)
        self.published.append(path.name)
        if self.unavailable == sha256:
            raise ConnectionError("remote readback unavailable")
        return StoredArtifact(
            uri=f"{destination}/sha256/{sha256}",
            sha256=sha256,
            bytes=size,
            version_id=self.version,
        )


@pytest.mark.parametrize("version", ["v1", None, "null"])
def test_intermediate_publication_reuses_only_verified_versions_and_final_rechecks(
    tmp_path: Path, version: str | None
) -> None:
    spec = specification(FileStore(tmp_path / "control.sqlite"))
    root = tmp_path / "producer"
    root.mkdir()
    (root / "toolchain.txt").write_text("frozen native identity")
    (root / "training.log").write_text("initial")
    artifacts = PublicationStore(version)
    verified: dict[str, StoredArtifact] = {}
    publish_snapshot(
        spec, root, 1, complete=False, artifacts=artifacts, verified=verified
    )
    assert artifacts.published == ["toolchain.txt", "training.log", "manifest.json"]
    artifacts.published.clear()
    (root / "training.log").write_text("advancing")
    publish_snapshot(
        spec, root, 2, complete=False, artifacts=artifacts, verified=verified
    )
    assert artifacts.published == (
        ["training.log", "manifest.json"]
        if version == "v1"
        else ["toolchain.txt", "training.log", "manifest.json"]
    )
    artifacts.unavailable = digest((root / "toolchain.txt").read_bytes())
    with pytest.raises(ConnectionError, match="readback unavailable"):
        publish_snapshot(
            spec, root, 3, complete=True, artifacts=artifacts, verified=verified
        )
    artifacts.unavailable = None
    artifacts.published.clear()
    publish_snapshot(
        spec, root, 3, complete=True, artifacts=artifacts, verified=verified
    )
    assert artifacts.published == ["toolchain.txt", "training.log", "manifest.json"]
    artifacts.published.clear()
    # Losing process-local state requires another complete readback, never trust
    # a new process's knowledge of the content-addressed key alone.
    publish_snapshot(spec, root, 4, complete=False, artifacts=artifacts, verified={})
    assert artifacts.published == ["toolchain.txt", "training.log", "manifest.json"]


def test_failed_publication_never_populates_verified_cache(tmp_path: Path) -> None:
    spec = specification(FileStore(tmp_path / "control.sqlite"))
    root = tmp_path / "producer"
    root.mkdir()
    (root / "training.log").write_text("private failure evidence")
    artifacts = PublicationStore()
    artifacts.unavailable = digest((root / "training.log").read_bytes())
    verified: dict[str, StoredArtifact] = {}
    with pytest.raises(ConnectionError):
        publish_snapshot(
            spec, root, 1, complete=False, artifacts=artifacts, verified=verified
        )
    assert verified == {}
    artifacts.unavailable = None
    publish_snapshot(
        spec, root, 1, complete=False, artifacts=artifacts, verified=verified
    )
    assert artifacts.published == ["training.log", "training.log", "manifest.json"]


def test_snapshot_uses_database_authority_and_preserves_artifact_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer"
    (root / "run").mkdir(parents=True)
    export = root / "run/run.json"
    run = run_fixture(export)
    policy = root / "run/raw.pt"
    policy.write_bytes(b"immutable bytes")
    run.stages[0].artifacts["raw"] = {
        "path": str(policy),
        "sha256": digest(policy.read_bytes()),
        "bytes": policy.stat().st_size,
    }
    with VerifyStore(root / "training.sqlite") as store:
        store.save_training_run(run)
        # The export deliberately lags the authoritative DB, as it can during training.
        snapshot_evidence(root, tmp_path / "snapshot")
    saved = TrainingRun.model_validate_json(
        (tmp_path / "snapshot/run/run.json").read_text()
    )
    with VerifyStore(tmp_path / "snapshot/training.sqlite", read_only=True) as store:
        assert saved == store.training_run(run.id)
    assert saved.stages[0].artifacts["raw"]["path"] == str(policy)
    assert (tmp_path / "snapshot/run/raw.pt").read_bytes() == b"immutable bytes"
    policy.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="bytes differ"):
        snapshot_evidence(root, tmp_path / "bad-snapshot")


def test_snapshot_retains_rejected_diagnostics_without_policy_admission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer"
    (root / "run").mkdir(parents=True)
    run = run_fixture(root / "run/run.json")
    run.stages[0].artifacts = {}
    failed = root / "run/numerical-failure.pt"
    failed.write_bytes(b"private failed tensors, not a loadable policy")
    run.stages[0].rejected_artifacts["numerical_failure"] = {
        "path": str(failed),
        "sha256": digest(failed.read_bytes()),
        "bytes": failed.stat().st_size,
    }
    with VerifyStore(root / "training.sqlite") as store:
        store.save_training_run(run)
    snapshot_evidence(root, tmp_path / "snapshot")
    assert (
        tmp_path / "snapshot/run/numerical-failure.pt"
    ).read_bytes() == failed.read_bytes()
    saved = TrainingRun.model_validate_json(
        (tmp_path / "snapshot/run/run.json").read_bytes()
    )
    assert saved.stages[0].rejected_artifacts == run.stages[0].rejected_artifacts
    assert "numerical_failure" not in saved.stages[0].artifacts


def test_terminal_numerical_bundle_uses_existing_snapshot_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer"
    (root / "run/incident").mkdir(parents=True)
    run = run_fixture(root / "run/run.json")
    run.status = "failed"
    run.stages[0].artifacts = {}
    for name in ("incident.json", "tensors.pt", "state.pt"):
        path = root / "run/incident" / name
        path.write_bytes(f"private fixture {name}".encode())
        run.stages[0].rejected_artifacts[f"numerical/{name}"] = {
            "path": str(path),
            "sha256": digest(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    with VerifyStore(root / "training.sqlite") as store:
        store.save_training_run(run)
    snapshot = tmp_path / "snapshot"
    snapshot_evidence(root, snapshot)
    for reference in run.stages[0].rejected_artifacts.values():
        relative = Path(reference["path"]).relative_to(root)
        assert digest((snapshot / relative).read_bytes()) == reference["sha256"]
