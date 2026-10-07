"""Consistent, immutable generations of remote evidence.

SQLite backup establishes the TrainingRun cut, and its JSON export is derived from
that same backup. Only committed artifact references and closed evaluator attempts
are copied. Producer paths stay intact in Bundle; fetch relocates bytes, not facts.
"""

from pathlib import Path
import shutil
import sqlite3
import tempfile

from pydantic import Field, model_validator

from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact, verify_file
from manabot.training.artifacts import retained_artifacts
from manabot.training.checkpoint_queue import Attempt
from manabot.training.models import TrainingRun
from manabot.training.monitoring import training_dashboard
from manabot.verify.store import VerifyStore

from .bundle import Bundle, BundleFile
from .jobs import RemoteJobSpec
from .plan import Frozen, digest


class JobManifest(Frozen):
    spec_sha256: str
    generation: int = Field(ge=1)
    complete: bool
    bundle: Bundle
    artifacts: tuple[StoredArtifact, ...]

    @model_validator(mode="after")
    def consistent(self) -> "JobManifest":
        if len(self.bundle.files) != len(self.artifacts):
            raise ValueError("snapshot artifact count differs")
        for item, reference in zip(self.bundle.files, self.artifacts, strict=True):
            if item.sha256 != reference.sha256 or item.size != reference.bytes:
                raise ValueError("snapshot artifact bytes differ")
        return self


def _copy_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("evidence symlinks are unsupported")
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def snapshot_evidence(root: Path, snapshot: Path) -> TrainingRun | None:
    """Take an online database backup; never copy a live SQLite/WAL file."""
    snapshot.mkdir(parents=True, exist_ok=True)
    run: TrainingRun | None = None
    exported = root / "run/run.json"
    database = root / "training.sqlite"
    if exported.exists() and database.exists():
        exported_run = TrainingRun.model_validate_json(exported.read_bytes())
        target_db = snapshot / "training.sqlite"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5) as source:
            with sqlite3.connect(target_db) as target:
                source.backup(target)
        with VerifyStore(target_db, read_only=True) as store:
            run = store.training_run(exported_run.id)
        (snapshot / "run").mkdir(exist_ok=True)
        (snapshot / "run/run.json").write_text(run.model_dump_json(indent=2))
        for artifact in retained_artifacts(run).values():
            source_path = Path(artifact.path)
            relative = source_path.relative_to(root)
            if source_path.is_symlink():
                raise ValueError("committed artifact is a symlink")
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            verify_file(target, artifact.sha256, artifact.bytes)
        (snapshot / "training-dashboard.json").write_text(
            training_dashboard(run).model_dump_json(indent=2)
        )
    for name in (
        "training.log",
        "bootstrap.log",
        "toolchain.txt",
        "bootstrap-timing.json",
        "training-exit.txt",
        "supervisor-error.json",
    ):
        path = root / name
        if path.exists():
            # Logs may be growing; this generation binds exactly the copied prefix.
            shutil.copyfile(path, snapshot / name)
    monitoring = root / "monitoring"
    if monitoring.exists():
        for path in sorted(monitoring.glob("attempt-*/attempt.json")):
            attempt = Attempt.model_validate_json(path.read_bytes())
            target = snapshot / path.parent.relative_to(root)
            target.mkdir(parents=True, exist_ok=True)
            if attempt.status != "running":
                _copy_tree(path.parent, target)
            else:
                shutil.copyfile(path, target / "attempt.json")
        # Atomic dashboard JSON records are safe snapshots, even while learning.
        for path in monitoring.glob("**/*dashboard.json"):
            target = snapshot / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return run


def publish_snapshot(
    spec: RemoteJobSpec,
    root: Path,
    generation: int,
    *,
    complete: bool,
    artifacts: S3ArtifactStore | None = None,
) -> StoredArtifact:
    artifacts = artifacts or S3ArtifactStore()
    destination = f"{spec.prefix}/runtime/artifacts"
    with tempfile.TemporaryDirectory(prefix="manabot-snapshot-") as directory:
        snapshot = Path(directory)
        snapshot_evidence(root, snapshot)
        entries: list[BundleFile] = []
        references: list[StoredArtifact] = []
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(snapshot).as_posix()
            item = BundleFile(
                producer_path=str(root / relative),
                relative_path=relative,
                size=path.stat().st_size,
                sha256=digest(path.read_bytes()),
            )
            references.append(
                artifacts.publish(path, destination, sha256=item.sha256, size=item.size)
            )
            entries.append(item)
        manifest = JobManifest(
            spec_sha256=spec.identity,
            generation=generation,
            complete=complete,
            bundle=Bundle(files=tuple(entries)),
            artifacts=tuple(references),
        )
        path = snapshot / "manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        # Publish the manifest only after every referenced blob passes readback.
        return artifacts.publish(
            path,
            destination,
            sha256=digest(path.read_bytes()),
            size=path.stat().st_size,
        )
