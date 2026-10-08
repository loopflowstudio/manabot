"""A snapshot keeps the database cut and committed artifact bytes together."""

from pathlib import Path

import pytest

from manabot.remote.plan import digest
from manabot.remote.snapshots import snapshot_evidence
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore
from tests.training.test_checkpoint_queue import run_fixture


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
