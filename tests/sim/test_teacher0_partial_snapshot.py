"""Fail-closed tests for immutable partial Teacher-0 snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.run_teacher0_partial_snapshot import (
    DATASET_SOURCE_FILES,
    TRAINER_SOURCE_FILES,
    SnapshotError,
    _file_sha256,
    freeze_snapshot,
    verify_snapshot,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_run(tmp_path: Path, *, shards: int = 3) -> tuple[Path, Path]:
    root = tmp_path / "source"
    for relative in set(DATASET_SOURCE_FILES) | set(TRAINER_SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n")
    run_dir = root / ".runs" / "recovery"
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    _write_json(
        run_dir / "manifest.json", {"status": "running", "config": {"games": 3000}}
    )
    fingerprint = "f" * 64
    summaries = []
    for index in range(shards):
        npz_path = dataset_dir / f"shard_{index:05d}.npz"
        npz_path.write_bytes(f"npz-{index}".encode())
        provenance = {
            "dataset_run_fingerprint": fingerprint,
            "game_offset": index * 8,
            "git_commit": "a" * 40,
            "num_games": 8,
            "policy_target_kind": "score_softmax",
            "round": 0,
            "seed": 197 + index * 1_000_000,
            "teacher_spec": {"kind": "search", "sims": 256},
            "value_target_kind": "terminal_outcome",
        }
        summary = {
            "decisions": 10 + index,
            "num_games": 8,
            "out_path": str(npz_path),
            "provenance": provenance,
            "run_fingerprint": fingerprint,
            "sha256": _file_sha256(npz_path),
            "shard_index": index,
        }
        _write_json(npz_path.with_suffix(".json"), summary)
        summaries.append(summary)
    _write_json(
        dataset_dir / "manifest.json",
        {
            "schema_version": 3,
            "status": "running",
            "games": 3000,
            "games_per_shard": 8,
            "run_fingerprint": fingerprint,
            "run_contract": {
                "schema_version": 1,
                "games": 3000,
                "games_per_shard": 8,
                "round": 0,
                "seed": 197,
                "sims": 256,
                "source_commit": "a" * 40,
                "teacher_spec": {"kind": "search", "sims": 256},
            },
            "shards": summaries,
        },
    )
    return root, run_dir


def _freeze(tmp_path: Path) -> tuple[Path, Path, dict]:
    root, run_dir = _source_run(tmp_path)
    snapshot_dir = tmp_path / "snapshot"
    manifest = freeze_snapshot(
        run_dir,
        snapshot_dir,
        shard_count=2,
        trainer_root=root,
    )
    return root, snapshot_dir, manifest


def test_freeze_copies_only_declared_complete_prefix_and_verifies(
    tmp_path: Path,
) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)

    verified = verify_snapshot(
        snapshot_dir,
        expected_identity=manifest["snapshot_identity_sha256"],
        trainer_root=root,
    )

    assert verified["cutoff"] == {
        "kind": "contiguous_complete_shard_prefix",
        "shard_count": 2,
        "first_shard_index": 0,
        "last_shard_index": 1,
        "games": 16,
        "decisions": 21,
    }
    assert [item["shard_index"] for item in verified["shards"]] == [0, 1]
    assert not (snapshot_dir / "shard_00002.npz").exists()


def test_freeze_fails_when_declared_prefix_is_not_durable(tmp_path: Path) -> None:
    root, run_dir = _source_run(tmp_path, shards=1)

    with pytest.raises(SnapshotError, match="not yet durable"):
        freeze_snapshot(
            run_dir,
            tmp_path / "snapshot",
            shard_count=2,
            trainer_root=root,
        )


def test_freeze_fails_on_source_digest_mismatch(tmp_path: Path) -> None:
    root, run_dir = _source_run(tmp_path)
    shard = run_dir / "dataset" / "shard_00000.npz"
    shard.write_bytes(b"mutated")

    with pytest.raises(SnapshotError, match="digest differs"):
        freeze_snapshot(
            run_dir,
            tmp_path / "snapshot",
            shard_count=2,
            trainer_root=root,
        )


def test_freeze_fails_on_parent_run_fingerprint_mismatch(tmp_path: Path) -> None:
    root, run_dir = _source_run(tmp_path)
    dataset_manifest_path = run_dir / "dataset" / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    sidecar_path = run_dir / "dataset" / "shard_00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["run_fingerprint"] = "e" * 64
    sidecar["provenance"]["dataset_run_fingerprint"] = "e" * 64
    dataset_manifest["shards"][0] = sidecar
    _write_json(sidecar_path, sidecar)
    _write_json(dataset_manifest_path, dataset_manifest)

    with pytest.raises(SnapshotError, match="provenance mismatch"):
        freeze_snapshot(
            run_dir,
            tmp_path / "snapshot",
            shard_count=2,
            trainer_root=root,
        )


def test_consumer_fails_on_snapshot_byte_mutation(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    (snapshot_dir / "shard_00000.npz").write_bytes(b"mutated")

    with pytest.raises(SnapshotError, match="digest mismatch"):
        verify_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_consumer_fails_on_snapshot_file_set_change(tmp_path: Path, mode: str) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    if mode == "missing":
        (snapshot_dir / "shard_00001.json").unlink()
        match = "required file is missing"
    else:
        (snapshot_dir / "shard_99999.npz").write_bytes(b"extra")
        match = "file set mismatch"

    with pytest.raises(SnapshotError, match=match):
        verify_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_fails_when_trainer_source_changes(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    (root / TRAINER_SOURCE_FILES[0]).write_text("changed\n")

    with pytest.raises(SnapshotError, match="trainer source identity changed"):
        verify_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_requires_exact_declared_snapshot_identity(tmp_path: Path) -> None:
    root, snapshot_dir, _ = _freeze(tmp_path)

    with pytest.raises(SnapshotError, match="declared consumer input"):
        verify_snapshot(
            snapshot_dir,
            expected_identity="0" * 64,
            trainer_root=root,
        )
