"""Freeze and verify an immutable belief-conditioned conditional snapshot.

Mirrors ``run_teacher0_partial_snapshot.py`` exactly (same SnapshotError,
atomic-publish, fail-closed-verify, identity re-derivation pattern) and bumps
the snapshot manifest to ``schema_version: 2`` with a ``condition_schema``
block and a belief-conditioned ``CLAIM_BOUNDARY``.

The snapshot freezes a contiguous prefix of conditional shards (produced by
``manabot.sim.conditional_distill``) into an immutable, digest-bound directory.
``verify_conditional_snapshot`` refuses to consume the snapshot unless every
byte, identity, condition schema, claim boundary, and the exact trainer source
still match.

This is plumbing + measurement integrity evidence for INT-14. It is NOT a
strength claim: ``CLAIM_BOUNDARY.strength_claim`` is False, the condition
source is a synthetic uniform-determinization toy, and no learned belief head
is added.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from experiments.runners.run_teacher0_partial_snapshot import (
    REPO_ROOT,
    SnapshotError,
    _atomic_json,
    _file_sha256,
    _identity,
    _json_sha256,
    _load_json,
    _snapshot_identity,
)
from manabot.sim.conditional_distill import (
    CONDITION_LABEL_FORMAT,
    conditional_strategy_shape_digest,
)
from manabot.sim.distill import _git_commit


def _trainer_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    return _identity(
        root,
        TRAINER_SOURCE_FILES,
        commit=_git_commit(),
        bind_commit=False,
    )


TRAINER_SOURCE_FILES = (
    "experiments/runners/run_belief_conditioned_snapshot.py",
    "experiments/runners/run_belief_conditioned_ablation.py",
    "experiments/runners/run_search_supervised.py",
    "manabot/sim/search_supervised.py",
    "manabot/sim/conditional_distill.py",
    "manabot/sim/distill.py",
    "manabot/model/agent.py",
)
DATASET_SOURCE_FILES = (
    "manabot/sim/conditional_distill.py",
    "manabot/sim/distill.py",
)

CLAIM_BOUNDARY = {
    "claim": "belief_conditioned_plumbing_ablation",
    "learned_belief_head": False,
    "pbs_range_net": False,
    "per_hand_value_vector": False,
    "condition_source": "synthetic_uniform_determinization_toy",
    "strength_claim": False,
    "policy_target_kind": "score_softmax_not_mcts_visits",
    "value_target_kind": "terminal_outcome",
    "teacher_algorithm": "flat_determinized_monte_carlo",
}

CONDITION_SCHEMA = {
    "condition_label_format": CONDITION_LABEL_FORMAT,
    "condition_roles": ["true", "has", "lacks", "q", "not_q"],
    "condition_strategy_result_shape_digest": conditional_strategy_shape_digest(),
    "condition_source": "synthetic_uniform_determinization_toy",
}


def _dataset_identity(
    source_root: Path,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    run_contract = dataset_manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise SnapshotError("dataset manifest is missing run_contract")
    source = _identity(
        source_root,
        DATASET_SOURCE_FILES,
        commit=run_contract.get("source_commit"),
    )
    source.update(
        run_contract=run_contract,
        run_contract_sha256=_json_sha256(run_contract),
        run_fingerprint=dataset_manifest.get("run_fingerprint"),
    )
    without_identity = {
        key: value for key, value in source.items() if key != "identity_sha256"
    }
    source["identity_sha256"] = _json_sha256(without_identity)
    return source


def _select_prefix(
    dataset_manifest: dict[str, Any], shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise SnapshotError("shard_count must be positive")
    if dataset_manifest.get("schema_version") != 2:
        raise SnapshotError(
            "conditional snapshots require dataset manifest schema_version 2"
        )
    run_fingerprint = dataset_manifest.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or len(run_fingerprint) != 64:
        raise SnapshotError("dataset manifest has no valid run fingerprint")
    by_index: dict[int, dict[str, Any]] = {}
    for item in dataset_manifest.get("shards") or []:
        index = item.get("shard_index")
        if not isinstance(index, int) or index in by_index:
            raise SnapshotError(
                "dataset manifest has invalid or duplicate shard indexes"
            )
        by_index[index] = item
    missing = [index for index in range(shard_count) if index not in by_index]
    if missing:
        raise SnapshotError(
            f"declared prefix is not yet durable; missing shards {missing}"
        )
    return [by_index[index] for index in range(shard_count)]


def _check_condition_provenance(provenance: dict[str, Any]) -> None:
    if provenance.get("condition_label_format") != CONDITION_LABEL_FORMAT:
        raise SnapshotError("shard condition_label_format mismatch")
    if provenance.get("condition_schema_digest") != conditional_strategy_shape_digest():
        raise SnapshotError("shard condition_schema_digest mismatch")
    if provenance.get("policy_target_kind") != "score_softmax":
        raise SnapshotError("shard policy_target_kind mismatch")
    if provenance.get("value_target_kind") != "terminal_outcome":
        raise SnapshotError("shard value_target_kind mismatch")


def freeze_conditional_snapshot(
    source_dataset_dir: Path,
    snapshot_dir: Path,
    *,
    shard_count: int,
    trainer_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Atomically copy a contiguous conditional-shard prefix into a snapshot."""

    source_dataset_dir = source_dataset_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    if snapshot_dir.exists():
        raise SnapshotError(f"snapshot destination already exists: {snapshot_dir}")
    dataset_manifest_path = source_dataset_dir / "manifest.json"
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    dataset_manifest = _load_json(dataset_manifest_path)
    prefix = _select_prefix(dataset_manifest, shard_count)
    run_fingerprint = str(dataset_manifest["run_fingerprint"])
    source_root = source_dataset_dir.parent
    dataset_source = _dataset_identity(source_root, dataset_manifest)
    trainer_source = _trainer_identity(trainer_root)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{snapshot_dir.name}.tmp-", dir=snapshot_dir.parent)
    )
    records: list[dict[str, Any]] = []
    try:
        for expected_index, summary in enumerate(prefix):
            source_npz = Path(str(summary.get("out_path", ""))).resolve()
            source_json = source_npz.with_suffix(".json")
            if source_npz.parent != source_dataset_dir.resolve():
                raise SnapshotError(
                    f"shard {expected_index} escapes the source dataset"
                )
            sidecar = _load_json(source_json)
            if sidecar != summary:
                raise SnapshotError(
                    f"shard {expected_index} sidecar differs from the durable manifest"
                )
            provenance = sidecar.get("provenance") or {}
            if (
                sidecar.get("shard_index") != expected_index
                or sidecar.get("run_fingerprint") != run_fingerprint
                or provenance.get("dataset_run_fingerprint") != run_fingerprint
            ):
                raise SnapshotError(f"shard {expected_index} provenance mismatch")
            _check_condition_provenance(provenance)
            npz_sha = _file_sha256(source_npz)
            json_sha = _file_sha256(source_json)
            if sidecar.get("sha256") != npz_sha:
                raise SnapshotError(
                    f"shard {expected_index} digest differs from sidecar"
                )
            dest_npz = staging / source_npz.name
            dest_json = staging / source_json.name
            shutil.copyfile(source_npz, dest_npz)
            shutil.copyfile(source_json, dest_json)
            if (
                _file_sha256(source_npz) != npz_sha
                or _file_sha256(source_json) != json_sha
            ):
                raise SnapshotError(
                    f"source shard {expected_index} mutated during copy"
                )
            if _file_sha256(dest_npz) != npz_sha or _file_sha256(dest_json) != json_sha:
                raise SnapshotError(
                    f"snapshot shard {expected_index} copy is not exact"
                )
            records.append(
                {
                    "shard_index": expected_index,
                    "game_offset": provenance.get("game_offset"),
                    "num_games": sidecar.get("num_games"),
                    "decisions": sidecar.get("decisions"),
                    "identity_sha256": _json_sha256(
                        {
                            "shard_index": expected_index,
                            "run_fingerprint": run_fingerprint,
                            "provenance": provenance,
                        }
                    ),
                    "source_npz_path": str(source_npz),
                    "source_json_path": str(source_json),
                    "snapshot_npz_path": dest_npz.name,
                    "snapshot_json_path": dest_json.name,
                    "npz_bytes": dest_npz.stat().st_size,
                    "json_bytes": dest_json.stat().st_size,
                    "npz_sha256": npz_sha,
                    "json_sha256": json_sha,
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "status": "immutable_conditional_prefix_snapshot",
            "created_at": datetime.now(UTC).isoformat(),
            "claim_boundary": CLAIM_BOUNDARY,
            "condition_schema": CONDITION_SCHEMA,
            "parent_dataset": {
                "path": str(source_dataset_dir),
                "manifest_path": str(dataset_manifest_path),
                "manifest_sha256_at_cutoff": hashlib.sha256(
                    dataset_manifest_bytes
                ).hexdigest(),
                "run_fingerprint": run_fingerprint,
                "status_at_cutoff": dataset_manifest.get("status"),
            },
            "cutoff": {
                "kind": "contiguous_complete_conditional_shard_prefix",
                "shard_count": shard_count,
                "first_shard_index": 0,
                "last_shard_index": shard_count - 1,
                "games": int(sum(int(record["num_games"]) for record in records)),
                "decisions": int(sum(int(record["decisions"]) for record in records)),
            },
            "dataset_source": dataset_source,
            "trainer_source": trainer_source,
            "shards": records,
        }
        manifest["snapshot_identity_sha256"] = _snapshot_identity(manifest)
        _atomic_json(staging / "snapshot.json", manifest)
        os_replace(staging, snapshot_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def os_replace(src: Path, dst: Path) -> None:
    import os

    os.replace(src, dst)


def verify_conditional_snapshot(
    snapshot_dir: Path,
    *,
    expected_identity: str,
    trainer_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    manifest = _load_json(snapshot_dir / "snapshot.json")
    actual_identity = _snapshot_identity(manifest)
    if manifest.get("snapshot_identity_sha256") != actual_identity:
        raise SnapshotError("snapshot manifest identity does not match its contents")
    if actual_identity != expected_identity:
        raise SnapshotError(
            "snapshot identity differs from the declared consumer input"
        )
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        raise SnapshotError("snapshot claim boundary changed")
    if manifest.get("condition_schema") != CONDITION_SCHEMA:
        raise SnapshotError("snapshot condition schema changed")
    recorded_trainer = manifest.get("trainer_source") or {}
    current_trainer = _trainer_identity(trainer_root)
    if (
        recorded_trainer.get("files") != current_trainer["files"]
        or recorded_trainer.get("identity_sha256") != current_trainer["identity_sha256"]
    ):
        raise SnapshotError("trainer source identity changed after snapshot freeze")
    parent_fingerprint = manifest.get("parent_dataset", {}).get("run_fingerprint")
    expected_files = {"snapshot.json"}
    records = manifest.get("shards") or []
    cutoff = manifest.get("cutoff") or {}
    if len(records) != cutoff.get("shard_count"):
        raise SnapshotError("snapshot shard count differs from its cutoff")
    for expected_index, record in enumerate(records):
        if record.get("shard_index") != expected_index:
            raise SnapshotError("snapshot is not a contiguous ordered prefix")
        npz_name = str(record.get("snapshot_npz_path", ""))
        json_name = str(record.get("snapshot_json_path", ""))
        if Path(npz_name).name != npz_name or Path(json_name).name != json_name:
            raise SnapshotError(f"snapshot shard {expected_index} has an unsafe path")
        expected_files.update((npz_name, json_name))
        npz_path = snapshot_dir / npz_name
        json_path = snapshot_dir / json_name
        if _file_sha256(npz_path) != record.get("npz_sha256"):
            raise SnapshotError(f"snapshot shard {expected_index} digest mismatch")
        if _file_sha256(json_path) != record.get("json_sha256"):
            raise SnapshotError(f"snapshot sidecar {expected_index} digest mismatch")
        sidecar = _load_json(json_path)
        provenance = sidecar.get("provenance") or {}
        _check_condition_provenance(provenance)
        identity = _json_sha256(
            {
                "shard_index": expected_index,
                "run_fingerprint": parent_fingerprint,
                "provenance": provenance,
            }
        )
        if (
            sidecar.get("shard_index") != expected_index
            or sidecar.get("run_fingerprint") != parent_fingerprint
            or provenance.get("dataset_run_fingerprint") != parent_fingerprint
            or sidecar.get("sha256") != record.get("npz_sha256")
            or record.get("identity_sha256") != identity
        ):
            raise SnapshotError(f"snapshot shard {expected_index} identity mismatch")
    actual_files = {path.name for path in snapshot_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise SnapshotError(
            f"snapshot file set mismatch; missing={missing}, extra={extra}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "verify"), required=True)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-identity")
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.stage == "freeze":
        if args.source_dataset is None:
            parser.error("--stage freeze requires --source-dataset")
        manifest = freeze_conditional_snapshot(
            args.source_dataset,
            args.snapshot_dir,
            shard_count=args.shard_count,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if not args.snapshot_identity:
        parser.error("--stage verify requires --snapshot-identity")
    manifest = verify_conditional_snapshot(
        args.snapshot_dir,
        expected_identity=args.snapshot_identity,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
