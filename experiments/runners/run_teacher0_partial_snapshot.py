"""Freeze and train from an immutable interim Teacher-0 shard prefix.

This runner never writes to the canonical generator directory. ``freeze``
copies a declared contiguous prefix of complete shard/sidecar pairs into a new
directory and atomically publishes a digest-bound manifest. ``train`` refuses
to consume the snapshot unless every byte, identity, source binding, and the
exact trainer source still match.

The resulting experiment is partial flat-Monte-Carlo score-softmax evidence.
It is not the 3,000-game preregistration, MCTS visit-count distillation, or a
Teacher-1 admission signal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from experiments.runners.run_search_supervised import (
    _dataset_diagnostics,
    _runtime_contract,
    _student_vs_random,
)
from manabot.sim.distill import OBS_KEYS, _git_commit, load_shards, save_bc_checkpoint
from manabot.sim.search_supervised import (
    SCORE_SOFTMAX_TARGET,
    TERMINAL_OUTCOME_TARGET,
    train_search_supervised,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SOURCE_FILES = (
    "experiments/runners/run_teacher0_partial_snapshot.py",
    "experiments/runners/run_search_supervised.py",
    "manabot/sim/search_supervised.py",
    "manabot/sim/distill.py",
    "manabot/model/agent.py",
)
DATASET_SOURCE_FILES = (
    "experiments/runners/run_distill_datagen.py",
    "manabot/sim/distill.py",
)
CLAIM_BOUNDARY = {
    "claim": "partial_interim_teacher0_policy_value_ablation",
    "full_3000_game_preregistration_satisfied": False,
    "mcts_visit_distribution_distillation": False,
    "teacher1_admission_or_unlock": False,
    "teacher_algorithm": "flat_determinized_monte_carlo",
    "policy_target_kind": "score_softmax_not_mcts_visits",
    "value_target_kind": "terminal_outcome",
}


class SnapshotError(RuntimeError):
    """The immutable snapshot or one of its provenance bindings is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SnapshotError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise SnapshotError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise SnapshotError(f"required file is missing: {path}") from error
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(
    root: Path,
    files: tuple[str, ...],
    *,
    commit: str | None,
    bind_commit: bool = True,
) -> dict[str, Any]:
    records = [
        {
            "path": relative,
            "sha256": _file_sha256(root / relative),
        }
        for relative in files
    ]
    payload: dict[str, Any] = {"git_commit_at_capture": commit, "files": records}
    identity_payload: dict[str, Any] = {"files": records}
    if bind_commit:
        identity_payload["git_commit"] = commit
    payload["identity_sha256"] = _json_sha256(identity_payload)
    return payload


def _trainer_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    return _identity(
        root,
        TRAINER_SOURCE_FILES,
        commit=_git_commit(),
        bind_commit=False,
    )


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


def _snapshot_identity(manifest: dict[str, Any]) -> str:
    return _json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "snapshot_identity_sha256"
        }
    )


def _source_root(source_run_dir: Path) -> Path:
    if source_run_dir.parent.name != ".runs":
        raise SnapshotError("source run must be a direct child of a .runs directory")
    return source_run_dir.parent.parent


def _select_prefix(
    dataset_manifest: dict[str, Any], shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise SnapshotError("shard_count must be positive")
    if dataset_manifest.get("schema_version") != 3:
        raise SnapshotError(
            "partial snapshots require dataset manifest schema_version 3"
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


def freeze_snapshot(
    source_run_dir: Path,
    snapshot_dir: Path,
    *,
    shard_count: int,
    trainer_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Atomically copy one complete contiguous shard prefix into a new snapshot."""

    source_run_dir = source_run_dir.resolve()
    snapshot_dir = snapshot_dir.resolve()
    if snapshot_dir.exists():
        raise SnapshotError(f"snapshot destination already exists: {snapshot_dir}")
    parent_manifest_path = source_run_dir / "manifest.json"
    dataset_dir = source_run_dir / "dataset"
    dataset_manifest_path = dataset_dir / "manifest.json"
    parent_manifest_bytes = parent_manifest_path.read_bytes()
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    parent_manifest = _load_json(parent_manifest_path)
    dataset_manifest = _load_json(dataset_manifest_path)
    if int(parent_manifest.get("config", {}).get("games", 0)) != 3000:
        raise SnapshotError("parent run is not the canonical 3,000-game recovery")
    prefix = _select_prefix(dataset_manifest, shard_count)
    run_fingerprint = str(dataset_manifest["run_fingerprint"])
    dataset_source = _dataset_identity(_source_root(source_run_dir), dataset_manifest)
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
            if source_npz.parent != dataset_dir.resolve():
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
                or provenance.get("policy_target_kind") != SCORE_SOFTMAX_TARGET
                or provenance.get("value_target_kind") != TERMINAL_OUTCOME_TARGET
            ):
                raise SnapshotError(f"shard {expected_index} provenance mismatch")
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
            "schema_version": 1,
            "status": "immutable_prefix_snapshot",
            "created_at": datetime.now(UTC).isoformat(),
            "claim_boundary": CLAIM_BOUNDARY,
            "parent_run": {
                "path": str(source_run_dir),
                "manifest_path": str(parent_manifest_path),
                "manifest_sha256_at_cutoff": hashlib.sha256(
                    parent_manifest_bytes
                ).hexdigest(),
                "dataset_manifest_path": str(dataset_manifest_path),
                "dataset_manifest_sha256_at_cutoff": hashlib.sha256(
                    dataset_manifest_bytes
                ).hexdigest(),
                "run_fingerprint": run_fingerprint,
                "status_at_cutoff": parent_manifest.get("status"),
            },
            "cutoff": {
                "kind": "contiguous_complete_shard_prefix",
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
        os.replace(staging, snapshot_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_snapshot(
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
    recorded_trainer = manifest.get("trainer_source") or {}
    current_trainer = _trainer_identity(trainer_root)
    if (
        recorded_trainer.get("files") != current_trainer["files"]
        or recorded_trainer.get("identity_sha256") != current_trainer["identity_sha256"]
    ):
        raise SnapshotError("trainer source identity changed after snapshot freeze")
    parent_fingerprint = manifest.get("parent_run", {}).get("run_fingerprint")
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


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def _inference_evidence(
    agent: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    *,
    device: str,
    repeats: int = 40,
) -> dict[str, Any]:
    dev = torch.device(device)
    agent.eval()
    row = np.asarray([0], dtype=np.int64)
    batch_count = min(256, len(dataset["action"]))
    batch_indices = np.arange(batch_count, dtype=np.int64)

    def obs(indices: np.ndarray) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(dataset[key][indices], dtype=torch.float32, device=dev)
            for key in OBS_KEYS
        }

    single = obs(row)
    batch = obs(batch_indices)
    for _ in range(5):
        agent(single)
    _sync(dev)
    latencies: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        agent(single)
        _sync(dev)
        latencies.append((time.perf_counter() - started) * 1000.0)
    batch_repeats = max(5, repeats // 4)
    started = time.perf_counter()
    for _ in range(batch_repeats):
        agent(batch)
    _sync(dev)
    elapsed = time.perf_counter() - started
    return {
        "device": device,
        "single_observation_repeats": repeats,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "batch_size": batch_count,
        "batch_repeats": batch_repeats,
        "observations_per_second": batch_count * batch_repeats / elapsed,
    }


def _content_address_checkpoint(path: Path) -> tuple[Path, str]:
    digest = _file_sha256(path)
    destination = path.parent / "sha256" / f"{digest}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _file_sha256(destination) != digest:
        raise SnapshotError("content-addressed checkpoint path contains wrong bytes")
    if destination.exists():
        path.unlink()
    else:
        path.replace(destination)
    return destination, digest


def train_snapshot(
    snapshot_dir: Path,
    out_dir: Path,
    *,
    expected_identity: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    seed: int,
    policy_temperature: float,
    device: str,
    evaluation_games: int,
    wall_cap_minutes: float,
) -> dict[str, Any]:
    if device != "mps" or not torch.backends.mps.is_available():
        raise SnapshotError("the partial experiment requires an available MPS device")
    if wall_cap_minutes <= 0 or not math.isfinite(wall_cap_minutes):
        raise SnapshotError("wall_cap_minutes must be finite and positive")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SnapshotError(f"training output directory is not empty: {out_dir}")
    snapshot = verify_snapshot(snapshot_dir, expected_identity=expected_identity)
    shard_paths = [
        snapshot_dir / item["snapshot_npz_path"] for item in snapshot["shards"]
    ]
    dataset = load_shards([str(path) for path in shard_paths])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + wall_cap_minutes * 60.0
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_fraction": val_fraction,
        "seed": seed,
        "policy_temperature": policy_temperature,
        "device": device,
        "evaluation_games": evaluation_games,
        "wall_cap_minutes": wall_cap_minutes,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "w2-234-teacher0-partial-snapshot-v1",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "snapshot_identity_sha256": expected_identity,
        "snapshot_manifest": snapshot,
        "trainer_source": _trainer_identity(),
        "runtime": _runtime_contract(seed, device),
        "config": config,
        "dataset_diagnostics": _dataset_diagnostics(
            dataset,
            policy_target_kind=SCORE_SOFTMAX_TARGET,
            temperature=policy_temperature,
        ),
        "arms": {},
    }
    manifest_path = out_dir / "manifest.json"
    _atomic_json(manifest_path, result)
    arms = (("policy_only", 0.0), ("policy_value", 1.0))
    try:
        for name, value_weight in arms:
            if time.perf_counter() >= deadline:
                raise TimeoutError("partial experiment reached its declared wall cap")
            arm_started = time.perf_counter()
            agent, obs_space, initial, history = train_search_supervised(
                dataset,
                policy_temperature=policy_temperature,
                policy_target_kind=SCORE_SOFTMAX_TARGET,
                value_target_kind=TERMINAL_OUTCOME_TARGET,
                value_weight=value_weight,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                val_fraction=val_fraction,
                seed=seed,
                device=device,
                deadline_monotonic=deadline,
                log=True,
            )
            temporary_checkpoint = out_dir / f"{name}.pt"
            save_bc_checkpoint(
                agent,
                obs_space,
                temporary_checkpoint,
                extra={
                    "experiment": result["experiment"],
                    "claim_boundary": CLAIM_BOUNDARY,
                    "arm": name,
                    "snapshot_identity_sha256": expected_identity,
                    "trainer_source": result["trainer_source"],
                    "config": config,
                    "value_weight": value_weight,
                },
            )
            checkpoint, checkpoint_sha = _content_address_checkpoint(
                temporary_checkpoint
            )
            gameplay = _student_vs_random(
                checkpoint,
                games=evaluation_games,
                seed=seed + 20_000,
                device=device,
            )
            result["arms"][name] = {
                "policy_target_kind": SCORE_SOFTMAX_TARGET,
                "value_target_kind": TERMINAL_OUTCOME_TARGET,
                "value_weight": value_weight,
                "initial_validation": asdict(initial),
                "history": [asdict(epoch) for epoch in history],
                "final_validation": asdict(history[-1].validation),
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "bytes": checkpoint.stat().st_size,
                },
                "seat_balanced_vs_random": gameplay,
                "inference": _inference_evidence(agent, dataset, device=device),
                "wall_seconds": time.perf_counter() - arm_started,
            }
            _atomic_json(manifest_path, result)
        initial_by_arm = {
            name: arm["initial_validation"] for name, arm in result["arms"].items()
        }
        if initial_by_arm["policy_only"] != initial_by_arm["policy_value"]:
            raise SnapshotError("matched arms did not start from identical validation")
        result["matched_controls"] = {
            "same_snapshot": True,
            "same_policy_target": True,
            "same_terminal_outcome_target": True,
            "same_capacity_initialization_split_optimizer_seed_and_cap": True,
            "identical_initial_validation": True,
            "identical_evaluation_seed": seed + 20_000,
            "only_difference": "value_weight: policy_only=0.0, policy_value=1.0",
        }
        result["status"] = "completed_interim"
    except TimeoutError as error:
        result["status"] = "stopped_wall_cap"
        result["failure"] = str(error)
        raise
    except BaseException as error:
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "verify", "train"), required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-identity")
    parser.add_argument("--shard-count", type=int, default=64)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--policy-temperature", type=float, default=0.05)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--evaluation-games", type=int, default=64)
    parser.add_argument("--wall-cap-minutes", type=float, default=45.0)
    args = parser.parse_args()

    if args.stage == "freeze":
        if args.source_run is None:
            parser.error("--stage freeze requires --source-run")
        manifest = freeze_snapshot(
            args.source_run,
            args.snapshot_dir,
            shard_count=args.shard_count,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if not args.snapshot_identity:
        parser.error("--stage verify/train requires --snapshot-identity")
    if args.stage == "verify":
        manifest = verify_snapshot(
            args.snapshot_dir,
            expected_identity=args.snapshot_identity,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.out_dir is None:
        parser.error("--stage train requires --out-dir")
    result = train_snapshot(
        args.snapshot_dir,
        args.out_dir,
        expected_identity=args.snapshot_identity,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        policy_temperature=args.policy_temperature,
        device=args.device,
        evaluation_games=args.evaluation_games,
        wall_cap_minutes=args.wall_cap_minutes,
    )
    print(f"{result['status']}: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
