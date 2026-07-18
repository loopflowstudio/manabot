"""Fail-closed verifier for INT-8's retained INT-4 smoke inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manabot.sim.distill import OBS_KEYS, load_shards
from manabot.sim.flat_mc import load_checkpoint_agent

from .models import canonical_sha256, file_sha256


class RetainedInputError(RuntimeError):
    """An exact retained byte, manifest, or current-loader gate failed."""

    def __init__(self, evidence: dict[str, Any]):
        super().__init__(json.dumps(evidence, sort_keys=True))
        self.evidence = evidence


def _fail(
    boundary: str,
    *,
    expected: Any,
    actual: Any,
    path: Path | None = None,
) -> None:
    evidence = {
        "status": "input_incompatible",
        "boundary": boundary,
        "expected": expected,
        "actual": actual,
    }
    if path is not None:
        evidence["path"] = str(path)
    raise RetainedInputError(evidence)


def _require(
    condition: bool,
    boundary: str,
    *,
    expected: Any,
    actual: Any,
    path: Path | None = None,
) -> None:
    if not condition:
        _fail(boundary, expected=expected, actual=actual, path=path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _fail("json_read", expected="valid JSON", actual=str(error), path=path)


def _write_failure(path: Path | None, error: RetainedInputError) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(error.evidence, indent=2, sort_keys=True) + "\n")


def _verify_payload(manifest: dict[str, Any], root: Path) -> Path:
    payload = root / manifest["payload"]["path"]
    expected_rows = manifest["payload"]["files"]
    expected_paths = [str(row["path"]) for row in expected_rows]
    actual_paths = sorted(
        str(path.relative_to(payload))
        for path in payload.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        actual_paths == expected_paths,
        "payload_closed_file_set",
        expected=expected_paths,
        actual=actual_paths,
        path=payload,
    )
    actual_rows = []
    for expected in expected_rows:
        path = payload / expected["path"]
        _require(
            path.is_file() and not path.is_symlink(),
            "payload_regular_file",
            expected="regular file, not symlink",
            actual={"exists": path.exists(), "symlink": path.is_symlink()},
            path=path,
        )
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        _require(
            actual == expected,
            "payload_leaf_identity",
            expected=expected,
            actual=actual,
            path=path,
        )
        actual_rows.append(actual)
    tree_sha256 = canonical_sha256(sorted(actual_rows, key=lambda row: row["path"]))
    _require(
        tree_sha256 == manifest["payload"]["sha256"],
        "payload_tree_identity",
        expected=manifest["payload"]["sha256"],
        actual=tree_sha256,
        path=payload,
    )
    return payload


def _verify_manifests(manifest: dict[str, Any], root: Path, payload: Path) -> None:
    contract_identity = manifest["contract"]
    contract_path = root / contract_identity["path"]
    actual_contract = {
        "bytes": contract_path.stat().st_size if contract_path.is_file() else None,
        "file_sha256": file_sha256(contract_path) if contract_path.is_file() else None,
    }
    expected_contract = {
        "bytes": contract_identity["bytes"],
        "file_sha256": contract_identity["file_sha256"],
    }
    _require(
        actual_contract == expected_contract,
        "contract_byte_identity",
        expected=expected_contract,
        actual=actual_contract,
        path=contract_path,
    )
    contract_canonical = canonical_sha256(_load_json(contract_path))
    _require(
        contract_canonical == contract_identity["canonical_sha256"],
        "contract_canonical_identity",
        expected=contract_identity["canonical_sha256"],
        actual=contract_canonical,
        path=contract_path,
    )
    required = manifest["required_identities"]
    root_manifest = _load_json(payload / "manifest.json")
    dataset_manifest = _load_json(payload / "dataset/manifest.json")
    training_manifest = _load_json(payload / "training/manifest.json")
    verification = _load_json(payload / "verification.json")
    expected_root = {
        "contract_sha256": contract_identity["canonical_sha256"],
        "profile_sha256": required["profile_sha256"],
        "evidence_class": "engineering_smoke_non_admission",
        "diagnosis": "engineering_smoke_only_no_admission_claim",
    }
    actual_root = {
        "contract_sha256": root_manifest.get("contract_sha256"),
        "profile_sha256": root_manifest.get("profile_sha256"),
        "evidence_class": root_manifest.get("evidence_class"),
        "diagnosis": root_manifest.get("stages", {})
        .get("arena", {})
        .get("result", {})
        .get("diagnosis"),
    }
    _require(
        actual_root == expected_root,
        "root_manifest_identity",
        expected=expected_root,
        actual=actual_root,
        path=payload / "manifest.json",
    )
    shard_rows = dataset_manifest.get("shards", [])
    actual_shards = [row.get("sha256") for row in shard_rows]
    games = sum(int(row.get("games", 0)) for row in shard_rows)
    labels = sum(int(row.get("summary", {}).get("decisions", 0)) for row in shard_rows)
    sources = {
        row.get("summary", {}).get("provenance", {}).get("git_commit")
        for row in shard_rows
    }
    _require(
        len(shard_rows) == 2
        and actual_shards == required["shard_sha256"]
        and games == required["games"]
        and labels == required["labels"]
        and sources == {required["source_commit"]},
        "dataset_manifest_identity",
        expected={
            "shards": required["shard_sha256"],
            "games": required["games"],
            "labels": required["labels"],
            "source": required["source_commit"],
        },
        actual={
            "shards": actual_shards,
            "games": games,
            "labels": labels,
            "sources": sorted(str(source) for source in sources),
        },
        path=payload / "dataset/manifest.json",
    )
    checkpoint_rows = training_manifest.get("checkpoints", {})
    actual_checkpoints = [
        row.get("sha256") for _, row in sorted(checkpoint_rows.items())
    ]
    _require(
        len(checkpoint_rows) == 4
        and sorted(actual_checkpoints) == sorted(required["checkpoint_sha256"]),
        "training_manifest_identity",
        expected=sorted(required["checkpoint_sha256"]),
        actual=sorted(actual_checkpoints),
        path=payload / "training/manifest.json",
    )
    replay = verification.get("trajectory_replay", {})
    mismatch_fields = [
        "frame_mismatches",
        "command_mismatches",
        "outcome_mismatches",
        "search_action_mismatches",
        "search_visit_mismatches",
        "search_q_mismatches",
        "search_value_mismatches",
        "search_world_mismatches",
        "search_metadata_mismatches",
        "missing_sampled_search_roots",
    ]
    actual_replay = {
        "verified": verification.get("verified"),
        "passed": replay.get("passed"),
        "decisions": replay.get("decisions"),
        "private_exposures": replay.get("opponent_private_cards_exposed"),
        "mismatches": sum(int(replay.get(field, -1)) for field in mismatch_fields),
    }
    expected_replay = {
        "verified": True,
        "passed": True,
        "decisions": required["replayed_decisions"],
        "private_exposures": required["private_exposures"],
        "mismatches": required["replay_mismatches"],
    }
    _require(
        actual_replay == expected_replay,
        "replay_receipt_identity",
        expected=expected_replay,
        actual=actual_replay,
        path=payload / "verification.json",
    )


def _verify_loader_source(root: Path) -> dict[str, Any]:
    checked = _load_json(root / "compatibility.json")
    rows = checked["loader_source"]["files"]
    repo_root = Path(__file__).resolve().parents[2]
    actual_rows = []
    for expected in rows:
        path = repo_root / expected["path"]
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
        }
        _require(
            actual == expected,
            "loader_source_identity",
            expected=expected,
            actual=actual,
            path=path,
        )
        actual_rows.append(actual)
    actual_sha = canonical_sha256(actual_rows)
    _require(
        actual_sha == checked["loader_source"]["sha256"],
        "loader_source_bundle_identity",
        expected=checked["loader_source"]["sha256"],
        actual=actual_sha,
    )
    return checked


def _verify_loaders(
    manifest: dict[str, Any], payload: Path, checked: dict[str, Any]
) -> dict[str, Any]:
    shard_paths = sorted((payload / "dataset").glob("shard_*.npz"))
    dataset = load_shards(shard_paths)
    rows = len(dataset["action"])
    finite = all(np.isfinite(dataset[key]).all() for key in OBS_KEYS)
    legal = dataset["actions_valid"].astype(bool)
    action_rows = np.arange(rows)
    chosen_legal = bool(legal[action_rows, dataset["action"].astype(int)].all())
    games = len(np.unique(dataset["game_index"]))
    expected_dataset = {"rows": 507, "games": 4, "finite": True, "legal": True}
    actual_dataset = {
        "rows": rows,
        "games": games,
        "finite": bool(finite),
        "legal": chosen_legal and bool(legal.any(axis=1).all()),
    }
    _require(
        actual_dataset == expected_dataset,
        "current_shard_loader",
        expected=expected_dataset,
        actual=actual_dataset,
    )
    batch = slice(0, min(32, rows))
    tensor_obs = {
        key: torch.as_tensor(dataset[key][batch], dtype=torch.float32)
        for key in OBS_KEYS
    }
    checkpoint_rows = _load_json(payload / "training/manifest.json")["checkpoints"]
    path_by_sha = {
        row["sha256"]: payload / "training" / Path(row["path"]).name
        for row in checkpoint_rows.values()
    }
    evidence = []
    for expected in checked["checkpoints"]:
        checkpoint_path = path_by_sha[expected["sha256"]]
        agent, _ = load_checkpoint_agent(str(checkpoint_path))
        parameter_count = sum(parameter.numel() for parameter in agent.parameters())
        with torch.inference_mode():
            first_logits, first_value = agent(tensor_obs)
            second_logits, second_value = agent(tensor_obs)
            probabilities = torch.softmax(first_logits, dim=-1)
        illegal_mass = float(
            probabilities.masked_select(~tensor_obs["actions_valid"].bool()).sum()
        )
        actual = {
            "arm": expected["arm"],
            "sha256": file_sha256(checkpoint_path),
            "parameter_count": parameter_count,
            "logits_shape": list(first_logits.shape),
            "value_shape": list(first_value.shape),
            "deterministic_repeated_bytes": first_logits.numpy().tobytes()
            == second_logits.numpy().tobytes()
            and first_value.numpy().tobytes() == second_value.numpy().tobytes(),
            "illegal_probability_mass": illegal_mass,
        }
        _require(
            actual == expected,
            "current_checkpoint_loader",
            expected=expected,
            actual=actual,
            path=checkpoint_path,
        )
        _require(
            torch.isfinite(first_logits).all().item()
            and torch.isfinite(first_value).all().item(),
            "current_checkpoint_finite_output",
            expected=True,
            actual=False,
            path=checkpoint_path,
        )
        evidence.append(actual)
    return {
        "dataset": actual_dataset,
        "checkpoints": evidence,
        "conversion_performed": False,
        "rewriting_performed": False,
        "retraining_performed": False,
        "substitution_performed": False,
    }


def verify_retained_input(
    manifest_path: str | Path, *, failure_receipt: str | Path | None = None
) -> dict[str, Any]:
    """Recompute the exact byte/manifests gate before invoking any loader."""

    path = Path(manifest_path).resolve()
    failure_path = Path(failure_receipt) if failure_receipt is not None else None
    try:
        manifest = _load_json(path)
        root = path.parent
        checked = _load_json(root / "compatibility.json")
        _require(
            file_sha256(path) == checked["input_manifest"]["file_sha256"],
            "input_manifest_identity",
            expected=checked["input_manifest"]["file_sha256"],
            actual=file_sha256(path),
            path=path,
        )
        payload = _verify_payload(manifest, root)
        _verify_manifests(manifest, root, payload)
        checked = _verify_loader_source(root)
        loader = _verify_loaders(manifest, payload, checked)
        return {
            "schema_version": 1,
            "status": "compatible",
            "evidence_class": "engineering_smoke_only_no_admission_claim",
            "input_manifest_sha256": file_sha256(path),
            "payload_sha256": manifest["payload"]["sha256"],
            "contract_file_sha256": manifest["contract"]["file_sha256"],
            "contract_canonical_sha256": manifest["contract"]["canonical_sha256"],
            "loader_source_sha256": checked["loader_source"]["sha256"],
            **loader,
        }
    except RetainedInputError as error:
        _write_failure(failure_path, error)
        raise
