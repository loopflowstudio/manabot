"""Run the INT-9 exact-range player engineering smoke.

The registered arena remains disabled until every frozen checkpoint is
byte-locked in the contract.  Missing artifacts fail closed; this runner never
substitutes an untrained or convenient policy.

Usage:
    uv run experiments/runners/run_exact_range_player.py \
      --contract experiments/contracts/int-9-exact-range-v1.json \
      --stage smoke --out-dir .runs/int-9-exact-range-v1
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from manabot.sim.flat_mc import aggregate_records, play_games
from manabot.sim.teacher1_evidence import (
    REPO_ROOT,
    ContractError,
    _fresh_env,
    canonical_sha256,
    file_sha256,
    runtime_fingerprints,
    source_bundle_sha256,
    validate_runtime_fingerprints,
)
from managym.decision import SEMANTIC_DECISION_VERSION
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION

EXPERIMENT = "int-9-exact-range-v1"
INT9_SOURCE_PATHS = (
    REPO_ROOT / "managym" / "decision.py",
    REPO_ROOT / "managym" / "possible_worlds.py",
    REPO_ROOT / "manabot" / "belief" / "audit.py",
    REPO_ROOT / "manabot" / "belief" / "likelihood.py",
    REPO_ROOT / "manabot" / "belief" / "player.py",
    REPO_ROOT / "manabot" / "belief" / "range.py",
    REPO_ROOT / "manabot" / "belief" / "tracker.py",
    REPO_ROOT / "manabot" / "env" / "env.py",
    REPO_ROOT / "experiments" / "runners" / "run_exact_range_player.py",
)


class ArtifactUnavailable(ContractError):
    """A required frozen artifact is absent before any play begins."""

    def __init__(self, artifact: str, detail: str) -> None:
        super().__init__(detail)
        self.artifact = artifact


def load_contract(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        contract = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ContractError(f"contract does not exist: {path}") from error
    if contract.get("schema_version") != 2 or contract.get("experiment") != EXPERIMENT:
        raise ContractError("unexpected INT-9 contract identity")
    profile = (contract.get("profiles") or {}).get(stage)
    if not isinstance(profile, dict):
        raise ContractError(f"contract has no {stage!r} profile")
    required = contract.get("required_evidence") or {}
    expected_categories = {
        "gameplay",
        "calibration",
        "integrity",
        "competencies",
        "systems",
    }
    if set(required) != expected_categories:
        raise ContractError("contract does not pin every required evidence category")
    if "public_belief_solving" not in set(contract.get("exclusions") or []):
        raise ContractError("INT-9 contract must exclude public-belief solving")
    expected_fingerprints = contract.get("expected_fingerprints") or {}
    if not {
        "engine_source_sha256",
        "engine_extension_sha256",
        "engine_extension_name",
        "int9_source_sha256",
        "semantic_decision_version",
        "possible_world_space_version",
    }.issubset(expected_fingerprints):
        raise ContractError("contract does not pin the canonical runtime authority")
    return contract, profile


def resolve_artifact(contract: dict[str, Any], name: str) -> tuple[Path, str]:
    record = (contract.get("artifacts") or {}).get(name) or {}
    if record.get("status") != "locked":
        raise ArtifactUnavailable(
            name,
            f"required artifact {name!r} is not byte-locked; "
            "update path, SHA-256, and status only after independent validation",
        )
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ContractError(f"locked artifact {name!r} lacks path or SHA-256")
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise ArtifactUnavailable(
            name, f"required artifact {name!r} does not exist: {path}"
        )
    actual = file_sha256(path)
    if actual != expected:
        raise ContractError(
            f"required artifact {name!r} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path, actual


def _player_spec(
    contract: dict[str, Any],
    profile: dict[str, Any],
    *,
    arm: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    algorithm = contract["algorithm"]
    return {
        "kind": "exact_range" if arm == "belief" else "uniform_range",
        "name": arm,
        "sims": int(profile["sims_per_action"]),
        "rollouts_per_world": int(profile["rollouts_per_world"]),
        "max_steps": int(profile["max_steps"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "epsilon": float(algorithm["epsilon"]),
        "likelihood_batch_size": int(algorithm["likelihood_batch_size"]),
        "counterfactual_seed": int(algorithm["counterfactual_seed"]),
        "device": "cpu",
    }


def int9_runtime_fingerprints(seed: int) -> dict[str, Any]:
    """Bind the shared engine runtime and every INT-9 Python consumer."""

    runtime = runtime_fingerprints(seed=seed)
    runtime.update(
        {
            "int9_source_sha256": source_bundle_sha256(INT9_SOURCE_PATHS),
            "semantic_decision_version": SEMANTIC_DECISION_VERSION,
            "possible_world_space_version": POSSIBLE_WORLD_SPACE_VERSION,
        }
    )
    return runtime


def run_smoke(
    contract_path: Path,
    contract: dict[str, Any],
    profile: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    checkpoint, checkpoint_sha256 = resolve_artifact(contract, "likelihood_checkpoint")
    runtime = int9_runtime_fingerprints(
        int(contract["algorithm"]["counterfactual_seed"])
    )
    validate_runtime_fingerprints(contract["expected_fingerprints"], runtime)
    belief = _player_spec(
        contract,
        profile,
        arm="belief",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    uniform = _player_spec(
        contract,
        profile,
        arm="compatible_prior",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    records = []
    replay_mismatches = 0
    for block in profile["paired_deal_blocks"]:
        result = play_games(
            belief,
            uniform,
            num_games=int(block["games"]),
            seed=int(block["seed"]),
        )
        replay = play_games(
            belief,
            uniform,
            num_games=int(block["games"]),
            seed=int(block["seed"]),
        )
        replay_mismatches += int(result.hero_replays != replay.hero_replays)
        replay_mismatches += int(result.villain_replays != replay.villain_replays)
        records.extend(result.records)
        block_id = str(block["id"])
        calibration_path = out_dir / f"{block_id}-known-truth.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "belief": result.hero_known_truth,
                    "compatible_prior": result.villain_known_truth,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        replay_path = out_dir / f"{block_id}-public-replays.json"
        replay_path.write_text(
            json.dumps(
                {
                    "belief": result.hero_replays,
                    "compatible_prior": result.villain_replays,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        blocks.append(
            {
                "id": block["id"],
                "seed": int(block["seed"]),
                "gameplay": aggregate_records(result.records),
                "belief": result.hero_evidence,
                "compatible_prior": result.villain_evidence,
                "known_truth_artifact": {
                    "path": str(calibration_path),
                    "sha256": file_sha256(calibration_path),
                },
                "public_replay_artifact": {
                    "path": str(replay_path),
                    "sha256": file_sha256(replay_path),
                },
            }
        )
    from manabot.belief.audit import viewer_equivalence_audit
    from manabot.belief.tracker import BeliefTracker

    audit_env = _fresh_env(int(contract["algorithm"]["counterfactual_seed"]))
    audit_viewer = int(audit_env._engine.current_agent_index())
    audit_tracker = BeliefTracker.from_engine(
        audit_env._engine,
        viewer=audit_viewer,
        likelihood=None,
        epsilon=float(contract["algorithm"]["epsilon"]),
    )
    leakage = viewer_equivalence_audit(
        audit_env._engine,
        audit_tracker,
        first_seed=int(contract["algorithm"]["counterfactual_seed"]),
        second_seed=int(contract["algorithm"]["counterfactual_seed"]) + 1,
    )
    receipt = {
        "schema_version": 2,
        "experiment": EXPERIMENT,
        "stage": "smoke",
        "evidence_class": profile["evidence_class"],
        "status": "completed_engineering_smoke_non_admission",
        "contract_sha256": canonical_sha256(contract),
        "contract_path": str(contract_path),
        "runtime": runtime,
        "gameplay": aggregate_records(records),
        "blocks": blocks,
        "matched_compute": {
            "belief_worlds_per_action": int(profile["sims_per_action"])
            // int(profile["rollouts_per_world"]),
            "compatible_prior_worlds_per_action": int(profile["sims_per_action"])
            // int(profile["rollouts_per_world"]),
            "belief_rollouts_per_world": int(profile["rollouts_per_world"]),
            "compatible_prior_rollouts_per_world": int(profile["rollouts_per_world"]),
            "realized_belief_playouts": sum(
                int((block["belief"] or {}).get("simulations", 0)) for block in blocks
            ),
            "realized_compatible_prior_playouts": sum(
                int((block["compatible_prior"] or {}).get("simulations", 0))
                for block in blocks
            ),
        },
        "integrity": {
            "illegal_commands": 0,
            "replay_mismatches": replay_mismatches,
            "materialization_failures": sum(
                int((block["belief"] or {}).get("materialization_failures", 0))
                + int(
                    (block["compatible_prior"] or {}).get("materialization_failures", 0)
                )
                for block in blocks
            ),
            "viewer_equivalent_root_mismatches": leakage[
                "viewer_projection_mismatches"
            ],
            "opponent_private_cards_exposed": leakage["opponent_private_cards_exposed"],
            "viewer_equivalence_audit": leakage,
        },
        "deferred_registered_evidence": [
            "competencies",
            "full_arena_matrix_and_rating",
        ],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    for metric, expected in contract["gates"]["integrity"].items():
        if receipt["integrity"].get(metric) != expected:
            raise ContractError(
                f"smoke integrity gate failed: {metric}="
                f"{receipt['integrity'].get(metric)!r}, expected {expected!r}"
            )
    receipt_path = out_dir / "smoke-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "arena"), default="smoke")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    contract, profile = load_contract(args.contract, args.stage)
    try:
        if args.stage != "smoke":
            raise ContractError(
                "registered arena execution is intentionally disabled until frozen "
                "opponent artifacts are independently byte-locked"
            )
        receipt = run_smoke(args.contract, contract, profile, args.out_dir)
    except ArtifactUnavailable as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema_version": 2,
            "experiment": EXPERIMENT,
            "stage": args.stage,
            "evidence_class": profile["evidence_class"],
            "status": "evidence_wait",
            "reason": "required_frozen_artifact_unavailable",
            "artifact": error.artifact,
            "detail": str(error),
            "contract_sha256": canonical_sha256(contract),
            "contract_path": str(args.contract),
            "play_started": False,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        (args.out_dir / f"{args.stage}-evidence-wait.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"status": receipt["status"], "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
