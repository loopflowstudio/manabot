#!/usr/bin/env python3
"""Verify and envelope the first frozen INT-6 arena rating run.

Generation is intentionally performed by the authenticated historical INT-6
runner. This current-source runner derives only closed-set R4 evidence from an
already verified anchor/challenge pair; it never generates or replays games.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_VERIFIER_PATH = Path(__file__).resolve()
EXPERIMENT = "int-18-first-world-pinned-arena-v1"
FROZEN_EXECUTION_COMMIT = "76d0834797316c3b6e153ed10e5fadd146a8980a"
FROZEN_CONTRACT = REPO_ROOT / "experiments/contracts/int-6-skill-arena-v1.json"
FROZEN_CANDIDATE = REPO_ROOT / "experiments/candidates/int-6-dpuct-32-w4-v1.json"
INT9_CONTRACT = REPO_ROOT / "experiments/contracts/int-9-exact-range-v1.json"
LIKELIHOOD_CANDIDATE = REPO_ROOT / (
    "experiments/data/int-7-value-target-comparison-v1/sha256/"
    "3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/"
    "result/checkpoints/visit_teacher_root-seed-197.pt"
)
FROZEN_CONTRACT_SHA256 = (
    "fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71"
)
FROZEN_ARENA_SOURCE_SHA256 = (
    "b722c8119ebb31fce137231e1434a379059cfd782c734d34b506db0e5ceefe76"
)
FROZEN_DPUCT_SOURCE_SHA256 = (
    "7236414edec8be6d1013cf14d87098a501cd7e504964a27a8dcc58e60972d7fe"
)
FROZEN_EXTENSION_SHA256 = (
    "18d04fe651eddf958da9ebbe0024fca762ffee91dd17c08c6b17f41d1b065504"
)
FROZEN_CANDIDATE_IDENTITY_SHA256 = (
    "6b1eb7855864bd81cfe1995fa98ab104b2c2e18b721ff37989df7873183d3904"
)
LIKELIHOOD_CANDIDATE_SHA256 = (
    "067947696d13f6993ad4b3fb8ec8cdbebd4ad1df9cf6de2a16235c7b384b6c23"
)
PLAYER_IDS = (
    "random-v1",
    "scripted-greedy-v1",
    "flat-mc-4-v1",
    "flat-mc-16-v1",
    "flat-mc-64-v1",
    "determinized-puct-32-w4-v1",
)
EXPECTED_GAMES = {"smoke": 60, "production": 720}
EXPECTED_DEAL_BLOCKS = {"smoke": 2, "production": 24}
EXPECTED_BOOTSTRAP_REPLICATES = {"smoke": 100, "production": 2000}
DPUCT_SOURCE_PATHS = (
    "manabot/sim/flat_mc.py",
    "manabot/sim/mcts.py",
    "manabot/sim/search_branch.py",
    "managym/src/python/bindings.rs",
)


class Int18Error(RuntimeError):
    """The frozen run or derived R4 envelope failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_verifier_receipt() -> dict[str, Any]:
    return {
        "path": CURRENT_VERIFIER_PATH.relative_to(REPO_ROOT).as_posix(),
        "file_sha256": file_sha256(CURRENT_VERIFIER_PATH),
        "role": "current-int18-envelope-derivation-and-verification",
        "boundary": "post-generation-envelope-only",
        "verification_mode": "read-only-no-replay-no-generation",
        "generation_authority": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise Int18Error(f"invalid or missing JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise Int18Error(f"JSON artifact is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _committed_bytes(relative: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{FROZEN_EXECUTION_COMMIT}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise Int18Error(f"frozen source is unavailable: {relative}") from error


def _committed_bundle(paths: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    receipts: dict[str, str] = {}
    for relative in sorted(paths):
        data = _committed_bytes(relative)
        name = relative.encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        receipts[relative] = hashlib.sha256(data).hexdigest()
    return digest.hexdigest(), receipts


def _identity_receipt() -> dict[str, Any]:
    contract = load_json(FROZEN_CONTRACT)
    candidate = load_json(FROZEN_CANDIDATE)
    contract_sha = file_sha256(FROZEN_CONTRACT)
    if contract_sha != FROZEN_CONTRACT_SHA256:
        raise Int18Error("frozen INT-6 contract file changed")
    if _committed_bytes(FROZEN_CONTRACT.relative_to(REPO_ROOT).as_posix()) != (
        FROZEN_CONTRACT.read_bytes()
    ):
        raise Int18Error("frozen INT-6 contract is not present at the execution commit")
    if _committed_bytes(FROZEN_CANDIDATE.relative_to(REPO_ROOT).as_posix()) != (
        FROZEN_CANDIDATE.read_bytes()
    ):
        raise Int18Error("frozen dPUCT registration is not present at the execution commit")
    arena_sha, arena_files = _committed_bundle(tuple(contract["source_paths"]))
    if arena_sha != FROZEN_ARENA_SOURCE_SHA256:
        raise Int18Error("frozen arena source closure does not authenticate")
    dpuct_sha, dpuct_files = _committed_bundle(DPUCT_SOURCE_PATHS)
    if dpuct_sha != FROZEN_DPUCT_SOURCE_SHA256:
        raise Int18Error("frozen dPUCT source closure does not authenticate")
    if canonical_sha256(candidate) != FROZEN_CANDIDATE_IDENTITY_SHA256:
        raise Int18Error("frozen dPUCT registration identity changed")
    anchor_identities = {
        row["player_id"]: canonical_sha256(row) for row in contract["anchors"]
    }
    return {
        "execution_commit": FROZEN_EXECUTION_COMMIT,
        "contract": {
            "path": FROZEN_CONTRACT.relative_to(REPO_ROOT).as_posix(),
            "file_sha256": contract_sha,
            "arena_key": contract["key"],
            "arena_source_sha256": arena_sha,
            "arena_source_files": arena_files,
            "engine_extension_sha256": contract["runtime"][
                "engine_extension_sha256"
            ],
        },
        "anchors": anchor_identities,
        "candidate": {
            "path": FROZEN_CANDIDATE.relative_to(REPO_ROOT).as_posix(),
            "identity_sha256": canonical_sha256(candidate),
            "source_sha256": dpuct_sha,
            "source_files": dpuct_files,
        },
    }


def _verify_historical_manifest(
    directory: Path, *, kind: str, stage: str
) -> dict[str, Any]:
    manifest = load_json(directory / "manifest.json")
    if manifest.get("kind") != kind or manifest.get("profile") != stage:
        raise Int18Error(f"historical {kind} identity mismatch")
    if manifest.get("contract_sha256") != FROZEN_CONTRACT_SHA256:
        raise Int18Error(f"historical {kind} contract mismatch")
    runtime = manifest.get("runtime") or {}
    if runtime.get("engine_extension_sha256") != FROZEN_EXTENSION_SHA256:
        raise Int18Error(f"historical {kind} extension mismatch")
    unsigned = dict(manifest)
    actual_manifest_sha = unsigned.pop("manifest_sha256", None)
    if actual_manifest_sha != canonical_sha256(unsigned):
        raise Int18Error(f"historical {kind} manifest digest mismatch")
    for receipt in manifest.get("artifacts", {}).values():
        path = directory / receipt["path"]
        if not path.is_file() or file_sha256(path) != receipt["sha256"]:
            raise Int18Error(f"historical {kind} artifact mismatch: {path}")
    for receipt in manifest.get("traces", []):
        path = directory / receipt["path"]
        if not path.is_file() or file_sha256(path) != receipt["sha256"]:
            raise Int18Error(f"historical {kind} trace mismatch: {path}")
    return manifest


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise Int18Error("match rows are not JSON objects")
    return rows


def _verify_rows(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    if len(rows) != EXPECTED_GAMES[stage]:
        raise Int18Error("match game count does not match the frozen schedule")
    cells = {str(row["cell_id"]) for row in rows}
    expected_cells = {
        "__".join(sorted(pair)) for pair in combinations(PLAYER_IDS, 2)
    }
    if cells != expected_cells:
        raise Int18Error("match cells are not the complete frozen cohort")
    expected_blocks = set(range(EXPECTED_DEAL_BLOCKS[stage]))
    if {int(row["deal_block"]) for row in rows} != expected_blocks:
        raise Int18Error("match deal blocks do not match the frozen schedule")
    legs = {
        (str(row["cell_id"]), int(row["deal_block"]), int(row["leg"]))
        for row in rows
    }
    expected_legs = {
        (cell, block, leg)
        for cell in expected_cells
        for block in expected_blocks
        for leg in (0, 1)
    }
    if legs != expected_legs:
        raise Int18Error("match rows omit or duplicate a paired seat leg")
    failures = []
    decisions = 0
    for row in rows:
        decisions += int(row["decisions"])
        if not row.get("replay_passed") or any(
            int(value) for value in row.get("integrity", {}).values()
        ):
            failures.append(
                {
                    "cell_id": row["cell_id"],
                    "deal_block": row["deal_block"],
                    "leg": row["leg"],
                }
            )
    if failures:
        raise Int18Error("match rows contain replay or integrity failures")
    return {
        "games": len(rows),
        "cells": len(cells),
        "deal_blocks": len(expected_blocks),
        "seat_legs": len(legs),
        "decisions": decisions,
        "integrity_failures": 0,
    }


def _connectivity(matrix: dict[str, Any]) -> dict[str, Any]:
    expected_edges = {tuple(sorted(pair)) for pair in combinations(PLAYER_IDS, 2)}
    observed_edges = {
        tuple(sorted((str(cell["player_a"]), str(cell["player_b"]))))
        for cell in matrix.values()
    }
    adjacency = {player: set() for player in PLAYER_IDS}
    for left, right in observed_edges:
        if left not in adjacency or right not in adjacency:
            raise Int18Error("payoff matrix contains a foreign player")
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = []
    remaining = set(PLAYER_IDS)
    while remaining:
        root = min(remaining)
        component = set()
        frontier = [root]
        while frontier:
            player = frontier.pop()
            if player in component:
                continue
            component.add(player)
            frontier.extend(adjacency[player] - component)
        components.append(sorted(component))
        remaining -= component
    missing = sorted("__".join(edge) for edge in expected_edges - observed_edges)
    unexpected = sorted("__".join(edge) for edge in observed_edges - expected_edges)
    payload = {
        "players": list(PLAYER_IDS),
        "nodes": len(PLAYER_IDS),
        "observed_edges": len(observed_edges),
        "expected_edges": len(expected_edges),
        "complete_graph": observed_edges == expected_edges,
        "components": components,
        "component_count": len(components),
        "random_anchor_reaches_all": len(components) == 1
        and "random-v1" in components[0],
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "adjacency": {
            player: sorted(neighbors) for player, neighbors in adjacency.items()
        },
    }
    if not (
        payload["complete_graph"]
        and payload["component_count"] == 1
        and payload["random_anchor_reaches_all"]
        and not missing
        and not unexpected
    ):
        raise Int18Error("arena population is not the complete connected cohort")
    return payload


def _paired_uncertainty(
    rating: dict[str, Any], matrix: dict[str, Any], stage: str
) -> dict[str, Any]:
    bootstrap = rating.get("bootstrap") or {}
    if bootstrap.get("replicates") != EXPECTED_BOOTSTRAP_REPLICATES[stage]:
        raise Int18Error("rating bootstrap replicate count drifted")
    if bootstrap.get("failures") != 0:
        raise Int18Error("rating bootstrap contains failed fits")
    if set(bootstrap.get("ratings", {})) != set(PLAYER_IDS):
        raise Int18Error("rating bootstrap omits a player")
    expected_differences = {
        f"{left}__minus__{right}" for left, right in combinations(sorted(PLAYER_IDS), 2)
    }
    if set(bootstrap.get("rating_differences", {})) != expected_differences:
        raise Int18Error("rating bootstrap omits a pairwise difference")
    return {
        "bootstrap_unit": rating["bootstrap_unit"],
        "bootstrap_seed": 63001,
        "replicates": bootstrap["replicates"],
        "failures": bootstrap["failures"],
        "interval_percentiles": [2.5, 50.0, 97.5],
        "ratings": bootstrap["ratings"],
        "rating_differences": bootstrap["rating_differences"],
        "paired_cells": {
            cell_id: cell["paired_blocks"] for cell_id, cell in sorted(matrix.items())
        },
    }


def _exact_range_wait() -> dict[str, Any]:
    contract = load_json(INT9_CONTRACT)
    artifact = contract["artifacts"]["likelihood_checkpoint"]
    candidate_receipt: dict[str, Any]
    if LIKELIHOOD_CANDIDATE.is_file():
        digest = file_sha256(LIKELIHOOD_CANDIDATE)
        if digest != LIKELIHOOD_CANDIDATE_SHA256:
            raise Int18Error("retained likelihood candidate bytes changed")
        candidate_receipt = {
            "path": LIKELIHOOD_CANDIDATE.relative_to(REPO_ROOT).as_posix(),
            "sha256": digest,
            "bytes": LIKELIHOOD_CANDIDATE.stat().st_size,
            "loadable_candidate_only": True,
            "selected_by_registered_contract": False,
        }
    else:
        candidate_receipt = {
            "path": LIKELIHOOD_CANDIDATE.relative_to(REPO_ROOT).as_posix(),
            "available": False,
        }
    return {
        "status": "evidence_wait",
        "play_started": False,
        "reason": "registered_likelihood_artifact_unresolved",
        "int9_contract_path": INT9_CONTRACT.relative_to(REPO_ROOT).as_posix(),
        "int9_contract_sha256": file_sha256(INT9_CONTRACT),
        "registration_status": contract["registration_status"],
        "required_artifact": artifact,
        "retained_unregistered_candidate": candidate_receipt,
        "arena_integration": {
            "int6_player_registration_available": False,
            "int6_semantic_lifecycle_available": False,
            "substitution_performed": False,
            "neutral_likelihood_used": False,
            "authored_belief_used": False,
        },
        "impact": (
            "The frozen-anchor/dPUCT rating run is valid and independent; the "
            "Belief-Aware Play exact-range arena KR remains open."
        ),
    }


def _tree_receipts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "int18-manifest.json"
    ]


def derive(stage: str, out_dir: Path) -> dict[str, Any]:
    anchor_dir = out_dir / "anchor"
    challenge_dir = out_dir / "challenge"
    identity = _identity_receipt()
    current_verifier = _current_verifier_receipt()
    anchor = _verify_historical_manifest(
        anchor_dir, kind="anchor-freeze", stage=stage
    )
    challenge = _verify_historical_manifest(
        challenge_dir, kind="challenge", stage=stage
    )
    if challenge.get("anchor_manifest_sha256") != anchor.get("manifest_sha256"):
        raise Int18Error("challenge is not bound to the supplied anchor artifact")
    if canonical_sha256(challenge.get("candidate")) != (
        FROZEN_CANDIDATE_IDENTITY_SHA256
    ):
        raise Int18Error("challenge candidate registration changed")
    rows = _load_rows(challenge_dir / "matches.jsonl")
    run_summary = _verify_rows(rows, stage)
    rating = load_json(challenge_dir / "rating.json")
    matrix = load_json(challenge_dir / "payoff-matrix.json")
    if len(rating.get("ratings", {})) != len(PLAYER_IDS):
        raise Int18Error("rating result omits a frozen player")
    if len(matrix) != len(tuple(combinations(PLAYER_IDS, 2))):
        raise Int18Error("payoff matrix is incomplete")
    connectivity = _connectivity(matrix)
    uncertainty = _paired_uncertainty(rating, matrix, stage)
    exact_range = _exact_range_wait()
    write_json(out_dir / "connectivity.json", connectivity)
    write_json(out_dir / "paired-deal-uncertainty.json", uncertainty)
    write_json(out_dir / "exact-range-evidence-wait.json", exact_range)
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "stage": stage,
        "evidence_class": (
            "engineering_smoke_non_admission"
            if stage == "smoke"
            else "production_rating"
        ),
        "status": "complete",
        "identity": identity,
        "current_verifier": current_verifier,
        "anchor_manifest_sha256": anchor["manifest_sha256"],
        "challenge_manifest_sha256": challenge["manifest_sha256"],
        "arena_key": challenge["arena_key"],
        "run": run_summary,
        "connectivity_sha256": file_sha256(out_dir / "connectivity.json"),
        "paired_deal_uncertainty_sha256": file_sha256(
            out_dir / "paired-deal-uncertainty.json"
        ),
        "exact_range_evidence_wait_sha256": file_sha256(
            out_dir / "exact-range-evidence-wait.json"
        ),
        "rating_sha256": file_sha256(challenge_dir / "rating.json"),
        "payoff_matrix_sha256": file_sha256(
            challenge_dir / "payoff-matrix.json"
        ),
        "d_puct_rating": rating["ratings"]["determinized-puct-32-w4-v1"],
        "disposition": load_json(challenge_dir / "promotion.json"),
        "development_pairwise_matches_are_admission_evidence": False,
        "belief_aware_play_arena_kr_holds": False,
    }
    write_json(out_dir / "int18-result.json", result)
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "stage": stage,
        "evidence_class": result["evidence_class"],
        "current_verifier": current_verifier,
        "result_sha256": file_sha256(out_dir / "int18-result.json"),
        "files": _tree_receipts(out_dir),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(out_dir / "int18-manifest.json", manifest)
    return verify(stage, out_dir)


def verify(stage: str, out_dir: Path) -> dict[str, Any]:
    manifest = load_json(out_dir / "int18-manifest.json")
    unsigned_manifest = dict(manifest)
    actual_manifest_sha = unsigned_manifest.pop("manifest_sha256", None)
    if actual_manifest_sha != canonical_sha256(unsigned_manifest):
        raise Int18Error("INT-18 envelope manifest digest mismatch")
    current_verifier = _current_verifier_receipt()
    if manifest.get("current_verifier") != current_verifier:
        raise Int18Error("INT-18 current verifier manifest receipt mismatch")
    expected_paths = {row["path"] for row in manifest["files"]}
    actual_paths = {
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "int18-manifest.json"
    }
    if actual_paths != expected_paths:
        raise Int18Error("INT-18 envelope is not a closed artifact set")
    for row in manifest["files"]:
        path = out_dir / row["path"]
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise Int18Error(f"INT-18 file receipt mismatch: {row['path']}")
    result = load_json(out_dir / "int18-result.json")
    if result.get("stage") != stage or result.get("status") != "complete":
        raise Int18Error("INT-18 result stage or status mismatch")
    if manifest.get("result_sha256") != file_sha256(out_dir / "int18-result.json"):
        raise Int18Error("INT-18 result digest mismatch")
    if result.get("current_verifier") != current_verifier:
        raise Int18Error("INT-18 current verifier result receipt mismatch")
    if result.get("exact_range_evidence_wait_sha256") != file_sha256(
        out_dir / "exact-range-evidence-wait.json"
    ):
        raise Int18Error("INT-18 exact-range evidence wait receipt mismatch")
    rows = _load_rows(out_dir / "challenge/matches.jsonl")
    run_summary = _verify_rows(rows, stage)
    matrix = load_json(out_dir / "challenge/payoff-matrix.json")
    connectivity = _connectivity(matrix)
    rating = load_json(out_dir / "challenge/rating.json")
    uncertainty = _paired_uncertainty(rating, matrix, stage)
    if load_json(out_dir / "connectivity.json") != connectivity:
        raise Int18Error("INT-18 connectivity derivation mismatch")
    if load_json(out_dir / "paired-deal-uncertainty.json") != uncertainty:
        raise Int18Error("INT-18 paired uncertainty derivation mismatch")
    exact_range = load_json(out_dir / "exact-range-evidence-wait.json")
    if exact_range.get("status") != "evidence_wait" or exact_range.get(
        "play_started"
    ) is not False:
        raise Int18Error("INT-18 exact-range evidence wait status mismatch")
    _identity_receipt()
    return {
        "verified": True,
        "no_generation": True,
        "no_replay": True,
        "stage": stage,
        "manifest_sha256": actual_manifest_sha,
        "current_verifier_sha256": current_verifier["file_sha256"],
        **run_summary,
        "components": connectivity["component_count"],
        "bootstrap_replicates": uncertainty["replicates"],
        "exact_range_status": "evidence_wait",
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "production"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = (
            verify(args.stage, args.out_dir)
            if args.verify_only
            else derive(args.stage, args.out_dir)
        )
    except (Int18Error, OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"INT-18 failed closed: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
