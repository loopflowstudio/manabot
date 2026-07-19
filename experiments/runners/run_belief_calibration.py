"""Generate and verify the frozen INT-17 belief-calibration curves.

Usage:
    uv run --extra dev python experiments/runners/run_belief_calibration.py \
      --contract experiments/contracts/int-17-belief-calibration-v1.json \
      --out-dir .runs/int-17-belief-calibration-v1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

from etude import server
from manabot.belief.audit import (
    PairedKnownTruthPoint,
    aggregate_belief_truth,
    score_paired_known_truth,
)
from manabot.belief.likelihood import FrozenPolicyLikelihood
from manabot.belief.tracker import BeliefTracker
from manabot.sim.teacher1_evidence import (
    REPO_ROOT,
    ContractError,
    canonical_sha256,
    file_sha256,
    runtime_fingerprints,
    source_bundle_sha256,
    validate_runtime_fingerprints,
)
import managym
from managym.decision import (
    SEMANTIC_DECISION_VERSION,
    Command,
    DecisionFrame,
    SemanticTransition,
    apply_semantic_command,
)
from managym.possible_worlds import POSSIBLE_WORLD_SPACE_VERSION, PossibleWorldSpace

EXPERIMENT = "int-17-belief-calibration-v1"
SOURCE_PATHS = (
    REPO_ROOT / "etude" / "public_commitment_parity.py",
    REPO_ROOT / "experiments" / "runners" / "run_belief_calibration.py",
    REPO_ROOT / "manabot" / "belief" / "audit.py",
    REPO_ROOT / "manabot" / "belief" / "likelihood.py",
    REPO_ROOT / "manabot" / "belief" / "range.py",
    REPO_ROOT / "manabot" / "belief" / "tracker.py",
    REPO_ROOT / "managym" / "decision.py",
    REPO_ROOT / "managym" / "possible_worlds.py",
)


@dataclass(frozen=True, slots=True)
class Preflight:
    commands: int
    commitments: int
    world_rows: int
    viewer_world_rows: dict[int, int]
    viewer_action_updates: dict[int, int]
    viewer_max_support: dict[int, int]
    identity_stream_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": self.commands,
            "commitments": self.commitments,
            "world_rows": self.world_rows,
            "viewer_world_rows": {
                str(key): value for key, value in self.viewer_world_rows.items()
            },
            "viewer_action_updates": {
                str(key): value for key, value in self.viewer_action_updates.items()
            },
            "viewer_max_support": {
                str(key): value for key, value in self.viewer_max_support.items()
            },
            "identity_stream_sha256": self.identity_stream_sha256,
        }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _clean_source_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ContractError("generation requires a clean tracked worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ContractError(f"contract does not exist: {path}") from error
    if contract.get("schema_version") != 1:
        raise ContractError("unexpected INT-17 contract schema")
    if contract.get("experiment") != EXPERIMENT:
        raise ContractError("unexpected INT-17 experiment identity")
    if contract.get("world") != "w2":
        raise ContractError("INT-17 is frozen to world w2")
    if contract.get("evidence_class") != "selected_trace_calibration_only":
        raise ContractError("INT-17 must remain calibration-only evidence")
    if contract.get("cohort", {}).get("viewers") != [0, 1]:
        raise ContractError("INT-17 v1 must score both fixed viewers")
    if contract.get("algorithm", {}).get("model_fallback") != "forbidden":
        raise ContractError("INT-17 must forbid likelihood fallback")
    if contract.get("caps", {}).get("workers") != 1:
        raise ContractError("INT-17 must use exactly one worker")
    return contract


def _resolve_locked_file(record: Mapping[str, Any], label: str) -> Path:
    if record.get("status") != "locked":
        raise ContractError(f"{label} is not byte-locked")
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ContractError(f"{label} lacks path or SHA-256")
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def _load_inputs(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_path = _resolve_locked_file(contract["trace"], "authority trace")
    provider_path = _resolve_locked_file(
        contract["provider_receipt"], "RUL-11 provider receipt"
    )
    trace = json.loads(trace_path.read_text())
    provider = json.loads(provider_path.read_text())
    _resolve_locked_file(
        contract["likelihood_checkpoint"]["source_manifest"],
        "likelihood source manifest",
    )
    cohort = contract["cohort"]
    if len(trace.get("decisions", ())) != int(cohort["commands"]):
        raise ContractError("authority trace command count drifted")
    if trace.get("terminal_witness", {}).get("revision") != int(cohort["commands"]):
        raise ContractError("authority trace terminal revision drifted")
    expected_summary = {
        "commands": int(cohort["commands"]),
        "commitments": int(cohort["commitments"]),
        "unadmitted_commands": int(cohort["unadmitted_commands"]),
        "tracker_records_per_viewer": int(cohort["commands"]),
        "consumed_commitments": int(cohort["commitments"]),
        "rules_provider_gaps": 0,
        "identity_stream_mismatches": 0,
        "negative_proof_mutations": 0,
    }
    if provider.get("summary") != expected_summary:
        raise ContractError("RUL-11 provider summary drifted")
    if provider.get("identity", {}).get("command_tape_sha256") != contract["trace"].get(
        "command_tape_sha256"
    ):
        raise ContractError("RUL-11 command-tape identity drifted")
    surface_hashes = {
        surface.get("identity_stream_sha256")
        for surface in provider.get("surfaces", {}).values()
    }
    if surface_hashes != {contract["provider_receipt"]["identity_stream_sha256"]}:
        raise ContractError("RUL-11 identity stream drifted")
    return trace, provider


def _fresh_env(seed: int) -> managym.Env:
    env = managym.Env(seed=seed)
    env.reset(
        [
            managym.PlayerConfig("Hero", dict(server.UR_LESSONS_DECK)),
            managym.PlayerConfig("Villain", dict(server.GW_ALLIES_DECK)),
        ]
    )
    return env


def _semantic_command(row: Mapping[str, Any], frame: DecisionFrame) -> Command:
    command = row["command"]
    return Command(
        command_id=str(command["command_id"]),
        expected_revision=frame.revision,
        offer_id=int(command["offer_id"]),
        answers=tuple(command.get("answers", ())),
    )


def _identity_row(
    ordinal: int, actor: int, transition: SemanticTransition
) -> dict[str, Any]:
    receipt = transition.receipt
    return {
        "ordinal": ordinal,
        "actor": actor,
        "command_id": receipt.command_id,
        "before_revision": receipt.before_revision,
        "after_revision": receipt.after_revision,
        "public_commitment": (
            None
            if receipt.public_commitment is None
            else dict(receipt.public_commitment)
        ),
    }


def preflight(contract: Mapping[str, Any], trace: Mapping[str, Any]) -> Preflight:
    cohort = contract["cohort"]
    env = _fresh_env(int(cohort["game_seed"]))
    viewers = [int(viewer) for viewer in cohort["viewers"]]
    world_rows = {viewer: 0 for viewer in viewers}
    action_updates = {viewer: 0 for viewer in viewers}
    max_support = {viewer: 0 for viewer in viewers}
    identity_stream: list[dict[str, Any]] = []
    for ordinal, row in enumerate(trace["decisions"]):
        if ordinal != int(row["ordinal"]):
            raise ContractError("authority decision ordinal drifted")
        if row["state"]["before"] != env.state_digest():
            raise ContractError(f"authority state-before drifted at {ordinal}")
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        actor = int(row["actor"])
        if frame.actor != actor:
            raise ContractError(f"authority actor drifted at {ordinal}")
        root = env.clone_env()
        transition = apply_semantic_command(env, _semantic_command(row, frame))
        if row["state"]["after"] != env.state_digest():
            raise ContractError(f"authority state-after drifted at {ordinal}")
        identity_stream.append(_identity_row(ordinal, actor, transition))
        if transition.receipt.public_commitment is None:
            continue
        viewer = (actor + 1) % 2
        if viewer not in world_rows:
            continue
        support = PossibleWorldSpace.from_engine(root, viewer).support_size
        world_rows[viewer] += support
        action_updates[viewer] += 1
        max_support[viewer] = max(max_support[viewer], support)
    result = Preflight(
        commands=len(identity_stream),
        commitments=sum(action_updates.values()),
        world_rows=sum(world_rows.values()),
        viewer_world_rows=world_rows,
        viewer_action_updates=action_updates,
        viewer_max_support=max_support,
        identity_stream_sha256=_sha256_bytes(_canonical_bytes(identity_stream)),
    )
    expected = contract["preflight"]
    if result.to_dict() != expected:
        raise ContractError(
            "preflight drifted: "
            f"expected {_canonical_bytes(expected).decode()}, "
            f"got {_canonical_bytes(result.to_dict()).decode()}"
        )
    if result.world_rows > int(contract["caps"]["counterfactual_world_rows"]):
        raise ContractError("counterfactual-world-row cap exceeded before inference")
    return result


def int17_runtime_fingerprints(seed: int) -> dict[str, Any]:
    runtime = runtime_fingerprints(seed=seed)
    runtime.update(
        {
            "int17_source_sha256": source_bundle_sha256(SOURCE_PATHS),
            "semantic_decision_version": SEMANTIC_DECISION_VERSION,
            "possible_world_space_version": POSSIBLE_WORLD_SPACE_VERSION,
        }
    )
    return runtime


def _point_row(
    point: PairedKnownTruthPoint,
    *,
    point_kind: str,
    ordinal: int | None,
    actor: int | None,
    transition: SemanticTransition | None,
) -> dict[str, Any]:
    payload = point.to_dict()
    commitment = (
        None
        if transition is None or transition.receipt.public_commitment is None
        else dict(transition.receipt.public_commitment)
    )
    payload.update(
        {
            "point_kind": point_kind,
            "ordinal": ordinal,
            "actor": actor,
            "command_id": (
                None if transition is None else transition.receipt.command_id
            ),
            "before_revision": (
                None if transition is None else transition.receipt.before_revision
            ),
            "after_revision": (
                1 if transition is None else transition.receipt.after_revision
            ),
            "public_commitment": commitment,
            "opponent_commitment": bool(
                transition is not None
                and actor != point.viewer
                and commitment is not None
            ),
        }
    )
    return payload


def _validate_point(point: PairedKnownTruthPoint) -> None:
    for label, belief in (("posterior", point.posterior), ("prior", point.prior)):
        if belief.true_world_index is None:
            raise ContractError(f"actual hand is outside {label} support")
        if belief.true_hand_probability <= 0.0 or not math.isfinite(
            belief.true_hand_probability
        ):
            raise ContractError(f"{label} true-hand mass is zero or non-finite")
        if not math.isfinite(belief.true_hand_log_loss):
            raise ContractError(f"{label} true-hand log loss is non-finite")
    if not math.isfinite(point.log_loss_improvement_nats):
        raise ContractError("paired log-loss improvement is non-finite")
    if not math.isfinite(point.truth_mass_ratio):
        raise ContractError("paired truth-mass ratio is non-finite")


def _arm_summary(points: Iterable[PairedKnownTruthPoint], arm: str) -> dict[str, Any]:
    selected = [getattr(point, arm) for point in points]
    return aggregate_belief_truth(selected, bins=10)


def _paired_summary(points: list[PairedKnownTruthPoint]) -> dict[str, Any]:
    if not points:
        raise ContractError("calibration summary has no points")
    improvements = np.asarray(
        [point.log_loss_improvement_nats for point in points], dtype=np.float64
    )
    ratios = np.asarray([point.truth_mass_ratio for point in points], dtype=np.float64)
    return {
        "points": len(points),
        "posterior": _arm_summary(points, "posterior"),
        "prior": _arm_summary(points, "prior"),
        "mean_log_loss_improvement_nats": float(np.mean(improvements)),
        "median_log_loss_improvement_nats": float(np.median(improvements)),
        "final_log_loss_improvement_nats": float(improvements[-1]),
        "mean_truth_mass_ratio": float(np.mean(ratios)),
        "median_truth_mass_ratio": float(np.median(ratios)),
        "fraction_posterior_mass_above_prior": float(np.mean(ratios > 1.0)),
    }


def summarize(
    points: Mapping[int, list[PairedKnownTruthPoint]],
    rows: list[dict[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    per_viewer: dict[str, Any] = {}
    prediction_passed = True
    for viewer, viewer_points in points.items():
        commitment_points = [
            point
            for point, row in zip(
                viewer_points,
                [row for row in rows if int(row["viewer"]) == viewer],
                strict=True,
            )
            if row["opponent_commitment"]
        ]
        all_summary = _paired_summary(viewer_points)
        commitment_summary = _paired_summary(commitment_points)
        passed = (
            commitment_summary["mean_log_loss_improvement_nats"]
            > float(contract["prediction"]["mean_log_loss_improvement_nats_gt"])
            and commitment_summary["fraction_posterior_mass_above_prior"]
            > float(contract["prediction"]["fraction_posterior_mass_above_prior_gt"])
        )
        prediction_passed = prediction_passed and passed
        per_viewer[str(viewer)] = {
            "all_points": all_summary,
            "opponent_commitment_points": commitment_summary,
            "prediction_passed": passed,
        }
    combined = [point for viewer_points in points.values() for point in viewer_points]
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "evidence_class": contract["evidence_class"],
        "games": 1,
        "viewers": len(points),
        "points": len(combined),
        "per_viewer": per_viewer,
        "combined": _paired_summary(combined),
        "prediction_passed": prediction_passed,
        "interpretation": (
            "selected_trace_directional_prediction_passed"
            if prediction_passed
            else "selected_trace_directional_prediction_refuted"
        ),
        "claim_boundary": (
            "one_seed_selected_trace_calibration_only_no_method_or_strength_claim"
        ),
    }


def run_calibration(
    contract: Mapping[str, Any],
    trace: Mapping[str, Any],
    preflight_result: Preflight,
) -> dict[str, Any]:
    algorithm = contract["algorithm"]
    caps = contract["caps"]
    checkpoint_path = _resolve_locked_file(
        contract["likelihood_checkpoint"], "likelihood checkpoint"
    )
    checkpoint_sha256 = str(contract["likelihood_checkpoint"]["sha256"])
    likelihood = FrozenPolicyLikelihood(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        batch_size=int(algorithm["batch_size"]),
        device=str(algorithm["device"]),
        counterfactual_seed=int(algorithm["counterfactual_seed"]),
    )
    env = _fresh_env(int(contract["cohort"]["game_seed"]))
    trackers = {
        viewer: BeliefTracker.from_engine(
            env,
            viewer=viewer,
            likelihood=likelihood,
            epsilon=float(algorithm["epsilon"]),
        )
        for viewer in contract["cohort"]["viewers"]
    }
    expected_model = f"frozen-policy-likelihood/sha256:{checkpoint_sha256}"
    if {tracker.posterior.model_id for tracker in trackers.values()} != {
        expected_model
    }:
        raise ContractError("tracker did not bind the frozen likelihood model")

    started = time.perf_counter()
    cpu_started = time.process_time()
    points: dict[int, list[PairedKnownTruthPoint]] = {
        viewer: [] for viewer in trackers
    }
    rows: list[dict[str, Any]] = []
    for viewer, tracker in trackers.items():
        point = score_paired_known_truth(env, tracker, game_index=0, step=-1)
        _validate_point(point)
        points[viewer].append(point)
        rows.append(
            _point_row(
                point,
                point_kind="initial",
                ordinal=None,
                actor=None,
                transition=None,
            )
        )

    for ordinal, row in enumerate(trace["decisions"]):
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        actor = int(row["actor"])
        root = env.clone_env()
        root_witness = root.state_digest()
        root_cursor = root.semantic_event_cursor()
        transition = apply_semantic_command(env, _semantic_command(row, frame))
        for tracker in trackers.values():
            tracker.observe(
                env,
                acting=actor,
                transition=transition,
                likelihood_root=root,
            )
            if root.state_digest() != root_witness:
                raise ContractError("likelihood evaluation mutated its source root")
            if root.semantic_event_cursor() != root_cursor:
                raise ContractError("likelihood evaluation mutated its source cursor")
        for viewer, tracker in trackers.items():
            point = score_paired_known_truth(
                env, tracker, game_index=0, step=ordinal
            )
            _validate_point(point)
            points[viewer].append(point)
            rows.append(
                _point_row(
                    point,
                    point_kind="transition",
                    ordinal=ordinal,
                    actor=actor,
                    transition=transition,
                )
            )
        wall_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        peak_rss_bytes = _peak_rss_bytes()
        if wall_seconds > float(caps["wall_seconds"]):
            raise ContractError("wall-time cap exceeded")
        if cpu_seconds > float(caps["core_seconds"]):
            raise ContractError("core-time cap exceeded")
        if peak_rss_bytes > int(caps["peak_rss_bytes"]):
            raise ContractError("peak-RSS cap exceeded")
        if transition.receipt.public_commitment is not None:
            print(
                json.dumps(
                    {
                        "status": "running",
                        "ordinal": ordinal,
                        "commitment": transition.receipt.public_commitment,
                        "wall_seconds": wall_seconds,
                        "peak_rss_bytes": peak_rss_bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_points = int(contract["cohort"]["points_per_viewer"])
    if {len(viewer_points) for viewer_points in points.values()} != {expected_points}:
        raise ContractError("calibration point count drifted")
    if {len(tracker.records) for tracker in trackers.values()} != {
        int(contract["cohort"]["commands"])
    }:
        raise ContractError("tracker transition count drifted")
    action_updates = sum(tracker.stats.action_updates for tracker in trackers.values())
    if action_updates != int(contract["cohort"]["commitments"]):
        raise ContractError("real likelihood action-update count drifted")

    wall_seconds = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    runtime = int17_runtime_fingerprints(int(algorithm["counterfactual_seed"]))
    validate_runtime_fingerprints(contract["expected_runtime"], runtime)
    return {
        "curves": rows,
        "summary": summarize(points, rows, contract),
        "tracker_receipts": {
            str(viewer): tracker.replay_receipt()
            for viewer, tracker in trackers.items()
        },
        "runtime": runtime,
        "resources": {
            "preflight": preflight_result.to_dict(),
            "wall_seconds": wall_seconds,
            "core_seconds": cpu_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "likelihood_seconds": sum(
                tracker.stats.likelihood_seconds for tracker in trackers.values()
            ),
            "workers": 1,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": checkpoint_sha256,
            "model_id": expected_model,
        },
    }


def _curve_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _scientific_payload(result: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        "curves.jsonl": _curve_bytes(result["curves"]),
        "summary.json": _json_bytes(result["summary"]),
        "tracker-receipts.json": _json_bytes(result["tracker_receipts"]),
    }


def write_result(
    out_dir: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ContractError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = _scientific_payload(result)
    generation = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "completed_selected_trace_calibration",
        "contract_path": str(contract_path),
        "contract_sha256": canonical_sha256(contract),
        "preregistration_commit": _clean_source_revision(),
        "runtime": result["runtime"],
        "checkpoint": result["checkpoint"],
        "resources": result["resources"],
        "no_gameplay_generation": True,
        "authority_truth_access": "post_update_audit_only",
    }
    resource_events = [
        {
            "stage": "preflight",
            "payload": result["resources"]["preflight"],
        },
        {
            "stage": "generation",
            "payload": {
                key: value
                for key, value in result["resources"].items()
                if key != "preflight"
            },
        },
    ]
    payloads["generation-receipt.json"] = _json_bytes(generation)
    payloads["resource-ledger.jsonl"] = _curve_bytes(resource_events)
    for name, raw in payloads.items():
        (out_dir / name).write_bytes(raw)
    files = [
        {"path": name, "bytes": len(raw), "sha256": _sha256_bytes(raw)}
        for name, raw in sorted(payloads.items())
    ]
    artifact_sha256 = _sha256_bytes(_canonical_bytes(files))
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "evidence_class": contract["evidence_class"],
        "contract_sha256": canonical_sha256(contract),
        "artifact_sha256": artifact_sha256,
        "files": files,
    }
    manifest_raw = _json_bytes(manifest)
    if sum(len(raw) for raw in payloads.values()) + len(manifest_raw) > int(
        contract["caps"]["artifact_bytes"]
    ):
        raise ContractError("retained artifact-byte cap exceeded")
    (out_dir / "manifest.json").write_bytes(manifest_raw)
    return manifest


def _load_manifest(out_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"retained manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != EXPERIMENT:
        raise ContractError("retained manifest experiment drifted")
    if manifest.get("contract_sha256") != canonical_sha256(contract):
        raise ContractError("retained manifest contract drifted")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContractError("retained manifest has no file list")
    for record in files:
        path = out_dir / str(record["path"])
        if not path.is_file():
            raise ContractError(f"retained result file is missing: {path}")
        raw = path.read_bytes()
        if len(raw) != int(record["bytes"]):
            raise ContractError(f"retained result byte count drifted: {path}")
        if _sha256_bytes(raw) != record["sha256"]:
            raise ContractError(f"retained result digest drifted: {path}")
    if manifest.get("artifact_sha256") != _sha256_bytes(_canonical_bytes(files)):
        raise ContractError("retained artifact digest drifted")
    return manifest


def verify_result(
    out_dir: Path,
    contract: Mapping[str, Any],
    trace: Mapping[str, Any],
    preflight_result: Preflight,
) -> dict[str, Any]:
    before = {
        path.relative_to(out_dir).as_posix(): file_sha256(path)
        for path in out_dir.rglob("*")
        if path.is_file()
    }
    manifest = _load_manifest(out_dir, contract)
    replayed = run_calibration(contract, trace, preflight_result)
    expected_payloads = _scientific_payload(replayed)
    for name, expected in expected_payloads.items():
        actual = (out_dir / name).read_bytes()
        if actual != expected:
            raise ContractError(f"verify-only replay differs at {name}")
    after = {
        path.relative_to(out_dir).as_posix(): file_sha256(path)
        for path in out_dir.rglob("*")
        if path.is_file()
    }
    if before != after:
        raise ContractError("verify-only mutated the retained result")
    return {
        "status": "verified",
        "artifact_sha256": manifest["artifact_sha256"],
        "no_generation": True,
        "retained_files": len(after),
        "retained_tree_unchanged": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only and args.verify_only:
        parser.error("--preflight-only and --verify-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    trace, _ = _load_inputs(contract)
    started = time.perf_counter()
    preflight_result = preflight(contract, trace)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "seconds": time.perf_counter() - started,
                    **preflight_result.to_dict(),
                },
                sort_keys=True,
            )
        )
        return
    if args.verify_only:
        receipt = verify_result(args.out_dir, contract, trace, preflight_result)
        print(json.dumps(receipt, sort_keys=True))
        return
    runtime = int17_runtime_fingerprints(int(contract["algorithm"]["counterfactual_seed"]))
    validate_runtime_fingerprints(contract["expected_runtime"], runtime)
    result = run_calibration(contract, trace, preflight_result)
    manifest = write_result(args.out_dir, args.contract, contract, result)
    print(
        json.dumps(
            {
                "status": "completed_selected_trace_calibration",
                "artifact_sha256": manifest["artifact_sha256"],
                "summary": result["summary"],
                "out_dir": str(args.out_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
