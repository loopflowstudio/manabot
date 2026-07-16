"""Run the pre-registered Teacher-1 admission gate without training students.

This runner intentionally does not call the resilient distillation generator.
It evaluates the landed deterministic PUCT reference at three budgets, records
viewer-safe replay evidence, and stops at the teacher gate.  A later serial PR
may invoke distillation only when this manifest says ``completed_pass``.

The checked-in contract is immutable.  Host-specific matched-wall mappings and
the exact frozen Teacher-0 checkpoint live in a separately checked-in control
lock, which must bind to the contract hash before this command will run.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from manabot.sim.flat_mc import aggregate_records, play_games
from manabot.sim.teacher1_evidence import (
    ContractError,
    canonical_sha256,
    evaluate_root_stability,
    file_sha256,
    receipt_dict,
    record_teacher_trajectories,
    replay_teacher_trajectories,
    runtime_fingerprints,
    validate_runtime_fingerprints,
)
from manabot.verify.competency import SCENARIOS, run_scenario_suite


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ContractError(f"required file does not exist: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = _load_json(path)
    if contract.get("schema_version") != 1:
        raise ContractError("Teacher-1 contract schema_version must be 1")
    if contract.get("experiment") != "w2-234-teacher1-admission-v1":
        raise ContractError("unexpected Teacher-1 experiment identity")
    return contract, canonical_sha256(contract)


def _load_control_lock(
    path: Path, *, contract_hash: str, budgets: list[int]
) -> tuple[dict[str, Any], str]:
    lock = _load_json(path)
    if lock.get("schema_version") != 1:
        raise ContractError("control lock schema_version must be 1")
    if lock.get("contract_sha256") != contract_hash:
        raise ContractError("control lock does not bind to this contract")
    checkpoint = lock.get("checkpoint_control") or {}
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    expected_checkpoint_hash = checkpoint.get("sha256")
    if checkpoint.get("arm") != "policy_value":
        raise ContractError(
            "control checkpoint must be the preselected policy_value arm"
        )
    if not expected_checkpoint_hash or not checkpoint_path.is_file():
        raise ContractError("control lock must name an existing frozen checkpoint")
    if file_sha256(checkpoint_path) != expected_checkpoint_hash:
        raise ContractError("frozen checkpoint SHA-256 does not match the lock")
    source_manifest = lock.get("recovery_manifest") or {}
    source_manifest_path = Path(str(source_manifest.get("path", "")))
    if not source_manifest_path.is_file():
        raise ContractError("control lock must name the completed recovery manifest")
    if file_sha256(source_manifest_path) != source_manifest.get("sha256"):
        raise ContractError("recovery manifest SHA-256 does not match the lock")
    source_status = _load_json(source_manifest_path).get("status")
    if source_status not in {"completed_pass", "completed_diagnostic_failure"}:
        raise ContractError(
            f"recovery manifest is not terminal and usable: {source_status!r}"
        )
    calibration = lock.get("latency_calibration") or {}
    calibration_path = Path(str(calibration.get("path", "")))
    if not calibration_path.is_file():
        raise ContractError("control lock must name a latency calibration artifact")
    if file_sha256(calibration_path) != calibration.get("sha256"):
        raise ContractError("latency calibration SHA-256 does not match the lock")
    calibration_payload = _load_json(calibration_path)
    if calibration_payload.get("host") != lock.get("host"):
        raise ContractError("latency calibration and control lock hosts differ")
    matched = lock.get("flat_sims_per_action_by_teacher_budget") or {}
    missing = [budget for budget in budgets if str(budget) not in matched]
    if missing:
        raise ContractError(f"control lock is missing matched budgets: {missing}")
    for budget in budgets:
        value = matched[str(budget)]
        if not isinstance(value, int) or value < 1:
            raise ContractError(
                "matched Teacher-0 sims/action must be positive integers"
            )
        comparison = (calibration_payload.get("matches") or {}).get(str(budget)) or {}
        if comparison.get("flat_sims_per_action") != value:
            raise ContractError(
                f"latency calibration does not support Teacher-1 budget {budget}"
            )
        if float(comparison.get("relative_p50_gap", 1.0)) > 0.10:
            raise ContractError(
                f"Teacher-0 control for Teacher-1 budget {budget} is not within 10%"
            )
    return lock, canonical_sha256(lock)


def _teacher_spec(contract: dict[str, Any], budget: int) -> dict[str, Any]:
    teacher = contract["teacher"]
    return {
        "kind": "determinized_puct",
        "name": f"t1-{budget}-w{teacher['worlds']}",
        "sims": budget,
        "worlds": int(teacher["worlds"]),
        "c_puct": float(teacher["c_puct"]),
        "max_steps": int(teacher["max_steps"]),
    }


def _search_stats(stats: Any) -> dict[str, Any] | None:
    if stats is None:
        return None
    payload: dict[str, Any] = stats.to_dict()
    seconds = np.asarray(stats.decision_seconds, dtype=np.float64)
    payload.update(
        p50_decision_ms=(
            float(np.quantile(seconds, 0.50)) * 1000 if len(seconds) else None
        ),
        p95_decision_ms=(
            float(np.quantile(seconds, 0.95)) * 1000 if len(seconds) else None
        ),
        labels_per_second=(stats.decisions / stats.seconds if stats.seconds else None),
        cap_rate=(stats.cap_hits / stats.simulations if stats.simulations else 0.0),
    )
    return payload


def _play_cell(
    hero: dict[str, Any], villain: dict[str, Any], *, games: int, seed: int
) -> dict[str, Any]:
    result = play_games(hero, villain, num_games=games, seed=seed)
    return {
        **aggregate_records(result.records),
        "hero": result.hero,
        "villain": result.villain,
        "wall_seconds": result.wall_seconds,
        "hero_search": _search_stats(result.hero_search),
        "villain_search": _search_stats(result.villain_search),
    }


def _integrity_from_audit(artifact: dict[str, Any]) -> dict[str, Any]:
    invalid_visit_rows = 0
    invalid_value_rows = 0
    invalid_legal_rows = 0
    invalid_selected_rows = 0
    mutated_shapes = 0
    encoded_offer_mismatches = 0
    mutated_roots = 0
    for game in artifact["games"]:
        for decision in game["decisions"]:
            offers = decision["frame"]["offers"]
            search = decision["search"]
            visits = np.asarray(search["visit_counts"], dtype=np.float64)
            q_values = np.asarray(search["q_values"], dtype=np.float64)
            legal = {int(offer["id"]) for offer in offers}
            command = int(decision["command"]["offer_id"])
            if (
                not np.isfinite(visits).all()
                or (visits < 0).any()
                or int(visits.sum()) != int(search["simulations"])
            ):
                invalid_visit_rows += 1
            if (
                not np.isfinite(q_values).all()
                or (q_values < 0).any()
                or (q_values > 1).any()
                or not 0 <= float(search["root_value"]) <= 1
            ):
                invalid_value_rows += 1
            if command not in legal:
                invalid_legal_rows += 1
            elif visits[command] != visits.max():
                invalid_selected_rows += 1
            if len(visits) != len(q_values) or len(visits) != len(offers):
                mutated_shapes += 1
            if int(search["encoded_legal_count"]) != len(offers):
                encoded_offer_mismatches += 1
            if not bool(search["root_unchanged"]):
                mutated_roots += 1
    return {
        "invalid_visit_rows": invalid_visit_rows,
        "invalid_value_rows": invalid_value_rows,
        "invalid_legal_rows": invalid_legal_rows,
        "invalid_selected_rows": invalid_selected_rows,
        "shape_mismatches": mutated_shapes,
        "encoded_offer_mismatches": encoded_offer_mismatches,
        "mutated_authoritative_roots": mutated_roots,
        "passed": not any(
            (
                invalid_visit_rows,
                invalid_value_rows,
                invalid_legal_rows,
                invalid_selected_rows,
                mutated_shapes,
                encoded_offer_mismatches,
                mutated_roots,
            )
        ),
    }


def _value_calibration_from_audit(artifact: dict[str, Any]) -> dict[str, Any]:
    """Compare actor-relative root values with the played terminal outcomes."""

    predictions: list[float] = []
    outcomes: list[float] = []
    for game in artifact["games"]:
        winner = game["winner"]
        for decision in game["decisions"]:
            predictions.append(float(decision["search"]["root_value"]))
            outcomes.append(
                0.5 if winner is None else float(decision["actor"] == winner)
            )
    if not predictions:
        return {"decisions": 0, "brier": None, "ece_10": None, "bins": []}
    prediction_array = np.asarray(predictions, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        selected = (prediction_array >= lower) & (
            prediction_array <= upper if index == 9 else prediction_array < upper
        )
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        mean_prediction = float(np.mean(prediction_array[selected]))
        outcome_rate = float(np.mean(outcome_array[selected]))
        ece += (count / len(prediction_array)) * abs(mean_prediction - outcome_rate)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "outcome_rate": outcome_rate,
            }
        )
    return {
        "decisions": len(predictions),
        "mean_prediction": float(np.mean(prediction_array)),
        "outcome_rate": float(np.mean(outcome_array)),
        "brier": float(np.mean((prediction_array - outcome_array) ** 2)),
        "ece_10": float(ece),
        "bins": bins,
        "interpretation": (
            "diagnostic terminal calibration of random-leaf root values; "
            "not a hard teacher-admission gate"
        ),
    }


def _competency_gates(
    results: dict[str, Any], *, high_name: str, allowed_regression: float
) -> dict[str, Any]:
    delayed = []
    no_large_regression = True
    details: dict[str, Any] = {}
    for scenario_name in SCENARIOS:
        block = results[scenario_name]
        if "error" in block.get(high_name, {}) or "error" in block.get("random", {}):
            details[scenario_name] = {
                "teacher": block.get(high_name),
                "random": block.get("random"),
            }
            no_large_regression = False
            if scenario_name in {
                "s1_counter_the_bomb",
                "s2_hold_the_wipe",
                "s5_hold_up_quench",
            }:
                delayed.append(0.0)
            continue
        high = float(block[high_name]["correct_rate"])
        random = float(block["random"]["correct_rate"])
        details[scenario_name] = {"teacher": high, "random": random}
        if scenario_name in {
            "s1_counter_the_bomb",
            "s2_hold_the_wipe",
            "s5_hold_up_quench",
        }:
            delayed.append(high)
        no_large_regression &= high >= random - allowed_regression
    return {
        "one_delayed_scenario_at_threshold": max(delayed, default=0.0) >= 0.20,
        "no_scenario_regresses_over_tolerance": no_large_regression,
        "details": details,
    }


def _evaluate_gates(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    budgets = [int(value) for value in contract["teacher"]["budgets"]]
    low, high = min(budgets), max(budgets)
    high_random = manifest["cells"][f"t1-{high}-vs-random"]
    high_checkpoint = manifest["cells"][f"t1-{high}-vs-checkpoint"]
    high_low = manifest["cells"][f"t1-{high}-vs-t1-{low}"]
    stability_low = manifest["root_stability"][str(low)]
    stability_high = manifest["root_stability"][str(high)]
    high_search = high_random["hero_search"]
    thresholds = contract["gates"]
    matched_compute: dict[str, Any] = {}
    for budget in budgets:
        cell = manifest["cells"][f"t1-{budget}-vs-flat-wall"]
        teacher_p50 = float(cell["hero_search"]["p50_decision_ms"])
        flat_p50 = float(cell["villain_search"]["p50_decision_ms"])
        relative_gap = abs(flat_p50 - teacher_p50) / teacher_p50
        matched_compute[str(budget)] = {
            "teacher_p50_decision_ms": teacher_p50,
            "flat_p50_decision_ms": flat_p50,
            "relative_gap": relative_gap,
            "passed": relative_gap <= thresholds["max_realized_p50_gap"],
        }
    competency = _competency_gates(
        manifest["competencies"],
        high_name=f"t1-{high}-w{contract['teacher']['worlds']}",
        allowed_regression=float(thresholds["competency_max_regression_vs_random"]),
    )
    gates = {
        "trajectory_replay": bool(manifest["trajectory_replay"]["passed"]),
        "target_integrity": bool(manifest["target_integrity"]["passed"]),
        "high_vs_low": high_low["win_rate"] >= thresholds["high_vs_low_win_rate"],
        "high_vs_checkpoint": high_checkpoint["win_rate"]
        >= thresholds["high_vs_checkpoint_win_rate"],
        "high_vs_random": high_random["win_rate"]
        >= thresholds["high_vs_random_win_rate"],
        "top_action_stability": stability_high["top_action_agreement"]
        >= thresholds["high_top_action_agreement"],
        "js_stability": stability_high["median_js_divergence"]
        <= thresholds["high_median_js_divergence"],
        "tree_growth": stability_high["mean_tree_nodes"]
        > stability_low["mean_tree_nodes"],
        "depth_growth": stability_high["mean_max_depth"]
        > stability_low["mean_max_depth"],
        "competency_signal": competency["one_delayed_scenario_at_threshold"],
        "competency_nonregression": competency["no_scenario_regresses_over_tolerance"],
        "latency": high_search["p95_decision_ms"] <= thresholds["high_p95_decision_ms"],
        "throughput": high_search["labels_per_second"]
        >= thresholds["high_labels_per_second_per_worker"],
        "cap_rate": high_search["cap_rate"] <= thresholds["max_playout_cap_rate"],
        "matched_compute": all(
            comparison["passed"] for comparison in matched_compute.values()
        ),
    }
    return {
        "checks": gates,
        "competency": competency,
        "matched_compute": matched_compute,
        "passed": all(gates.values()),
    }


def _verify_existing(
    run_dir: Path, contract: dict[str, Any], contract_hash: str
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    if manifest.get("contract_sha256") != contract_hash:
        raise ContractError("run manifest was produced by a different contract")
    audit_path = run_dir / "trajectory-audit.json"
    if file_sha256(audit_path) != manifest.get("trajectory_sha256"):
        raise RuntimeError("trajectory audit SHA-256 changed")
    artifact = _load_json(audit_path)
    actual = runtime_fingerprints(seed=int(contract["seeds"]["runtime"]))
    validate_runtime_fingerprints(contract["expected_fingerprints"], actual)
    replay = replay_teacher_trajectories(
        artifact,
        content_hash=actual["experience_content_hash"],
        asset_manifest_hash=actual["asset_manifest_hash"],
    )
    expected = manifest.get("trajectory_replay")
    current = receipt_dict(replay)
    if expected != current:
        raise RuntimeError("trajectory replay receipt changed")
    return {
        "contract_sha256": contract_hash,
        "trajectory_sha256": file_sha256(audit_path),
        "trajectory_replay": current,
        "verified": True,
    }


def run_teacher_gate(
    *,
    contract_path: Path,
    control_lock_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract, contract_hash = _load_contract(contract_path)
    budgets = [int(value) for value in contract["teacher"]["budgets"]]
    lock, lock_hash = _load_control_lock(
        control_lock_path, contract_hash=contract_hash, budgets=budgets
    )
    runtime = runtime_fingerprints(seed=int(contract["seeds"]["runtime"]))
    validate_runtime_fingerprints(contract["expected_fingerprints"], runtime)
    expected_host = contract["host"]
    actual_host = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    expected_host_subset = {key: expected_host[key] for key in actual_host}
    if actual_host != expected_host_subset:
        raise ContractError(f"host runtime {actual_host!r} != {expected_host_subset!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if (
            manifest.get("contract_sha256") != contract_hash
            or manifest.get("control_lock_sha256") != lock_hash
        ):
            raise ContractError("existing output belongs to a different run contract")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": contract["experiment"],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "contract_path": str(contract_path),
            "contract_sha256": contract_hash,
            "control_lock_path": str(control_lock_path),
            "control_lock_sha256": lock_hash,
            "runtime": runtime,
            "cells": {},
        }
        _atomic_json(manifest_path, manifest)

    started = time.perf_counter()
    prior_wall_seconds = float(manifest.get("wall_seconds_cumulative", 0.0))
    wall_cap = float(contract["caps"]["teacher_gate_wall_hours"]) * 3600
    artifact_cap = int(contract["caps"]["artifact_bytes"])

    def artifact_bytes() -> int:
        return sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())

    def save_manifest() -> None:
        elapsed = time.perf_counter() - started
        manifest["wall_seconds_this_invocation"] = elapsed
        manifest["wall_seconds_cumulative"] = prior_wall_seconds + elapsed
        manifest["artifact_bytes"] = artifact_bytes()
        _atomic_json(manifest_path, manifest)

    def check_cap() -> None:
        elapsed = time.perf_counter() - started
        if prior_wall_seconds + elapsed > wall_cap:
            manifest["status"] = "stopped_wall_cap"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            save_manifest()
            raise SystemExit("Teacher-1 gate reached its pre-registered wall cap")
        if artifact_bytes() > artifact_cap:
            manifest["status"] = "stopped_artifact_cap"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            save_manifest()
            raise SystemExit("Teacher-1 gate reached its pre-registered artifact cap")

    checkpoint_path = lock["checkpoint_control"]["path"]
    games = int(contract["evaluation"]["games_per_matchup"])
    base_seed = int(contract["seeds"]["matchups"])
    cell_specs: list[tuple[str, dict[str, Any], dict[str, Any], int]] = []
    for budget in budgets:
        teacher = _teacher_spec(contract, budget)
        flat = {
            "kind": "search",
            "name": f"t0-wall-for-t1-{budget}",
            "sims": int(lock["flat_sims_per_action_by_teacher_budget"][str(budget)]),
        }
        cell_specs.extend(
            [
                (f"t1-{budget}-vs-random", teacher, {"kind": "random"}, 10_000),
                (
                    f"t1-{budget}-vs-checkpoint",
                    teacher,
                    {
                        "kind": "checkpoint",
                        "path": checkpoint_path,
                        "name": "teacher0-policy-value-frozen",
                        "deterministic": True,
                    },
                    20_000,
                ),
                (f"t1-{budget}-vs-flat-wall", teacher, flat, 30_000),
            ]
        )
    low, high = min(budgets), max(budgets)
    cell_specs.append(
        (
            f"t1-{high}-vs-t1-{low}",
            _teacher_spec(contract, high),
            _teacher_spec(contract, low),
            40_000,
        )
    )
    for cell_id, hero, villain, seed_offset in cell_specs:
        if cell_id in manifest["cells"]:
            continue
        check_cap()
        manifest["cells"][cell_id] = _play_cell(
            hero,
            villain,
            games=games,
            seed=base_seed + seed_offset,
        )
        save_manifest()

    check_cap()
    if "root_stability" not in manifest:
        manifest["root_stability"] = evaluate_root_stability(
            budgets=budgets,
            worlds=int(contract["teacher"]["worlds"]),
            c_puct=float(contract["teacher"]["c_puct"]),
            roots=int(contract["evaluation"]["stability_roots"]),
            repeats=int(contract["evaluation"]["stability_repeats"]),
            seed=int(contract["seeds"]["stability"]),
            max_steps=int(contract["teacher"]["max_steps"]),
        )
        save_manifest()

    check_cap()
    audit_path = out_dir / "trajectory-audit.json"
    if not audit_path.exists():
        audit = record_teacher_trajectories(
            games=int(contract["evaluation"]["trajectory_audit_games"]),
            simulations=high,
            worlds=int(contract["teacher"]["worlds"]),
            c_puct=float(contract["teacher"]["c_puct"]),
            seed=int(contract["seeds"]["trajectory"]),
            content_hash=runtime["experience_content_hash"],
            asset_manifest_hash=runtime["asset_manifest_hash"],
            max_steps=int(contract["teacher"]["max_steps"]),
            provenance={
                "contract_sha256": contract_hash,
                "control_lock_sha256": lock_hash,
                "runtime": runtime,
            },
        )
        _atomic_json(audit_path, audit)
    audit = _load_json(audit_path)
    replay = replay_teacher_trajectories(
        audit,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
    )
    manifest["trajectory_sha256"] = file_sha256(audit_path)
    manifest["trajectory_replay"] = receipt_dict(replay)
    manifest["target_integrity"] = _integrity_from_audit(audit)
    manifest["value_calibration"] = _value_calibration_from_audit(audit)
    save_manifest()

    check_cap()
    scenario_path = out_dir / "competencies.json"
    competency_specs = [{"kind": "random"}] + [
        _teacher_spec(contract, budget) for budget in budgets
    ]
    manifest["competencies"] = run_scenario_suite(
        list(SCENARIOS),
        competency_specs,
        runs=int(contract["evaluation"]["scenario_runs"]),
        workers=int(contract["host"]["max_workers"]),
        base_seed=int(contract["seeds"]["scenarios"]),
        out_path=scenario_path,
    )
    manifest["competencies_sha256"] = file_sha256(scenario_path)
    check_cap()
    manifest["teacher_gate"] = _evaluate_gates(manifest, contract)
    manifest["status"] = (
        "completed_pass" if manifest["teacher_gate"]["passed"] else "completed_kill"
    )
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["student_training_authorized"] = bool(manifest["teacher_gate"]["passed"])
    save_manifest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="experiments/contracts/w2-234-teacher1-pilot-v1.json",
    )
    parser.add_argument(
        "--control-lock",
        default="experiments/contracts/w2-234-teacher1-control-lock-v1.json",
    )
    parser.add_argument("--out-dir", default=".runs/w2-234-teacher1-pilot-v1")
    parser.add_argument("--stage", choices=("teacher-gate",))
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--print-runtime", action="store_true")
    args = parser.parse_args()

    if args.print_runtime:
        print(json.dumps(runtime_fingerprints(), indent=2, sort_keys=True))
        return

    contract_path = Path(args.contract)
    contract, contract_hash = _load_contract(contract_path)
    if args.verify is not None:
        print(
            json.dumps(
                _verify_existing(args.verify, contract, contract_hash),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.stage != "teacher-gate":
        parser.error("pass --stage teacher-gate, --verify RUN_DIR, or --print-runtime")
    result = run_teacher_gate(
        contract_path=contract_path,
        control_lock_path=Path(args.control_lock),
        out_dir=Path(args.out_dir),
    )
    print(f"{result['status']}: teacher gate -> {Path(args.out_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
