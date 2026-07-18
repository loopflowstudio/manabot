#!/usr/bin/env python3
"""Freeze and challenge the world-pinned manabot skill arena."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from itertools import combinations
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np

from manabot.arena.competency import run_competencies
from manabot.arena.guidance import build_arena_player
from manabot.arena.match import play_cell
from manabot.arena.models import (
    ArenaContract,
    MatchRow,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.profile import (
    native_gameplay_profiles,
    profile_players,
    select_profile_roots,
    verify_profile,
)
from manabot.arena.rating import bootstrap_population, fit_population, payoff_matrix
from manabot.arena.replay import read_trace, replay_games
from manabot.sim.teacher1_evidence import (
    REPO_ROOT,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.verify.competency import SCENARIOS, aggregate_scenario_results


class ArenaError(RuntimeError):
    """The closed arena contract or artifact failed validation."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


def source_digest(paths: tuple[str, ...] | list[str]) -> str:
    return source_bundle_sha256([REPO_ROOT / path for path in paths])


def arena_runtime_fingerprints() -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    engine_runtime = runtime_fingerprints()
    engine_runtime.pop("matchup")
    engine_runtime.pop("pilot_source_sha256")
    return {
        **engine_runtime,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": metadata.version("numpy"),
        "torch_version": metadata.version("torch"),
        "pydantic_version": metadata.version("pydantic"),
        "psutil_version": metadata.version("psutil"),
        "torch_threads_per_player": torch.get_num_threads(),
        "inference_device": "cpu",
    }


def registration_source_paths(registration: PlayerRegistration) -> list[str]:
    kind = registration.player_spec["kind"]
    if kind == "scripted_greedy":
        return ["manabot/arena/players.py"]
    if kind == "determinized_puct":
        return [
            "manabot/sim/flat_mc.py",
            "manabot/sim/mcts.py",
            "manabot/sim/search_branch.py",
            "managym/src/python/bindings.rs",
        ]
    return ["manabot/sim/flat_mc.py"]


def validate_registration_source(registration: PlayerRegistration) -> None:
    if registration.runner_kind != "code":
        return
    actual = source_digest(registration_source_paths(registration))
    if registration.source_sha256 != actual:
        raise ArenaError(f"source drift for {registration.player_id}: {actual}")


def preflight_contract(path: Path) -> tuple[ArenaContract, str, dict[str, Any]]:
    raw = load_json(path)
    contract = ArenaContract.model_validate(raw)
    contract_sha = file_sha256(path)
    actual_source = source_digest(contract.source_paths)
    if actual_source != contract.source_sha256:
        raise ArenaError(f"arena source drift: {actual_source}")
    actual_runtime = arena_runtime_fingerprints()
    mismatches = {
        key: {"expected": expected, "actual": actual_runtime.get(key)}
        for key, expected in contract.runtime.items()
        if actual_runtime.get(key) != expected
    }
    if mismatches:
        raise ArenaError("runtime drift: " + json.dumps(mismatches, sort_keys=True))
    for registration in contract.anchors:
        validate_registration_source(registration)
    return contract, contract_sha, actual_runtime


def preflight_candidate(
    path: Path,
    checkpoint_path: Path | None,
    *,
    contract: ArenaContract,
    profile_name: str,
    validate_checkpoint_load: bool = True,
) -> tuple[PlayerRegistration, dict[str, str]]:
    registration = PlayerRegistration.model_validate(load_json(path))
    if registration.role != "challenger":
        raise ArenaError("candidate registration role must be challenger")
    if (
        registration.world != contract.key.world
        or registration.content_suite != contract.key.content_suite
        or registration.information_boundary != contract.key.viewer_boundary
        or registration.observation_abi_sha256
        != contract.runtime["observation_abi_sha256"]
        or registration.action_abi_sha256 != contract.runtime["action_abi_sha256"]
        or registration.matchup_sha256 != contract.runtime["matchup_sha256"]
    ):
        raise ArenaError("candidate world/content/viewer compatibility mismatch")
    if registration.evidence_class == "fixture" and profile_name != "smoke":
        raise ArenaError("fixture checkpoint registration is smoke-only")
    checkpoint_paths: dict[str, str] = {}
    if registration.runner_kind == "code":
        if checkpoint_path is not None:
            raise ArenaError("code candidate rejects --candidate-checkpoint")
        validate_registration_source(registration)
    else:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ArenaError("checkpoint candidate bytes are unavailable")
        if file_sha256(checkpoint_path) != registration.checkpoint_sha256:
            raise ArenaError("checkpoint candidate SHA-256 mismatch")
        if checkpoint_path.stat().st_size != registration.checkpoint_bytes:
            raise ArenaError("checkpoint candidate byte-size mismatch")
        checkpoint_paths[registration.player_id] = str(checkpoint_path)
        if validate_checkpoint_load:
            player, observation_space = build_arena_player(
                registration, seed=0, checkpoint_path=str(checkpoint_path)
            )
            if observation_space is None:
                raise ArenaError("checkpoint candidate has no observation space")
            observation_schema = {
                name: {"shape": list(shape), "dtype": "float32"}
                for name, shape in sorted(observation_space.shapes.items())
            }
            if canonical_sha256(observation_schema) != (
                registration.observation_abi_sha256
            ):
                raise ArenaError("checkpoint candidate observation-shape mismatch")
            agent = getattr(player, "agent", None)
            if agent is None:
                agent = getattr(getattr(player, "evaluator", None), "agent", None)
            parameter_count = (
                sum(parameter.numel() for parameter in agent.parameters())
                if agent is not None
                else None
            )
            if parameter_count != registration.parameter_count:
                raise ArenaError("checkpoint candidate parameter-count mismatch")
    return registration, checkpoint_paths


def append_ledger(out_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    path = out_dir / "resource-ledger.jsonl"
    previous = None
    if path.exists() and path.read_text().splitlines():
        previous = json.loads(path.read_text().splitlines()[-1])["event_sha256"]
    unsigned = {
        "stage": stage,
        "recorded_unix": time.time(),
        "previous_event_sha256": previous,
        "payload": payload,
    }
    event = {**unsigned, "event_sha256": canonical_sha256(unsigned)}
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def play_cells(
    *,
    contract: ArenaContract,
    pairs: list[tuple[PlayerRegistration, PlayerRegistration]],
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    comparison_seed_aliases: dict[str, str] | None = None,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    jobs = [
        {
            "key": contract.key,
            "player_a": first,
            "player_b": second,
            "deal_seeds": deal_seeds,
            "out_dir": out_dir,
            "checkpoint_paths": checkpoint_paths,
            "comparison_seed_aliases": comparison_seed_aliases,
        }
        for first, second in pairs
    ]
    with ProcessPoolExecutor(
        max_workers=contract.resource_caps.outcome_workers
    ) as executor:
        return list(executor.map(_play_cell_job, jobs))


def _play_cell_job(
    job: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return play_cell(**job)


def artifact_receipts(out_dir: Path, names: list[str]) -> dict[str, Any]:
    return {
        name: {"path": name, "sha256": file_sha256(out_dir / name)} for name in names
    }


def portable_trace_receipts(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = []
    for trace in traces:
        receipt = dict(trace)
        receipt["path"] = receipt.pop("artifact_path")
        receipts.append(receipt)
    return receipts


def artifact_bytes(out_dir: Path) -> int:
    return sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())


def resource_cap_receipt(
    out_dir: Path,
    contract: ArenaContract,
    *,
    started: float,
    wall_hours_max: float | None = None,
    core_hours_max: float | None = None,
    artifact_bytes_max: int | None = None,
    additional_artifact_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    wall_seconds = time.perf_counter() - started
    wall_hours = wall_seconds / 3600.0
    core_hours = wall_hours * contract.resource_caps.outcome_workers
    current_artifact_bytes = artifact_bytes(out_dir) + sum(
        artifact_bytes(path) for path in additional_artifact_dirs if path.exists()
    )
    wall_limit = (
        contract.resource_caps.wall_hours if wall_hours_max is None else wall_hours_max
    )
    core_limit = (
        contract.resource_caps.core_hours if core_hours_max is None else core_hours_max
    )
    artifact_limit = (
        contract.resource_caps.artifact_bytes
        if artifact_bytes_max is None
        else artifact_bytes_max
    )
    clauses = {
        "wall_hours": {
            "actual": wall_hours,
            "maximum": wall_limit,
            "passed": wall_hours <= wall_limit,
        },
        "core_hours_conservative": {
            "actual": core_hours,
            "maximum": core_limit,
            "passed": core_hours <= core_limit,
        },
        "artifact_bytes_at_decision": {
            "actual": current_artifact_bytes,
            "maximum": artifact_limit,
            "passed": current_artifact_bytes <= artifact_limit,
        },
    }
    return {
        "clauses": clauses,
        "passed": all(row["passed"] for row in clauses.values()),
    }


def finalize_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {**manifest, "manifest_sha256": manifest_digest(manifest)}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def verify_ledger(path: Path) -> dict[str, Any]:
    previous = None
    events = read_jsonl(path)
    for event in events:
        unsigned = dict(event)
        event_sha256 = unsigned.pop("event_sha256", None)
        if unsigned.get("previous_event_sha256") != previous:
            raise ArenaError("resource ledger chain mismatch")
        if event_sha256 != canonical_sha256(unsigned):
            raise ArenaError("resource ledger event digest mismatch")
        previous = event_sha256
    if not events or events[-1]["stage"] != "complete":
        raise ArenaError("resource ledger is incomplete")
    return {"events": len(events), "head_sha256": previous}


def report_markdown(
    *,
    fit: Any,
    promotion: dict[str, Any],
    matrix: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    lines = [
        "# Manabot Skill Arena",
        "",
        f"Disposition: **{promotion['disposition']}**",
        f"Reason: `{promotion.get('reason')}`",
        "",
        "## Ratings",
        "",
    ]
    for player, rating in sorted(fit.ratings.items(), key=lambda item: -item[1]):
        lines.append(f"- `{player}`: {rating:.1f} Elo")
    lines.extend(
        [
            "",
            f"Seat-0 effect: {fit.seat0_elo:.1f} Elo",
            f"Log loss: {fit.log_loss:.6f}",
            f"Payoff cells: {len(matrix)}",
            f"Replay passed: {replay.get('passed', False)}",
            "",
        ]
    )
    return "\n".join(lines)


def rating_payload(
    rows: list[dict[str, Any]], schedule: Any, contract: ArenaContract
) -> tuple[Any, dict[str, Any]]:
    model = contract.rating_model
    fit = fit_population(
        rows,
        anchor=model.anchor_player_id,
        prior_elo_std=model.prior_elo_std,
        tolerance=model.tolerance,
        max_iterations=model.max_iterations,
    )
    if not fit.converged:
        raise ArenaError("Bradley-Terry MAP fit did not converge")
    bootstrap = bootstrap_population(
        rows,
        replicates=schedule.bootstrap_replicates,
        seed=schedule.bootstrap_seed,
        anchor=model.anchor_player_id,
        prior_elo_std=model.prior_elo_std,
    )
    matrix = payoff_matrix(rows)
    largest_residual_cells = sorted(
        matrix.values(),
        key=lambda cell: abs(float(cell["raw_residual_percentage_points"])),
        reverse=True,
    )[:10]
    payload = {
        **model.model_dump(),
        "rating_model_sha256": contract.key.rating_prior_sha256,
        "ratings": fit.ratings,
        "seat0_elo": fit.seat0_elo,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "gradient_norm": fit.gradient_norm,
        "hessian_condition": fit.hessian_condition,
        "log_loss": fit.log_loss,
        "deviance": fit.deviance,
        "residual_rows": list(fit.rows),
        "largest_residual_cells": largest_residual_cells,
        "bootstrap": bootstrap,
    }
    return fit, payload


def promotion_payload(
    candidate: PlayerRegistration | None,
    rows: list[dict[str, Any]],
    *,
    smoke: bool,
    profile: dict[str, Any],
    competencies: dict[str, Any],
    contract: ArenaContract,
) -> dict[str, Any]:
    row_integrity_passed = all(
        row.get("replay_passed")
        and not any(int(value) for value in row["integrity"].values())
        for row in rows
    )
    matched_players = profile["matched_root"]["players"]
    profile_integrity_passed = all(
        int(player["root_mutations"]) == 0
        and int(player["illegal_actions"]) == 0
        and float(player["playout_cap_rate"]) <= contract.promotion.playout_cap_rate_max
        for player in matched_players.values()
    )
    resource_caps_passed = bool(profile["resource_caps"]["passed"])
    integrity_passed = (
        row_integrity_passed and profile_integrity_passed and resource_caps_passed
    )
    if not integrity_passed:
        disposition = {
            "disposition": "invalid_integrity",
            "reason": "zero_tolerance_gate",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    if smoke:
        disposition = {
            "disposition": "engineering_smoke_non_promotion",
            "reason": "fixture_or_smoke_profile",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    if candidate is None:
        disposition = {
            "disposition": "rated_not_promotion_eligible",
            "reason": "anchor_freeze",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    matched = profile["matched_root"]["players"][candidate.player_id]
    integrity = {
        "match_rows_passed": row_integrity_passed,
        "matched_root_mutations": int(matched["root_mutations"]),
        "matched_root_illegal_actions": int(matched["illegal_actions"]),
        "playout_cap_rate": float(matched["playout_cap_rate"]),
        "playout_cap_rate_max": contract.promotion.playout_cap_rate_max,
        "passed": int(matched["root_mutations"]) == 0
        and int(matched["illegal_actions"]) == 0
        and float(matched["playout_cap_rate"])
        <= contract.promotion.playout_cap_rate_max
        and resource_caps_passed,
        "resource_caps": profile["resource_caps"],
    }
    disposition = {
        "disposition": "rated_not_promotion_eligible",
        "reason": "incumbent_not_in_cohort",
        "candidate": candidate.player_id,
        "compute_class_id": candidate.compute_class_id,
        "eligibility": {
            "available": False,
            "reason": "incumbent_not_in_cohort",
            "fixed_compute": {"available": False},
            "rating_delta": {"available": False},
            "competency_noninferiority": {"available": False},
        },
        "integrity": integrity,
        "candidate_competencies_sha256": canonical_sha256(
            competencies["players"][candidate.player_id]
        ),
    }
    return {**disposition, "input_sha256": canonical_sha256(disposition)}


INT8_EVIDENCE_CLASS = "engineering_smoke_only_no_admission_claim"
INT8_COMPARISON_ALIAS = "int-8-guidance-comparison-v1"
INT8_BUDGETS = (8, 32, 128)
INT8_IMPLEMENTATION_SOURCE_PATHS = (
    "experiments/runners/run_skill_arena.py",
    "manabot/arena/competency.py",
    "manabot/arena/guidance.py",
    "manabot/arena/int8_input.py",
    "manabot/arena/match.py",
    "manabot/arena/models.py",
    "manabot/arena/players.py",
    "manabot/arena/profile.py",
    "manabot/arena/rating.py",
    "manabot/arena/replay.py",
)
INT8_MODULE_PATHS = {
    "guidance": "manabot/arena/guidance.py",
    "retained_input": "manabot/arena/int8_input.py",
}


def diagnostic_profile_variants(
    candidates: list[PlayerRegistration],
) -> list[PlayerRegistration]:
    """Materialize the frozen 8/32/128 profile identities for each prior arm."""

    variants = []
    for candidate in candidates:
        if candidate.player_spec["sims"] != 32:
            raise ArenaError("diagnostic gameplay candidates must use 32 traversals")
        stem = candidate.player_id.removesuffix("-32-v1")
        for budget in INT8_BUDGETS:
            spec = {**candidate.player_spec, "sims": budget}
            variants.append(
                PlayerRegistration.model_validate(
                    candidate.model_copy(
                        update={
                            "player_id": f"{stem}-{budget}-v1",
                            "display_name": candidate.display_name.replace(
                                "32 v1", f"{budget} v1"
                            ),
                            "player_spec": spec,
                            "compute_class_id": candidate.compute_class_id.replace(
                                "s32", f"s{budget}"
                            ),
                        }
                    ).model_dump()
                )
            )
    if len({variant.player_id for variant in variants}) != 9:
        raise ArenaError("diagnostic profile variant identities are not unique")
    return variants


def _profile_arm(registration: PlayerRegistration) -> str:
    if registration.player_id.startswith("uniform-prior"):
        return "uniform"
    if registration.player_id.startswith("chosen-policy-prior"):
        return "chosen"
    if registration.player_id.startswith("visit-policy-prior"):
        return "visit"
    raise ArenaError(f"unknown diagnostic prior arm: {registration.player_id}")


def _agreement_payload(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_samples = left["samples"]
    right_samples = right["samples"]
    if [row["root_id"] for row in left_samples] != [
        row["root_id"] for row in right_samples
    ]:
        raise ArenaError("label agreement roots are not matched")
    rows = [
        {
            "root_id": left_row["root_id"],
            "action_space_kind": left_row["action_space_kind"],
            "legal_action_count": left_row["legal_action_count"],
            "agreed": int(left_row["action"] == right_row["action"]),
        }
        for left_row, right_row in zip(left_samples, right_samples, strict=True)
    ]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "roots": len(group),
            "agreements": sum(int(row["agreed"]) for row in group),
            "rate": sum(int(row["agreed"]) for row in group) / len(group),
        }

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = f"{row['action_space_kind']}__legal-{row['legal_action_count']}"
        buckets.setdefault(bucket, []).append(row)
    return {
        "aggregate": summarize(rows),
        "buckets": {key: summarize(group) for key, group in sorted(buckets.items())},
    }


def guidance_mechanism_payload(
    profile: dict[str, Any], variants: list[PlayerRegistration]
) -> dict[str, Any]:
    by_arm_budget = {
        (_profile_arm(variant), int(variant.player_spec["sims"])): profile["players"][
            variant.player_id
        ]
        for variant in variants
    }
    agreement_pairs = {
        "uniform32_vs_chosen32": (("uniform", 32), ("chosen", 32)),
        "uniform32_vs_visit32": (("uniform", 32), ("visit", 32)),
        "chosen32_vs_visit32": (("chosen", 32), ("visit", 32)),
        "uniform32_vs_uniform128": (("uniform", 32), ("uniform", 128)),
        "chosen32_vs_uniform128": (("chosen", 32), ("uniform", 128)),
        "visit32_vs_uniform128": (("visit", 32), ("uniform", 128)),
    }
    agreements = {
        name: _agreement_payload(by_arm_budget[left], by_arm_budget[right])
        for name, (left, right) in agreement_pairs.items()
    }
    return {
        "schema_version": 1,
        "evidence_class": INT8_EVIDENCE_CLASS,
        "root_corpus_sha256": profile["root_corpus_sha256"],
        "profile_registration_sha256": {
            variant.player_id: variant.identity_sha256 for variant in variants
        },
        "selected_command_agreement": agreements,
        "high_budget_reference": "uniform-prior-puct-128-v1",
        "player_metrics": {
            variant.player_id: {
                "budget": int(variant.player_spec["sims"]),
                "arm": _profile_arm(variant),
                "p50_seconds": profile["players"][variant.player_id]["p50_seconds"],
                "p95_seconds": profile["players"][variant.player_id]["p95_seconds"],
                "nodes_per_second": profile["players"][variant.player_id][
                    "nodes_per_second"
                ],
                "decisions_per_second": profile["players"][variant.player_id][
                    "decisions_per_second"
                ],
                "cpu_seconds_per_label": profile["players"][variant.player_id][
                    "cpu_seconds_per_label"
                ],
                "nodes_per_label": profile["players"][variant.player_id][
                    "nodes_per_label"
                ],
                "mechanism": profile["players"][variant.player_id]["mechanism"],
            }
            for variant in variants
        },
    }


def _candidate_anchor_scores(
    rows: list[dict[str, Any]], candidate_id: str, anchor_ids: set[str]
) -> dict[tuple[str, int, int], float]:
    scores = {}
    for row in rows:
        if row["player_a"] != candidate_id or row["player_b"] not in anchor_ids:
            continue
        key = (str(row["player_b"]), int(row["deal_block"]), int(row["leg"]))
        scores[key] = float(row["score_a"])
    return scores


def _competency_correct_count(competencies: dict[str, Any], player_id: str) -> int:
    return sum(
        int(bool(run["correct"]))
        for scenario in competencies["players"][player_id].values()
        for run in scenario["runs"]
    )


def diagnostic_decision_payload(
    *,
    candidates: list[PlayerRegistration],
    rows: list[dict[str, Any]],
    competencies: dict[str, Any],
    profile: dict[str, Any],
    mechanism: dict[str, Any],
    resource_caps: dict[str, Any],
    anchor_ids: set[str],
) -> dict[str, Any]:
    candidate_by_arm = {_profile_arm(candidate): candidate for candidate in candidates}
    if set(candidate_by_arm) != {"uniform", "chosen", "visit"}:
        raise ArenaError("diagnostic requires exactly uniform/chosen/visit candidates")
    row_integrity = all(
        row.get("replay_passed")
        and not any(int(value) for value in row["integrity"].values())
        for row in rows
    )
    profile_integrity = all(
        int(player["root_mutations"]) == 0
        and int(player["illegal_actions"]) == 0
        and float(player["playout_cap_rate"]) <= 0.001
        for player in profile["players"].values()
    )
    integrity = {
        "match_rows_passed": row_integrity,
        "profile_rows_passed": profile_integrity,
        "resource_caps_passed": bool(resource_caps["passed"]),
        "passed": row_integrity and profile_integrity and bool(resource_caps["passed"]),
    }
    base = {
        "schema_version": 1,
        "evidence_class": INT8_EVIDENCE_CLASS,
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
        "integrity": integrity,
    }
    if not integrity["passed"]:
        unsigned = {
            **base,
            "disposition": "invalid_integrity",
            "decision": None,
            "reason": "zero_tolerance_or_resource_cap_gate",
        }
        return {**unsigned, "input_sha256": canonical_sha256(unsigned)}
    scores = {
        arm: _candidate_anchor_scores(rows, candidate.player_id, anchor_ids)
        for arm, candidate in candidate_by_arm.items()
    }
    if not scores["uniform"] or any(
        set(arm_scores) != set(scores["uniform"]) for arm_scores in scores.values()
    ):
        raise ArenaError("paired arena score cells are incomplete")
    paired_delta = {
        arm: float(
            np.mean(
                [
                    scores[arm][key] - scores["uniform"][key]
                    for key in sorted(scores["uniform"])
                ]
            )
        )
        for arm in ("chosen", "visit")
    }
    competency_counts = {
        arm: _competency_correct_count(competencies, candidate.player_id)
        for arm, candidate in candidate_by_arm.items()
    }
    agreements = mechanism["selected_command_agreement"]
    uniform_agreement = agreements["uniform32_vs_uniform128"]["aggregate"]["rate"]
    label_improvement = {
        "chosen": agreements["chosen32_vs_uniform128"]["aggregate"]["rate"]
        - uniform_agreement,
        "visit": agreements["visit32_vs_uniform128"]["aggregate"]["rate"]
        - uniform_agreement,
    }
    metrics = mechanism["player_metrics"]
    uniform_cost = metrics["uniform-prior-puct-32-v1"]
    clauses = {}
    for arm, other in (("chosen", "visit"), ("visit", "chosen")):
        candidate_id = candidate_by_arm[arm].player_id
        p95_ratio = float(metrics[candidate_id]["p95_seconds"]) / float(
            uniform_cost["p95_seconds"]
        )
        nodes_ratio = float(metrics[candidate_id]["nodes_per_second"]) / float(
            uniform_cost["nodes_per_second"]
        )
        arm_clauses = {
            "arena_delta": {
                "actual": paired_delta[arm],
                "minimum": 0.05,
                "passed": paired_delta[arm] >= 0.05,
            },
            "arena_separation": {
                "actual": paired_delta[arm] - paired_delta[other],
                "minimum": 0.05,
                "passed": paired_delta[arm] - paired_delta[other] >= 0.05,
            },
            "high_budget_label_agreement_improvement": {
                "actual": label_improvement[arm],
                "minimum": 0.05,
                "passed": label_improvement[arm] >= 0.05,
            },
            "competency_noninferiority": {
                "actual": competency_counts[arm] - competency_counts[other],
                "minimum": 0,
                "passed": competency_counts[arm] >= competency_counts[other],
            },
            "p95_latency_ratio": {
                "actual": p95_ratio,
                "maximum": 1.10,
                "passed": p95_ratio <= 1.10,
            },
            "nodes_per_second_ratio": {
                "actual": nodes_ratio,
                "minimum": 0.90,
                "passed": nodes_ratio >= 0.90,
            },
        }
        clauses[arm] = {
            "clauses": arm_clauses,
            "passed": all(clause["passed"] for clause in arm_clauses.values()),
        }
    clearing = [arm for arm in ("chosen", "visit") if clauses[arm]["passed"]]
    decision = (
        f"next_corpus_{'chosen_action' if clearing == ['chosen'] else 'visit_distribution'}"
        if len(clearing) == 1
        else "kill_retained_smoke_policy_guidance"
    )
    unsigned = {
        **base,
        "disposition": INT8_EVIDENCE_CLASS,
        "decision": decision,
        "reason": "exactly_one_signal_cleared"
        if len(clearing) == 1
        else "zero_or_two_signals_cleared",
        "paired_arena_delta": paired_delta,
        "high_budget_label_agreement_improvement": label_improvement,
        "competency_correct_counts": competency_counts,
        "arms": clauses,
    }
    return {**unsigned, "input_sha256": canonical_sha256(unsigned)}


def verify_manifest(
    out_dir: Path,
    contract: ArenaContract,
    contract_sha: str,
    *,
    expected_runtime: dict[str, Any] | None = None,
    experiment_contract_sha256: str | None = None,
    int8_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_json(out_dir / "manifest.json")
    artifact_kind = manifest.get("kind")
    if artifact_kind not in {
        "anchor-freeze",
        "challenge",
        "guidance-diagnostic",
    }:
        raise ArenaError("manifest artifact kind is unsupported")
    if artifact_kind == "guidance-diagnostic" and (
        manifest.get("evidence_class") != INT8_EVIDENCE_CLASS
        or manifest.get("comparison_seed_alias") != INT8_COMPARISON_ALIAS
    ):
        raise ArenaError("diagnostic evidence class or comparison alias mismatch")
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise ArenaError("manifest digest mismatch")
    if manifest.get("contract_sha256") != contract_sha:
        raise ArenaError("manifest contract mismatch")
    if manifest.get("arena_key") != contract.key.model_dump():
        raise ArenaError("manifest arena key mismatch")
    runtime = contract.runtime if expected_runtime is None else expected_runtime
    if manifest.get("runtime") != runtime:
        raise ArenaError("manifest runtime identity mismatch")
    if experiment_contract_sha256 is not None and (
        manifest.get("experiment_contract_sha256") != experiment_contract_sha256
        or manifest.get("int8_authority") != int8_authority
        or manifest.get("evidence_class") != INT8_EVIDENCE_CLASS
    ):
        raise ArenaError("INT-8 additive authority binding mismatch")
    profile_name = str(manifest.get("profile"))
    if profile_name not in contract.profiles:
        raise ArenaError("manifest profile mismatch")
    schedule = contract.schedules[profile_name]
    for receipt in manifest["artifacts"].values():
        path = out_dir / receipt["path"]
        if not path.is_file() or file_sha256(path) != receipt["sha256"]:
            raise ArenaError(f"artifact digest mismatch: {path}")
    ledger = verify_ledger(out_dir / "resource-ledger.jsonl")
    replay_receipts = []
    traced_games: dict[str, str] = {}
    source_profile_games: list[dict[str, Any]] | None = None
    for trace in manifest["traces"]:
        path = out_dir / trace.get("artifact_path", trace["path"])
        if not path.is_file() or file_sha256(path) != trace["sha256"]:
            raise ArenaError(f"trace digest mismatch: {path}")
        games = read_trace(path)
        if path.name == "random-v1__scripted-greedy-v1.commands.jsonl.gz":
            source_profile_games = games
        if len(games) != trace["games"]:
            raise ArenaError(f"trace game-count mismatch: {path}")
        replay_receipts.append(replay_games(games).to_dict())
        for game in games:
            digest = str(game["game_trace_sha256"])
            if digest in traced_games:
                raise ArenaError("duplicate game trace identity")
            traced_games[digest] = str(trace["sha256"])
    if replay_receipts and not all(receipt["passed"] for receipt in replay_receipts):
        raise ArenaError("Command replay mismatch")
    rows = read_jsonl(out_dir / "matches.jsonl")
    try:
        validated_rows = [MatchRow.model_validate(row) for row in rows]
    except ValueError as error:
        raise ArenaError(f"match row identity mismatch: {error}") from error
    expected_cells = {
        "anchor-freeze": 10,
        "challenge": 15,
        "guidance-diagnostic": 28,
    }[artifact_kind]
    expected_games = expected_cells * len(schedule.deal_seeds) * 2
    if len(rows) != expected_games:
        raise ArenaError("match schedule game-count mismatch")
    if len({row.cell_id for row in validated_rows}) != expected_cells:
        raise ArenaError("match schedule cell-count mismatch")
    if {row.deal_seed for row in validated_rows} != set(schedule.deal_seeds):
        raise ArenaError("match schedule deal-seed mismatch")
    row_keys = {(row.cell_id, row.deal_block, row.leg) for row in validated_rows}
    if len(row_keys) != expected_games:
        raise ArenaError("match schedule contains duplicate or missing seat legs")
    if any(
        row.deal_block >= len(schedule.deal_seeds)
        or schedule.deal_seeds[row.deal_block] != row.deal_seed
        for row in validated_rows
    ):
        raise ArenaError("match deal-block mapping mismatch")
    expected_seed_set = canonical_sha256(list(schedule.deal_seeds))
    if any(row.deal_seed_set_sha256 != expected_seed_set for row in validated_rows):
        raise ArenaError("match row schedule identity mismatch")
    local_rows = [
        row for row in validated_rows if row.game_trace_sha256 in traced_games
    ]
    expected_local_cells = {
        "anchor-freeze": 10,
        "challenge": 5,
        "guidance-diagnostic": 18,
    }[artifact_kind]
    if len(local_rows) != expected_local_cells * len(schedule.deal_seeds) * 2:
        raise ArenaError("local trace-to-match coverage mismatch")
    if any(
        row.trace_shard_sha256 != traced_games[row.game_trace_sha256]
        for row in local_rows
    ):
        raise ArenaError("match row trace-shard binding mismatch")
    players = [
        PlayerRegistration.model_validate(player)
        for player in load_json(out_dir / "players.json")
    ]
    expected_players = list(contract.anchors)
    if artifact_kind == "challenge":
        expected_players.append(
            PlayerRegistration.model_validate(manifest["candidate"])
        )
    diagnostic_candidates = (
        [
            PlayerRegistration.model_validate(candidate)
            for candidate in manifest.get("candidates", [])
        ]
        if artifact_kind == "guidance-diagnostic"
        else []
    )
    profile_variants = (
        [
            PlayerRegistration.model_validate(variant)
            for variant in manifest.get("profile_variants", [])
        ]
        if artifact_kind == "guidance-diagnostic"
        else []
    )
    if artifact_kind == "guidance-diagnostic":
        if (
            len(diagnostic_candidates) != 3
            or [_profile_arm(candidate) for candidate in diagnostic_candidates]
            != ["uniform", "chosen", "visit"]
            or profile_variants != diagnostic_profile_variants(diagnostic_candidates)
        ):
            raise ArenaError("diagnostic candidate registry mismatch")
        expected_players.extend(diagnostic_candidates)
    if players != expected_players:
        raise ArenaError("player registry artifact mismatch")
    candidate = (
        PlayerRegistration.model_validate(manifest["candidate"])
        if artifact_kind == "challenge"
        else None
    )
    registration_by_id = {
        registration.player_id: registration for registration in expected_players
    }
    if any(
        row.player_a_registration_sha256
        != registration_by_id[row.player_a].identity_sha256
        or row.player_b_registration_sha256
        != registration_by_id[row.player_b].identity_sha256
        for row in validated_rows
    ):
        raise ArenaError("match row registration binding mismatch")
    _, recomputed_rating = rating_payload(rows, schedule, contract)
    stored_rating = load_json(out_dir / "rating.json")
    if recomputed_rating != stored_rating:
        raise ArenaError("rating recomputation mismatch")
    if payoff_matrix(rows) != load_json(out_dir / "payoff-matrix.json"):
        raise ArenaError("payoff matrix/residual recomputation mismatch")
    competencies = load_json(out_dir / "competencies.json")
    if (
        competencies.get("scenario_seeds") != list(schedule.competency_seeds)
        or competencies.get("scenario_seed_set_sha256")
        != canonical_sha256(list(schedule.competency_seeds))
        or set(competencies["players"])
        != {player.player_id for player in expected_players}
    ):
        raise ArenaError("competency schedule or player identity mismatch")
    for player_id, scenarios in competencies["players"].items():
        if set(scenarios) != set(SCENARIOS):
            raise ArenaError(f"competency scenario identity mismatch: {player_id}")
        for scenario_name, evidence in scenarios.items():
            if [run["run_seed"] for run in evidence["runs"]] != list(
                schedule.competency_seeds
            ):
                raise ArenaError(
                    f"competency seed-row mismatch: {player_id}/{scenario_name}"
                )
            if aggregate_scenario_results(evidence["runs"]) != evidence["aggregate"]:
                raise ArenaError(
                    f"competency aggregate mismatch: {player_id}/{scenario_name}"
                )
    profile_payload = load_json(out_dir / "profile.json")
    if artifact_kind == "anchor-freeze":
        native_rows = rows
    elif artifact_kind == "challenge":
        native_rows = [
            row
            for row in rows
            if manifest["candidate"]["player_id"] in {row["player_a"], row["player_b"]}
        ]
    else:
        diagnostic_ids = {candidate.player_id for candidate in diagnostic_candidates}
        native_rows = [
            row
            for row in rows
            if diagnostic_ids.intersection({row["player_a"], row["player_b"]})
        ]
    if profile_payload["native_gameplay"] != native_gameplay_profiles(
        native_rows, worker_count=contract.resource_caps.outcome_workers
    ):
        raise ArenaError("native gameplay cost recomputation mismatch")
    verify_profile(profile_payload["matched_root"])
    if artifact_kind == "anchor-freeze":
        if source_profile_games is None:
            raise ArenaError("anchor artifact has no matched-root source trace")
        expected_roots = select_profile_roots(
            source_profile_games,
            warmup=contract.profile_roots.warmup,
            measured=contract.profile_roots.measured,
        )
        if profile_payload["matched_root"]["root_ids"] != [
            root["root_id"] for root in expected_roots
        ]:
            raise ArenaError("matched-root selection recomputation mismatch")
    if artifact_kind == "anchor-freeze":
        expected_profile_players = {player.player_id for player in contract.anchors}
    elif artifact_kind == "challenge":
        expected_profile_players = {candidate.player_id}
    else:
        expected_profile_players = {variant.player_id for variant in profile_variants}
    if set(profile_payload["matched_root"]["players"]) != expected_profile_players:
        raise ArenaError("matched-root profile player identity mismatch")
    resource_receipt = profile_payload["resource_caps"]
    if not resource_receipt.get("passed") or not all(
        clause["passed"] and float(clause["actual"]) <= float(clause["maximum"])
        for clause in resource_receipt["clauses"].values()
    ):
        raise ArenaError("resource cap receipt mismatch")
    if artifact_bytes(out_dir) > contract.resource_caps.artifact_bytes:
        raise ArenaError("artifact resource cap exceeded")
    if artifact_kind == "guidance-diagnostic":
        expected_cap_maxima = {
            "wall_hours": 2.0,
            "core_hours_conservative": 8.0,
            "artifact_bytes_at_decision": 1073741824.0,
        }
        actual_cap_maxima = {
            name: float(clause["maximum"])
            for name, clause in resource_receipt["clauses"].items()
        }
        if actual_cap_maxima != expected_cap_maxima:
            raise ArenaError("diagnostic resource-cap identity mismatch")
        mechanism = guidance_mechanism_payload(
            profile_payload["matched_root"], profile_variants
        )
        if mechanism != load_json(out_dir / "mechanism.json"):
            raise ArenaError("diagnostic mechanism recomputation mismatch")
        diagnostic_ids = {
            registration.player_id for registration in diagnostic_candidates
        }
        diagnostic_rows = [
            row
            for row in rows
            if row["player_a"] in diagnostic_ids or row["player_b"] in diagnostic_ids
        ]
        decision = diagnostic_decision_payload(
            candidates=diagnostic_candidates,
            rows=diagnostic_rows,
            competencies=competencies,
            profile=profile_payload["matched_root"],
            mechanism=mechanism,
            resource_caps=resource_receipt,
            anchor_ids={anchor.player_id for anchor in contract.anchors},
        )
        if decision != load_json(out_dir / "decision.json"):
            raise ArenaError("diagnostic decision recomputation mismatch")
        if (
            decision.get("evidence_class") != INT8_EVIDENCE_CLASS
            or decision.get("promotion_eligible") is not False
            or decision.get("admission_eligible") is not False
            or decision.get("method_level_claim") is not False
        ):
            raise ArenaError("diagnostic non-admission boundary mismatch")
    else:
        disposition_rows = (
            [
                row
                for row in rows
                if candidate.player_id in {row["player_a"], row["player_b"]}
            ]
            if candidate is not None
            else rows
        )
        recomputed_promotion = promotion_payload(
            candidate,
            disposition_rows,
            smoke=contract.profiles[profile_name].disposition != "production",
            profile=profile_payload,
            competencies=competencies,
            contract=contract,
        )
        if recomputed_promotion != load_json(out_dir / "promotion.json"):
            raise ArenaError("promotion receipt recomputation mismatch")
    return {
        "verified": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "games": len(rows),
        "replay": replay_receipts,
        "ledger": ledger,
    }


def freeze_anchors(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    experiment_contract_sha = None
    int8_authority = None
    if getattr(args, "experiment_contract", None) is None:
        contract, contract_sha, runtime = preflight_contract(args.contract)
    else:
        (
            contract,
            contract_sha,
            runtime,
            _,
            experiment_contract_sha,
            int8_authority,
            _,
            _,
        ) = preflight_int8_dependencies(args, validate_checkpoint_load=True)
    profile = contract.profiles[args.profile]
    schedule = contract.schedules[args.profile]
    if args.out_dir.exists():
        raise ArenaError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    append_ledger(
        args.out_dir,
        "preflight",
        {"contract_sha256": contract_sha, "profile": args.profile},
    )
    rows: list[dict[str, Any]] = []
    traces = []
    replays = []
    pairs = list(combinations(contract.anchors, 2))
    for cell_rows, trace, replay in play_cells(
        contract=contract,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=args.out_dir,
    ):
        rows.extend(cell_rows)
        traces.append(trace)
        replays.append(replay)
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(
        args.out_dir / "players.json",
        [player.model_dump() for player in contract.anchors],
    )
    competencies = run_competencies(
        list(contract.anchors), seeds=schedule.competency_seeds
    )
    write_json(args.out_dir / "competencies.json", competencies)
    fit, rating = rating_payload(rows, schedule, contract)
    matrix = payoff_matrix(rows)
    source_trace = next(
        trace
        for trace in traces
        if Path(trace["path"]).name == "random-v1__scripted-greedy-v1.commands.jsonl.gz"
    )
    source_games = read_trace(args.out_dir / source_trace["path"])
    profile_payload = {
        "native_gameplay": native_gameplay_profiles(
            rows, worker_count=contract.resource_caps.outcome_workers
        ),
        "matched_root": profile_players(
            list(contract.anchors),
            source_games=source_games,
            profile_roots=contract.profile_roots,
        ),
    }
    profile_payload["resource_caps"] = resource_cap_receipt(
        args.out_dir, contract, started=started
    )
    promotion = promotion_payload(
        None,
        rows,
        smoke=profile.disposition != "production",
        profile=profile_payload,
        competencies=competencies,
        contract=contract,
    )
    replay_payload = {
        "passed": all(item["passed"] for item in replays),
        "cells": replays,
    }
    write_json(args.out_dir / "replay.json", replay_payload)
    write_json(args.out_dir / "rating.json", rating)
    write_json(args.out_dir / "payoff-matrix.json", matrix)
    write_json(args.out_dir / "profile.json", profile_payload)
    write_json(args.out_dir / "promotion.json", promotion)
    (args.out_dir / "report.md").write_text(
        report_markdown(
            fit=fit, promotion=promotion, matrix=matrix, replay=replay_payload
        )
    )
    append_ledger(args.out_dir, "complete", {"games": len(rows), "cells": len(traces)})
    names = [
        "resource-ledger.jsonl",
        "players.json",
        "matches.jsonl",
        "replay.json",
        "competencies.json",
        "profile.json",
        "rating.json",
        "payoff-matrix.json",
        "promotion.json",
        "report.md",
    ]
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "anchor-freeze",
            "profile": args.profile,
            "contract_sha256": contract_sha,
            "arena_key": contract.key.model_dump(),
            "runtime": runtime,
            "traces": portable_trace_receipts(traces),
            "artifacts": artifact_receipts(args.out_dir, names),
            **(
                {
                    "evidence_class": INT8_EVIDENCE_CLASS,
                    "experiment_contract_sha256": experiment_contract_sha,
                    "int8_authority": int8_authority,
                }
                if experiment_contract_sha is not None
                else {}
            ),
        },
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "manifest": str(args.out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "disposition": promotion["disposition"],
            },
            sort_keys=True,
        )
    )


def challenge(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    contract, contract_sha, runtime = preflight_contract(args.contract)
    retained_checkpoint = None
    candidate_raw = PlayerRegistration.model_validate(load_json(args.candidate))
    if args.verify and candidate_raw.runner_kind == "checkpoint":
        retained_checkpoint = (
            args.out_dir / "checkpoints" / f"{candidate_raw.player_id}.pt"
        )
    checkpoint_argument = args.candidate_checkpoint or retained_checkpoint
    candidate, checkpoint_paths = preflight_candidate(
        args.candidate,
        checkpoint_argument,
        contract=contract,
        profile_name=args.profile,
        validate_checkpoint_load=not args.verify,
    )
    if args.anchor_artifact.name != "manifest.json":
        raise ArenaError("--anchor-artifact must name the frozen manifest.json")
    anchor_dir = args.anchor_artifact.parent
    anchor_verification = verify_manifest(anchor_dir, contract, contract_sha)
    anchor_manifest = load_json(args.anchor_artifact)
    if anchor_manifest["kind"] != "anchor-freeze":
        raise ArenaError("challenge requires an anchor-freeze artifact")
    if anchor_manifest["profile"] != args.profile:
        raise ArenaError("challenge and anchor artifact profiles differ")
    if args.verify:
        verification = verify_manifest(args.out_dir, contract, contract_sha)
        challenge_manifest = load_json(args.out_dir / "manifest.json")
        if (
            challenge_manifest.get("anchor_manifest_sha256")
            != anchor_verification["manifest_sha256"]
        ):
            raise ArenaError("challenge anchor binding mismatch")
        anchor_corpus = load_json(anchor_dir / "profile.json")["matched_root"][
            "root_corpus_sha256"
        ]
        challenge_corpus = load_json(args.out_dir / "profile.json")["matched_root"][
            "root_corpus_sha256"
        ]
        if challenge_corpus != anchor_corpus:
            raise ArenaError("challenge matched-root corpus drift")
        print(json.dumps(verification, sort_keys=True))
        return
    profile = contract.profiles[args.profile]
    schedule = contract.schedules[args.profile]
    if args.out_dir.exists():
        raise ArenaError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    if candidate.runner_kind == "checkpoint":
        retained = args.out_dir / "checkpoints" / f"{candidate.player_id}.pt"
        retained.parent.mkdir(parents=True)
        shutil.copyfile(checkpoint_argument, retained)
        checkpoint_paths[candidate.player_id] = str(retained)
    append_ledger(
        args.out_dir,
        "preflight",
        {
            "contract_sha256": contract_sha,
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidate_sha256": candidate.identity_sha256,
        },
    )
    new_rows: list[dict[str, Any]] = []
    traces = []
    replays = []
    pairs = [(candidate, anchor) for anchor in contract.anchors]
    for cell_rows, trace, replay in play_cells(
        contract=contract,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=args.out_dir,
        checkpoint_paths=checkpoint_paths,
    ):
        new_rows.extend(cell_rows)
        traces.append(trace)
        replays.append(replay)
    anchor_rows = read_jsonl(anchor_dir / "matches.jsonl")
    rows = anchor_rows + new_rows
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(
        args.out_dir / "players.json",
        [player.model_dump() for player in (*contract.anchors, candidate)],
    )
    anchor_competencies = load_json(anchor_dir / "competencies.json")
    candidate_competencies = run_competencies(
        [candidate], seeds=schedule.competency_seeds, checkpoint_paths=checkpoint_paths
    )
    competencies = dict(anchor_competencies)
    competencies["players"] = dict(anchor_competencies["players"])
    competencies["players"].update(candidate_competencies["players"])
    write_json(args.out_dir / "competencies.json", competencies)
    fit, rating = rating_payload(rows, schedule, contract)
    matrix = payoff_matrix(rows)
    source_trace = (
        anchor_dir / "traces" / ("random-v1__scripted-greedy-v1.commands.jsonl.gz")
    )
    profile_payload = {
        "native_gameplay": native_gameplay_profiles(
            new_rows, worker_count=contract.resource_caps.outcome_workers
        ),
        "matched_root": profile_players(
            [candidate],
            source_games=read_trace(source_trace),
            profile_roots=contract.profile_roots,
            checkpoint_paths=checkpoint_paths,
        ),
    }
    profile_payload["resource_caps"] = resource_cap_receipt(
        args.out_dir, contract, started=started
    )
    promotion = promotion_payload(
        candidate,
        new_rows,
        smoke=profile.disposition != "production",
        profile=profile_payload,
        competencies=competencies,
        contract=contract,
    )
    replay_payload = {
        "passed": all(item["passed"] for item in replays),
        "cells": replays,
        "anchor_verification": anchor_verification,
    }
    write_json(args.out_dir / "replay.json", replay_payload)
    write_json(args.out_dir / "rating.json", rating)
    write_json(args.out_dir / "payoff-matrix.json", matrix)
    write_json(args.out_dir / "profile.json", profile_payload)
    write_json(args.out_dir / "promotion.json", promotion)
    (args.out_dir / "report.md").write_text(
        report_markdown(
            fit=fit, promotion=promotion, matrix=matrix, replay=replay_payload
        )
    )
    append_ledger(
        args.out_dir,
        "complete",
        {"new_games": len(new_rows), "combined_games": len(rows)},
    )
    names = [
        "resource-ledger.jsonl",
        "players.json",
        "matches.jsonl",
        "replay.json",
        "competencies.json",
        "profile.json",
        "rating.json",
        "payoff-matrix.json",
        "promotion.json",
        "report.md",
    ]
    if candidate.runner_kind == "checkpoint":
        names.append(f"checkpoints/{candidate.player_id}.pt")
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "challenge",
            "profile": args.profile,
            "contract_sha256": contract_sha,
            "arena_key": contract.key.model_dump(),
            "runtime": runtime,
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidate": candidate.model_dump(),
            "traces": portable_trace_receipts(traces),
            "artifacts": artifact_receipts(args.out_dir, names),
        },
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "manifest": str(args.out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "candidate_elo": fit.ratings[candidate.player_id],
                "disposition": promotion["disposition"],
                "reason": promotion.get("reason"),
            },
            sort_keys=True,
        )
    )


def preflight_diagnostic_contract(
    path: Path,
    *,
    arena_contract: ArenaContract,
    arena_contract_sha256: str,
    candidates: list[PlayerRegistration],
    candidate_paths: list[Path],
    input_compatibility: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    contract = load_json(path)
    current_runtime = arena_runtime_fingerprints()
    current_implementation = {
        "source_paths": list(INT8_IMPLEMENTATION_SOURCE_PATHS),
        "source_sha256": source_digest(INT8_IMPLEMENTATION_SOURCE_PATHS),
    }
    current_modules = {
        name: {"path": module_path, "sha256": file_sha256(REPO_ROOT / module_path)}
        for name, module_path in INT8_MODULE_PATHS.items()
    }
    expected_arena = {
        "contract_file_sha256": arena_contract_sha256,
        "arena_key_sha256": canonical_sha256(arena_contract.key.model_dump()),
        "anchor_cohort_sha256": arena_contract.key.anchor_cohort_sha256,
        "rating_prior_sha256": arena_contract.key.rating_prior_sha256,
        "anchor_registration_identity_sha256": {
            anchor.player_id: anchor.identity_sha256
            for anchor in arena_contract.anchors
        },
        "profile": "smoke",
        "deal_seeds": list(arena_contract.schedules["smoke"].deal_seeds),
        "competency_seeds": list(arena_contract.schedules["smoke"].competency_seeds),
    }
    expected_candidates = {
        candidate.player_id: candidate.identity_sha256 for candidate in candidates
    }
    expected_candidate_files = {
        candidate.player_id: file_sha256(candidate_path)
        for candidate, candidate_path in zip(candidates, candidate_paths, strict=True)
    }
    expected_input = {
        "input_manifest_sha256": input_compatibility["input_manifest_sha256"],
        "payload_sha256": input_compatibility["payload_sha256"],
        "contract_file_sha256": input_compatibility["contract_file_sha256"],
        "contract_canonical_sha256": input_compatibility["contract_canonical_sha256"],
        "loader_source_sha256": input_compatibility["loader_source_sha256"],
    }
    checks = {
        "schema_version": (contract.get("schema_version"), 1),
        "experiment": (
            contract.get("experiment"),
            "int-8-student-signal-guidance-v1",
        ),
        "evidence_class": (contract.get("evidence_class"), INT8_EVIDENCE_CLASS),
        "arena": (contract.get("arena"), expected_arena),
        "input": (contract.get("input"), expected_input),
        "candidates": (contract.get("candidates"), expected_candidates),
        "candidate_files": (
            contract.get("candidate_files"),
            expected_candidate_files,
        ),
        "current_runtime": (contract.get("current_runtime"), current_runtime),
        "current_arena_implementation": (
            contract.get("current_arena_implementation"),
            current_implementation,
        ),
        "modules": (contract.get("modules"), current_modules),
        "factors": (
            contract.get("factors"),
            {
                "world": "w2",
                "content_suite": "w2-interactive-mirror-v1",
                "information_boundary": "acting-viewer-history-only-v1",
                "priors": [
                    "uniform-v1",
                    "chosen-policy-only-9004b87e2be4a893",
                    "visit-policy-only-c2c8dcec02dbcf19",
                ],
                "leaf_evaluator": "uniform-random-terminal-v1",
                "root_noise": "none",
                "worlds": 4,
                "c_puct": 1.5,
                "max_steps": 2000,
                "branch_driver_id": "full_clone/current_game_v1",
                "matched_root_traversals": [8, 32, 128],
                "arena_traversals": 32,
                "comparison_seed_alias": INT8_COMPARISON_ALIAS,
                "training": "none",
            },
        ),
        "decision_thresholds": (
            contract.get("decision_thresholds"),
            {
                "paired_arena_delta_minimum": 0.05,
                "other_signal_separation_minimum": 0.05,
                "high_budget_label_agreement_improvement_minimum": 0.05,
                "competency_correct_count_difference_minimum": 0,
                "p95_latency_ratio_maximum": 1.1,
                "nodes_per_second_ratio_minimum": 0.9,
                "ambiguous_result": "kill_retained_smoke_policy_guidance",
            },
        ),
        "resource_caps": (
            contract.get("resource_caps"),
            {
                "wall_hours": 2.0,
                "core_hours": 8.0,
                "artifact_bytes": 1073741824,
                "workers": 4,
            },
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ArenaError(
            "INT-8 diagnostic contract drift: " + json.dumps(mismatches, sort_keys=True)
        )
    orchestration = contract.get("orchestration_source", {})
    orchestration_path = REPO_ROOT / str(orchestration.get("path", ""))
    if (
        orchestration.get("path")
        != "experiments/runners/run_int8_student_signal_guidance.py"
        or not orchestration_path.is_file()
        or orchestration.get("sha256") != file_sha256(orchestration_path)
    ):
        raise ArenaError("INT-8 orchestration source identity drift")
    int8_authority = {
        "frozen_int6_contract_sha256": arena_contract_sha256,
        "current_arena_implementation": current_implementation,
        "modules": current_modules,
        "candidate_identity_sha256": expected_candidates,
        "candidate_file_sha256": expected_candidate_files,
    }
    return contract, file_sha256(path), current_runtime, int8_authority


def preflight_int8_dependencies(
    args: argparse.Namespace,
    *,
    validate_checkpoint_load: bool,
) -> tuple[
    ArenaContract,
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    list[PlayerRegistration],
    dict[str, str],
]:
    """Validate INT-8 without asking the frozen INT-6 receipt to describe new code."""

    contract = ArenaContract.model_validate(load_json(args.contract))
    contract_sha = file_sha256(args.contract)
    input_compatibility = load_json(args.input_compatibility)
    if (
        input_compatibility.get("status") != "compatible"
        or input_compatibility.get("evidence_class") != INT8_EVIDENCE_CLASS
        or any(
            input_compatibility.get(field)
            for field in (
                "conversion_performed",
                "rewriting_performed",
                "retraining_performed",
                "substitution_performed",
            )
        )
    ):
        raise ArenaError("INT-8 retained-input compatibility receipt is invalid")
    candidate_arguments = [
        (args.uniform_candidate, None),
        (args.chosen_candidate, args.chosen_checkpoint),
        (args.visit_candidate, args.visit_checkpoint),
    ]
    candidates = []
    checkpoint_paths: dict[str, str] = {}
    for candidate_path, checkpoint_path in candidate_arguments:
        candidate, paths = preflight_candidate(
            candidate_path,
            checkpoint_path,
            contract=contract,
            profile_name="smoke",
            validate_checkpoint_load=validate_checkpoint_load,
        )
        candidates.append(candidate)
        checkpoint_paths.update(paths)
    if [_profile_arm(candidate) for candidate in candidates] != [
        "uniform",
        "chosen",
        "visit",
    ]:
        raise ArenaError("INT-8 candidate ordering or identities drifted")
    (
        _,
        experiment_contract_sha,
        runtime,
        int8_authority,
    ) = preflight_diagnostic_contract(
        args.experiment_contract,
        arena_contract=contract,
        arena_contract_sha256=contract_sha,
        candidates=candidates,
        candidate_paths=[path for path, _ in candidate_arguments],
        input_compatibility=input_compatibility,
    )
    return (
        contract,
        contract_sha,
        runtime,
        input_compatibility,
        experiment_contract_sha,
        int8_authority,
        candidates,
        checkpoint_paths,
    )


def diagnostic(args: argparse.Namespace) -> None:
    started = getattr(args, "task_started", time.perf_counter())
    if args.profile != "smoke":
        raise ArenaError("INT-8 diagnostic is smoke-only")
    (
        contract,
        contract_sha,
        runtime,
        input_compatibility,
        experiment_contract_sha,
        int8_authority,
        candidates,
        source_checkpoint_paths,
    ) = preflight_int8_dependencies(args, validate_checkpoint_load=not args.verify)
    profile_variants = diagnostic_profile_variants(candidates)
    experiment_contract = load_json(args.experiment_contract)
    if args.anchor_artifact.name != "manifest.json":
        raise ArenaError("--anchor-artifact must name the frozen manifest.json")
    anchor_dir = args.anchor_artifact.parent
    anchor_verification = verify_manifest(
        anchor_dir,
        contract,
        contract_sha,
        expected_runtime=runtime,
        experiment_contract_sha256=experiment_contract_sha,
        int8_authority=int8_authority,
    )
    anchor_manifest = load_json(args.anchor_artifact)
    if (
        anchor_manifest.get("kind") != "anchor-freeze"
        or anchor_manifest.get("profile") != "smoke"
    ):
        raise ArenaError("INT-8 requires one smoke anchor-freeze artifact")
    if args.verify:
        verification = verify_manifest(
            args.out_dir,
            contract,
            contract_sha,
            expected_runtime=runtime,
            experiment_contract_sha256=experiment_contract_sha,
            int8_authority=int8_authority,
        )
        manifest = load_json(args.out_dir / "manifest.json")
        if (
            manifest.get("anchor_manifest_sha256")
            != anchor_verification["manifest_sha256"]
            or manifest.get("experiment_contract_sha256") != experiment_contract_sha
        ):
            raise ArenaError("diagnostic dependency binding mismatch")
        anchor_corpus = load_json(anchor_dir / "profile.json")["matched_root"][
            "root_corpus_sha256"
        ]
        diagnostic_corpus = load_json(args.out_dir / "profile.json")["matched_root"]
        if diagnostic_corpus["root_corpus_sha256"] != anchor_corpus:
            raise ArenaError("diagnostic matched-root corpus drift")
        print(json.dumps(verification, sort_keys=True))
        return
    if args.out_dir.exists():
        raise ArenaError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    checkpoint_paths: dict[str, str] = {}
    for candidate in candidates:
        if candidate.runner_kind != "checkpoint":
            continue
        source = Path(source_checkpoint_paths[candidate.player_id])
        retained = args.out_dir / "checkpoints" / f"{candidate.player_id}.pt"
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, retained)
        checkpoint_paths[candidate.player_id] = str(retained)
        arm = _profile_arm(candidate)
        for variant in profile_variants:
            if _profile_arm(variant) == arm:
                checkpoint_paths[variant.player_id] = str(retained)
    aliases = {
        registration.player_id: INT8_COMPARISON_ALIAS
        for registration in [*candidates, *profile_variants]
    }
    append_ledger(
        args.out_dir,
        "preflight",
        {
            "contract_sha256": contract_sha,
            "experiment_contract_sha256": experiment_contract_sha,
            "input_compatibility_sha256": file_sha256(args.input_compatibility),
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidate_sha256": {
                candidate.player_id: candidate.identity_sha256
                for candidate in candidates
            },
        },
    )
    schedule = contract.schedules["smoke"]
    new_rows: list[dict[str, Any]] = []
    traces = []
    replays = []
    pairs = [
        *(
            (candidate, anchor)
            for candidate in candidates
            for anchor in contract.anchors
        ),
        *combinations(candidates, 2),
    ]
    for cell_rows, trace, replay in play_cells(
        contract=contract,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=args.out_dir,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
    ):
        new_rows.extend(cell_rows)
        traces.append(trace)
        replays.append(replay)
    anchor_rows = read_jsonl(anchor_dir / "matches.jsonl")
    rows = anchor_rows + new_rows
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(
        args.out_dir / "players.json",
        [
            registration.model_dump()
            for registration in (*contract.anchors, *candidates)
        ],
    )
    anchor_competencies = load_json(anchor_dir / "competencies.json")
    candidate_competencies = run_competencies(
        candidates,
        seeds=schedule.competency_seeds,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
    )
    competencies = dict(anchor_competencies)
    competencies["players"] = dict(anchor_competencies["players"])
    competencies["players"].update(candidate_competencies["players"])
    write_json(args.out_dir / "competencies.json", competencies)
    fit, rating = rating_payload(rows, schedule, contract)
    matrix = payoff_matrix(rows)
    source_trace = anchor_dir / "traces/random-v1__scripted-greedy-v1.commands.jsonl.gz"
    matched_root = profile_players(
        profile_variants,
        source_games=read_trace(source_trace),
        profile_roots=contract.profile_roots,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
    )
    profile_payload = {
        "native_gameplay": native_gameplay_profiles(
            new_rows, worker_count=contract.resource_caps.outcome_workers
        ),
        "matched_root": matched_root,
    }
    profile_payload["resource_caps"] = resource_cap_receipt(
        args.out_dir,
        contract,
        started=started,
        wall_hours_max=2.0,
        core_hours_max=8.0,
        artifact_bytes_max=1073741824,
        additional_artifact_dirs=(anchor_dir,),
    )
    mechanism = guidance_mechanism_payload(matched_root, profile_variants)
    decision = diagnostic_decision_payload(
        candidates=candidates,
        rows=new_rows,
        competencies=competencies,
        profile=matched_root,
        mechanism=mechanism,
        resource_caps=profile_payload["resource_caps"],
        anchor_ids={anchor.player_id for anchor in contract.anchors},
    )
    replay_payload = {
        "passed": all(item["passed"] for item in replays),
        "cells": replays,
        "anchor_verification": anchor_verification,
    }
    write_json(args.out_dir / "replay.json", replay_payload)
    write_json(args.out_dir / "rating.json", rating)
    write_json(args.out_dir / "payoff-matrix.json", matrix)
    write_json(args.out_dir / "profile.json", profile_payload)
    write_json(args.out_dir / "mechanism.json", mechanism)
    write_json(args.out_dir / "decision.json", decision)
    (args.out_dir / "report.md").write_text(
        report_markdown(
            fit=fit, promotion=decision, matrix=matrix, replay=replay_payload
        )
    )
    append_ledger(
        args.out_dir,
        "complete",
        {
            "new_games": len(new_rows),
            "combined_games": len(rows),
            "decision": decision["decision"],
        },
    )
    names = [
        "resource-ledger.jsonl",
        "players.json",
        "matches.jsonl",
        "replay.json",
        "competencies.json",
        "profile.json",
        "mechanism.json",
        "rating.json",
        "payoff-matrix.json",
        "decision.json",
        "report.md",
    ]
    names.extend(
        f"checkpoints/{candidate.player_id}.pt"
        for candidate in candidates
        if candidate.runner_kind == "checkpoint"
    )
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "guidance-diagnostic",
            "profile": "smoke",
            "evidence_class": INT8_EVIDENCE_CLASS,
            "contract_sha256": contract_sha,
            "experiment_contract_sha256": experiment_contract_sha,
            "input_compatibility_sha256": file_sha256(args.input_compatibility),
            "arena_key": contract.key.model_dump(),
            "runtime": runtime,
            "int8_authority": int8_authority,
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidates": [candidate.model_dump() for candidate in candidates],
            "profile_variants": [variant.model_dump() for variant in profile_variants],
            "comparison_seed_alias": INT8_COMPARISON_ALIAS,
            "traces": portable_trace_receipts(traces),
            "artifacts": artifact_receipts(args.out_dir, names),
        },
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "manifest": str(args.out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "evidence_class": INT8_EVIDENCE_CLASS,
                "decision": decision["decision"],
                "promotion_eligible": False,
                "admission_eligible": False,
                "contract_prediction": experiment_contract.get("prediction"),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-anchors")
    freeze.add_argument("--contract", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    freeze.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    challenge_parser = subparsers.add_parser("challenge")
    challenge_parser.add_argument("--contract", type=Path, required=True)
    challenge_parser.add_argument("--anchor-artifact", type=Path, required=True)
    challenge_parser.add_argument("--candidate", type=Path, required=True)
    challenge_parser.add_argument("--candidate-checkpoint", type=Path)
    challenge_parser.add_argument("--out-dir", type=Path, required=True)
    challenge_parser.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    challenge_parser.add_argument("--verify", action="store_true")
    diagnostic_parser = subparsers.add_parser("guidance-diagnostic")
    diagnostic_parser.add_argument("--contract", type=Path, required=True)
    diagnostic_parser.add_argument("--experiment-contract", type=Path, required=True)
    diagnostic_parser.add_argument("--input-compatibility", type=Path, required=True)
    diagnostic_parser.add_argument("--anchor-artifact", type=Path, required=True)
    diagnostic_parser.add_argument("--uniform-candidate", type=Path, required=True)
    diagnostic_parser.add_argument("--chosen-candidate", type=Path, required=True)
    diagnostic_parser.add_argument("--chosen-checkpoint", type=Path, required=True)
    diagnostic_parser.add_argument("--visit-candidate", type=Path, required=True)
    diagnostic_parser.add_argument("--visit-checkpoint", type=Path, required=True)
    diagnostic_parser.add_argument("--out-dir", type=Path, required=True)
    diagnostic_parser.add_argument("--profile", choices=("smoke",), default="smoke")
    diagnostic_parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "freeze-anchors":
            freeze_anchors(args)
        elif args.command == "challenge":
            challenge(args)
        else:
            diagnostic(args)
    except (ArenaError, ValueError, FileNotFoundError) as error:
        print(
            json.dumps(
                {
                    "state": "failed",
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
