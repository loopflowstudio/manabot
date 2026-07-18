#!/usr/bin/env python3
"""Train and compare matched INT-7 value-target students in arena v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from experiments.runners.run_skill_arena import (
    arena_runtime_fingerprints,
    rating_payload,
)
from manabot.arena.competency import run_competencies
from manabot.arena.int7_value_targets import (
    ARM_ORDER,
    EVIDENCE_CLASS,
    MODEL_SEEDS,
    RESOURCE_CAPS,
    CumulativeResourceLedger,
    Int7PlayerRegistration,
    ResourceCapExceeded,
    checkpoint_calibration,
    int7_player_source_sha256,
    mechanism_payload,
    play_int7_cells,
    profile_int7_players,
    run_int7_competencies,
    verify_resource_ledger,
)
from manabot.arena.int8_input import RetainedInputError, verify_retained_input
from manabot.arena.models import (
    ArenaContract,
    MatchRow,
    SearchSemantics,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.profile import native_gameplay_profiles, verify_profile
from manabot.arena.rating import payoff_matrix
from manabot.arena.replay import read_trace, replay_games
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import load_shards, save_bc_checkpoint, split_by_game
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.search_supervised import (
    BLENDED_VALUE_TARGET,
    ROOT_VALUE_TARGET,
    TERMINAL_OUTCOME_TARGET,
    VISIT_DISTRIBUTION_TARGET,
    train_search_supervised,
    value_targets_from_dataset,
)
from manabot.sim.teacher1_evidence import source_bundle_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONTRACT = (
    REPO_ROOT / "experiments/contracts/int-7-value-target-comparison-v1.json"
)
ARENA_CONTRACT = REPO_ROOT / "experiments/contracts/int-6-skill-arena-v1.json"
INPUT_MANIFEST = REPO_ROOT / (
    "experiments/data/int-8-retained-int-4-smoke-v1/sha256/"
    "13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0/"
    "input-manifest.json"
)
RETENTION_ROOT = REPO_ROOT / "experiments/data/int-7-value-target-comparison-v1/sha256"
EXPECTED_INT6_SHA256 = (
    "fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71"
)
EXPECTED_INPUT_SHA256 = (
    "cfba4f299d86b0db83556c783b9bf63cfb92094353e4c5ea2ac1950b4671a7ba"
)
EXPECTED_PAYLOAD_SHA256 = (
    "13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0"
)
INT7_SOURCE_PATHS = (
    "experiments/runners/run_int7_value_target_comparison.py",
    "manabot/arena/int7_value_targets.py",
    "manabot/sim/mcts.py",
    "manabot/sim/search_supervised.py",
)
CURRENT_ARENA_SOURCE_PATHS = (
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
ARM_SETTINGS = {
    "visit_policy_only": {
        "value_target": "none",
        "trainer_value_target": TERMINAL_OUTCOME_TARGET,
        "value_weight": 0.0,
        "value_mode": "neutral",
    },
    "visit_terminal": {
        "value_target": TERMINAL_OUTCOME_TARGET,
        "trainer_value_target": TERMINAL_OUTCOME_TARGET,
        "value_weight": 1.0,
        "value_mode": "learned",
    },
    "visit_teacher_root": {
        "value_target": ROOT_VALUE_TARGET,
        "trainer_value_target": ROOT_VALUE_TARGET,
        "value_weight": 1.0,
        "value_mode": "learned",
    },
    "visit_blend_50_50": {
        "value_target": BLENDED_VALUE_TARGET,
        "trainer_value_target": BLENDED_VALUE_TARGET,
        "value_weight": 1.0,
        "value_mode": "learned",
    },
}
FINAL_TIE_ORDER = {
    "visit_policy_only": 0,
    "visit_terminal": 1,
    "visit_blend_50_50": 2,
    "visit_teacher_root": 3,
}


class Int7Error(RuntimeError):
    """The closed INT-7 experiment or retained evidence failed validation."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_digest(paths: tuple[str, ...]) -> str:
    return source_bundle_sha256([REPO_ROOT / path for path in paths])


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


def _numpy_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tree_receipts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _tree_snapshot(root: Path) -> list[tuple[str, int, str]]:
    return [
        (row["path"], row["bytes"], row["sha256"]) for row in _tree_receipts(root)
    ] + [
        (
            "manifest.json",
            (root / "manifest.json").stat().st_size,
            file_sha256(root / "manifest.json"),
        )
    ]


def _preregistration_commit(contract_path: Path) -> str:
    relative = contract_path.resolve().relative_to(REPO_ROOT).as_posix()
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise Int7Error("INT-7 contract has no preregistration commit")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != contract_path.read_bytes():
        raise Int7Error("INT-7 contract bytes are not the committed preregistration")
    return commit


def preflight_contract(
    contract_path: Path = EXPERIMENT_CONTRACT,
    *,
    require_committed: bool = True,
) -> tuple[dict[str, Any], str, ArenaContract, dict[str, Any], str | None]:
    if file_sha256(ARENA_CONTRACT) != EXPECTED_INT6_SHA256:
        raise Int7Error("frozen INT-6 contract bytes changed")
    arena = ArenaContract.model_validate(load_json(ARENA_CONTRACT))
    contract = load_json(contract_path)
    contract_sha = file_sha256(contract_path)
    runtime = arena_runtime_fingerprints()
    input_receipt = verify_retained_input(INPUT_MANIFEST)
    expected_arena = {
        "contract_path": "experiments/contracts/int-6-skill-arena-v1.json",
        "contract_file_sha256": EXPECTED_INT6_SHA256,
        "arena_key_sha256": canonical_sha256(arena.key.model_dump()),
        "anchor_cohort_sha256": arena.key.anchor_cohort_sha256,
        "rating_prior_sha256": arena.key.rating_prior_sha256,
        "anchor_registration_identity_sha256": {
            row.player_id: row.identity_sha256 for row in arena.anchors
        },
        "profile": "smoke",
        "deal_seeds": list(arena.schedules["smoke"].deal_seeds),
        "competency_seeds": list(arena.schedules["smoke"].competency_seeds),
        "profile_roots": arena.profile_roots.model_dump(),
    }
    expected_input = {
        "path": INPUT_MANIFEST.relative_to(REPO_ROOT).as_posix(),
        "input_manifest_sha256": EXPECTED_INPUT_SHA256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "contract_file_sha256": input_receipt["contract_file_sha256"],
        "contract_canonical_sha256": input_receipt["contract_canonical_sha256"],
        "loader_source_sha256": input_receipt["loader_source_sha256"],
        "rows": 507,
        "games": 4,
        "trajectory_audit_sha256": (
            "ae03c3bda06bdd65b090fefcaf1e23bb717c6f6566cc08731092c7911770f14f"
        ),
    }
    expected_implementation = {
        "source_paths": list(INT7_SOURCE_PATHS),
        "source_sha256": source_digest(INT7_SOURCE_PATHS),
        "files": {path: file_sha256(REPO_ROOT / path) for path in INT7_SOURCE_PATHS},
        "player_source_sha256": int7_player_source_sha256(),
    }
    expected_arena_implementation = {
        "source_paths": list(CURRENT_ARENA_SOURCE_PATHS),
        "source_sha256": source_digest(CURRENT_ARENA_SOURCE_PATHS),
    }
    fixed = {
        "schema_version": 1,
        "experiment": "int-7-value-target-comparison-v1",
        "evidence_class": EVIDENCE_CLASS,
        "input": expected_input,
        "arena": expected_arena,
        "current_runtime": runtime,
        "implementation": expected_implementation,
        "current_arena_implementation": expected_arena_implementation,
        "factors": {
            "world": "w2",
            "content_suite": "w2-interactive-mirror-v1",
            "information_boundary": "acting-viewer-history-only-v1",
            "policy_target": "visit_distribution",
            "value_targets": ["none", "terminal_outcome", "root_value", "blend_50_50"],
            "blend_formula": "0.5 * terminal_outcome + 0.5 * root_value",
            "model_seeds": list(MODEL_SEEDS),
            "teacher_data_seeds": 1,
            "split_seed": 197,
            "validation_game_fraction": 0.25,
            "epochs": 10,
            "batch_size": 256,
            "optimizer": "Adam(lr=0.001,default_betas,default_epsilon)",
            "policy_weight": 1.0,
            "joint_value_weight": 1.0,
            "device": "cpu",
            "torch_threads": 1,
            "architecture": "default-Agent-102722-parameters",
            "puct": {
                "traversals": 32,
                "worlds": 4,
                "c_puct": 1.5,
                "root_noise": "none",
                "max_steps": 2000,
                "branch_driver_id": "full_clone/current_game_v1",
                "batch_size": 1,
                "policy_only_value": 0.5,
            },
            "matrix_players": 17,
            "matrix_cells": 136,
            "matrix_games": 544,
            "measured_roots": 128,
        },
        "decision": {
            "primary": "mean_seed_specific_diagnostic_rating",
            "paired_score_separation_minimum": 0.05,
            "competency_noninferiority_difference_minimum": 0,
            "ambiguous": "retain_point_winner_but_smoke_ambiguous",
            "integrity_failure": "kill_invalid_evidence",
            "tie_order": list(ARM_ORDER),
            "promotion_eligible": False,
            "admission_eligible": False,
            "method_level_claim": False,
        },
        "resource_caps": RESOURCE_CAPS,
    }
    mismatches = {
        key: {"actual": contract.get(key), "expected": expected}
        for key, expected in fixed.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise Int7Error(
            "INT-7 contract drift: " + json.dumps(mismatches, sort_keys=True)
        )
    if contract.get("prediction") != {
        "point_winner": "visit_terminal",
        "scale_decision": "retain_point_winner_but_smoke_ambiguous",
        "limitation": "one 507-label teacher/data seed cannot support a method, strength, rating, promotion, or admission claim",
    }:
        raise Int7Error("INT-7 contract prediction drift")
    preregistration = (
        _preregistration_commit(contract_path) if require_committed else None
    )
    return contract, contract_sha, arena, runtime, preregistration


def _batch_order_sha256(train_indices: np.ndarray, seed: int, epochs: int) -> str:
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(epochs):
        order = train_indices.copy()
        rng.shuffle(order)
        orders.append(order.tolist())
    return canonical_sha256(orders)


def _training_receipt(
    *,
    arm: str,
    seed: int,
    dataset: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    initial_state: dict[str, Any],
    out_dir: Path,
    deadline: float,
) -> tuple[dict[str, Any], Path]:
    setting = ARM_SETTINGS[arm]
    target_kind = str(setting["trainer_value_target"])
    usable, targets = value_targets_from_dataset(dataset, target_kind)
    target_sha = (
        None
        if arm == "visit_policy_only"
        else _numpy_sha256(usable.astype(np.uint8), targets.astype(np.float32))
    )
    started = time.perf_counter()
    agent, observation_space, initial_validation, history = train_search_supervised(
        dataset,
        policy_target_kind=VISIT_DISTRIBUTION_TARGET,
        value_target_kind=target_kind,
        policy_weight=1.0,
        value_weight=float(setting["value_weight"]),
        lr=1e-3,
        epochs=10,
        batch_size=256,
        val_fraction=0.25,
        seed=seed,
        split_seed=197,
        device="cpu",
        initial_agent_state={
            key: value.clone() for key, value in initial_state.items()
        },
        deadline_monotonic=deadline,
    )
    elapsed = time.perf_counter() - started
    path = out_dir / "checkpoints" / f"{arm}-seed-{seed}.pt"
    extra = {
        "experiment": "int-7-value-target-comparison-v1",
        "arm": arm,
        "seed": seed,
        "policy_target_kind": VISIT_DISTRIBUTION_TARGET,
        "value_target_kind": setting["value_target"],
        "value_weight": setting["value_weight"],
        "split_seed": 197,
        "epochs": 10,
    }
    save_bc_checkpoint(agent, observation_space, path, extra=extra)
    examples = len(train_indices) * 10
    receipt = {
        **extra,
        "trainer_value_target_kind": target_kind,
        "target_array_sha256": target_sha,
        "train_indices_sha256": canonical_sha256(train_indices.tolist()),
        "validation_indices_sha256": canonical_sha256(validation_indices.tolist()),
        "batch_order_sha256": _batch_order_sha256(train_indices, seed, 10),
        "initial_state_sha256": _state_dict_sha256(initial_state),
        "final_state_sha256": _state_dict_sha256(agent.state_dict()),
        "optimizer": "Adam(lr=0.001,default_betas,default_epsilon)",
        "initial_validation": asdict(initial_validation),
        "history": [asdict(row) for row in history],
        "elapsed_seconds": elapsed,
        "examples": examples,
        "examples_per_second": examples / elapsed,
        "checkpoint_path": path.relative_to(out_dir).as_posix(),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": file_sha256(path),
        "parameter_count": sum(parameter.numel() for parameter in agent.parameters()),
    }
    return receipt, path


def train_students(
    dataset: dict[str, np.ndarray], out_dir: Path, ledger: CumulativeResourceLedger
) -> tuple[list[dict[str, Any]], dict[str, Path], np.ndarray]:
    train_indices, validation_indices = split_by_game(
        dataset, val_fraction=0.25, seed=197
    )
    receipts: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    stage_started = time.perf_counter()
    ledger.check(
        "training",
        projected_wall_seconds=300.0,
        projected_workers=1,
        projected_artifact_bytes=12 * 500_000,
    )
    deadline = ledger.started + RESOURCE_CAPS["wall_hours"] * 3600.0
    for seed in MODEL_SEEDS:
        torch.manual_seed(seed)
        initial_agent = Agent(ObservationSpace(), AgentHypers())
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in initial_agent.state_dict().items()
        }
        block = []
        for arm in ARM_ORDER:
            receipt, path = _training_receipt(
                arm=arm,
                seed=seed,
                dataset=dataset,
                train_indices=train_indices,
                validation_indices=validation_indices,
                initial_state=initial_state,
                out_dir=out_dir,
                deadline=deadline,
            )
            receipts.append(receipt)
            block.append(receipt)
            paths[f"{arm}:{seed}"] = path
        invariant_fields = (
            "seed",
            "policy_target_kind",
            "split_seed",
            "epochs",
            "train_indices_sha256",
            "validation_indices_sha256",
            "batch_order_sha256",
            "initial_state_sha256",
            "optimizer",
            "parameter_count",
        )
        for field in invariant_fields:
            if len({json.dumps(row[field], sort_keys=True) for row in block}) != 1:
                raise Int7Error(f"matched training factor drift: seed {seed}/{field}")
    ledger.finish(
        "training", elapsed_seconds=time.perf_counter() - stage_started, workers=1
    )
    return receipts, paths, validation_indices


def _registration(
    receipt: dict[str, Any],
    runtime: dict[str, Any],
    *,
    value_mode: str,
    profile_only: bool,
) -> Int7PlayerRegistration:
    arm = str(receipt["arm"])
    seed = int(receipt["seed"])
    mode_suffix = "" if not profile_only else "-neutral-profile"
    player_id = (
        f"int7-{arm.removeprefix('visit_').replace('_', '-')}-s{seed}{mode_suffix}"
    )
    return Int7PlayerRegistration(
        player_id=player_id,
        display_name=f"INT-7 {arm} seed {seed}{' neutral' if profile_only else ''}",
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "int7_checkpoint_puct",
            "sims": 32,
            "worlds": 4,
            "c_puct": 1.5,
            "max_steps": 2000,
            "branch_driver_id": "full_clone/current_game_v1",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": True,
            "root_noise": "none",
            "implementation_source_sha256": int7_player_source_sha256(),
            "value_mode": value_mode,
        },
        compute_class_id="checkpoint-puct-cpu-s32-w4-v1",
        information_boundary="acting-viewer-history-only-v1",
        world="w2",
        content_suite="w2-interactive-mirror-v1",
        observation_abi_sha256=runtime["observation_abi_sha256"],
        action_abi_sha256=runtime["action_abi_sha256"],
        matchup_sha256=runtime["matchup_sha256"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
        checkpoint_bytes=receipt["checkpoint_bytes"],
        parameter_count=receipt["parameter_count"],
        training_seed=seed,
        artifact_id=(f"int-7-{arm}-seed-{seed}-{receipt['checkpoint_sha256'][:16]}"),
        player_seed_derivation_id="arena-comparison-alias-player-v1",
        search_call_seed_derivation_id="mcts-mix-comparison-seed-decision-v1",
        search_semantics=SearchSemantics(
            branch_audit=False,
            root_prior="checkpoint-policy-softmax-v1",
            leaf_evaluator={
                "learned": "checkpoint-sigmoid-value-v1",
                "neutral": "neutral-0.5-after-checkpoint-forward-v1",
            }[value_mode],
        ),
        arm=arm,
        value_mode=value_mode,
        profile_only=profile_only,
    )


def build_registrations(
    receipts: list[dict[str, Any]], runtime: dict[str, Any], paths: dict[str, Path]
) -> tuple[
    list[Int7PlayerRegistration],
    list[Int7PlayerRegistration],
    dict[str, str],
    dict[str, str],
]:
    primary = []
    neutral_variants = []
    checkpoint_paths: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for receipt in receipts:
        setting = ARM_SETTINGS[str(receipt["arm"])]
        registration = _registration(
            receipt, runtime, value_mode=str(setting["value_mode"]), profile_only=False
        )
        primary.append(registration)
        path = paths[f"{receipt['arm']}:{receipt['seed']}"]
        checkpoint_paths[registration.player_id] = str(path)
        aliases[registration.player_id] = (
            f"int-7-value-target-seed-{receipt['seed']}-v1"
        )
        if receipt["arm"] != "visit_policy_only":
            neutral = _registration(
                receipt, runtime, value_mode="neutral", profile_only=True
            )
            neutral_variants.append(neutral)
            checkpoint_paths[neutral.player_id] = str(path)
            aliases[neutral.player_id] = aliases[registration.player_id]
    return primary, neutral_variants, checkpoint_paths, aliases


def _merge_competencies(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    if (
        base["scenario_seeds"] != extra["scenario_seeds"]
        or base["scenario_seed_set_sha256"] != extra["scenario_seed_set_sha256"]
    ):
        raise Int7Error("competency schedule drift")
    merged = dict(base)
    merged["players"] = {**base["players"], **extra["players"]}
    return merged


def _score_for_player(row: dict[str, Any], player_id: str) -> float:
    return (
        float(row["score_a"])
        if row["player_a"] == player_id
        else 1.0 - float(row["score_a"])
    )


def _candidate_anchor_score(
    rows: list[dict[str, Any]], candidate_id: str, anchor_ids: set[str]
) -> float:
    selected = [
        _score_for_player(row, candidate_id)
        for row in rows
        if candidate_id in {row["player_a"], row["player_b"]}
        and ({row["player_a"], row["player_b"]} - {candidate_id}).issubset(anchor_ids)
    ]
    if len(selected) != 20:
        raise Int7Error(f"candidate-anchor schedule mismatch: {candidate_id}")
    return float(np.mean(selected))


def _competency_correct(competencies: dict[str, Any], player_id: str) -> int:
    return sum(
        int(bool(run["correct"]))
        for scenario in competencies["players"][player_id].values()
        for run in scenario["runs"]
    )


def decision_payload(
    *,
    rows: list[dict[str, Any]],
    registrations: list[Int7PlayerRegistration],
    ratings: dict[str, float],
    competencies: dict[str, Any],
    profile: dict[str, Any],
    replay: dict[str, Any],
    anchor_ids: set[str],
) -> dict[str, Any]:
    by_arm = {arm: [] for arm in ARM_ORDER}
    for registration in registrations:
        by_arm[registration.arm].append(registration)
    methods: dict[str, Any] = {}
    for arm, candidates in by_arm.items():
        candidates.sort(key=lambda row: row.training_seed)
        rating_mean = float(np.mean([ratings[row.player_id] for row in candidates]))
        anchor_mean = float(
            np.mean(
                [
                    _candidate_anchor_score(rows, row.player_id, anchor_ids)
                    for row in candidates
                ]
            )
        )
        competency_total = sum(
            _competency_correct(competencies, row.player_id) for row in candidates
        )
        within_seed_scores = []
        for candidate in candidates:
            opponents = {
                row.player_id
                for row in registrations
                if row.training_seed == candidate.training_seed and row.arm != arm
            }
            selected = [
                _score_for_player(row, candidate.player_id)
                for row in rows
                if candidate.player_id in {row["player_a"], row["player_b"]}
                and (
                    {row["player_a"], row["player_b"]} - {candidate.player_id}
                ).issubset(opponents)
            ]
            within_seed_scores.extend(selected)
        methods[arm] = {
            "mean_diagnostic_rating": rating_mean,
            "mean_paired_score_against_anchors": anchor_mean,
            "within_seed_head_to_head_score": float(np.mean(within_seed_scores)),
            "competency_correct": competency_total,
            "mean_isolated_p95_seconds": float(
                np.mean(
                    [
                        profile["players"][row.player_id]["p95_seconds"]
                        for row in candidates
                    ]
                )
            ),
            "candidate_ids": [row.player_id for row in candidates],
        }
    policy_competency = methods["visit_policy_only"]["competency_correct"]
    for arm in ARM_ORDER:
        methods[arm]["competency_delta_vs_policy_only"] = (
            methods[arm]["competency_correct"] - policy_competency
        )
        methods[arm]["competency_noninferior"] = (
            methods[arm]["competency_delta_vs_policy_only"] >= 0
        )
    ranked = sorted(
        ARM_ORDER,
        key=lambda arm: (
            -methods[arm]["mean_diagnostic_rating"],
            -methods[arm]["mean_paired_score_against_anchors"],
            -methods[arm]["within_seed_head_to_head_score"],
            -methods[arm]["competency_correct"],
            methods[arm]["mean_isolated_p95_seconds"],
            FINAL_TIE_ORDER[arm],
        ),
    )
    point_winner = ranked[0]
    integrity = {
        "games": len(rows),
        "row_replay_passed": all(row["replay_passed"] for row in rows),
        "row_integrity_zero": all(
            not any(int(value) for value in row["integrity"].values()) for row in rows
        ),
        "no_truncations": all(not row["truncated"] for row in rows),
        "trace_replay_passed": bool(replay["passed"]),
        "profile_roots_preserved": all(
            int(block["root_mutations"]) == 0 and int(block["illegal_actions"]) == 0
            for block in profile["players"].values()
        ),
    }
    integrity_passed = all(value for key, value in integrity.items() if key != "games")
    separation = min(
        methods[point_winner]["mean_paired_score_against_anchors"]
        - methods[arm]["mean_paired_score_against_anchors"]
        for arm in ARM_ORDER
        if arm != point_winner
    )
    if not integrity_passed:
        decision = "kill_invalid_evidence"
    elif methods[point_winner]["competency_noninferior"] and separation >= 0.05:
        decision = f"continue_{point_winner}"
    else:
        decision = "retain_point_winner_but_smoke_ambiguous"
    unsigned = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "point_estimate_winner": point_winner,
        "decision": decision,
        "paired_score_minimum_separation": separation,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "methods": methods,
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
        "limitation": "one 507-row teacher/data seed; model seeds measure initialization sensitivity only",
    }
    return {**unsigned, "input_sha256": canonical_sha256(unsigned)}


def search_efficiency_payload(
    rows: list[dict[str, Any]],
    registrations: list[Int7PlayerRegistration],
    ratings: dict[str, float],
    profile: dict[str, Any],
    anchor_ids: set[str],
) -> dict[str, Any]:
    by_key = {(row.arm, row.training_seed): row for row in registrations}
    result = {}
    for arm in ARM_ORDER:
        if arm == "visit_policy_only":
            continue
        for seed in MODEL_SEEDS:
            candidate = by_key[(arm, seed)]
            control = by_key[("visit_policy_only", seed)]
            block = profile["players"][candidate.player_id]
            rating_uplift = ratings[candidate.player_id] - ratings[control.player_id]
            paired_uplift = _candidate_anchor_score(
                rows, candidate.player_id, anchor_ids
            ) - _candidate_anchor_score(rows, control.player_id, anchor_ids)
            result[f"{arm}-seed-{seed}"] = {
                "rating_uplift": rating_uplift,
                "paired_score_uplift": paired_uplift,
                "mean_nodes_per_decision": block["nodes_per_label"],
                "mean_cpu_seconds_per_decision": block["cpu_seconds_per_label"],
                "rating_uplift_per_node": rating_uplift / block["nodes_per_label"],
                "rating_uplift_per_cpu_second": rating_uplift
                / block["cpu_seconds_per_label"],
                "paired_score_uplift_per_node": paired_uplift
                / block["nodes_per_label"],
                "paired_score_uplift_per_cpu_second": paired_uplift
                / block["cpu_seconds_per_label"],
                "interpretation": "diagnostic_matrix_efficiency_not_transferable_strength",
            }
    return result


def _inherited_cost() -> dict[str, Any]:
    payload = INPUT_MANIFEST.parent / "payload"
    source_manifest = load_json(payload / "manifest.json")
    dataset_manifest = load_json(payload / "dataset/manifest.json")
    label_wall = sum(
        float(row["summary"]["wall_seconds"]) for row in dataset_manifest["shards"]
    )
    label_search = sum(
        float(row["summary"]["search"]["seconds"]) for row in dataset_manifest["shards"]
    )
    return {
        "labels": 507,
        "teacher_dataset_wall_seconds": label_wall,
        "teacher_search_seconds": label_search,
        "source_manifest_sha256": file_sha256(payload / "manifest.json"),
        "source_stage_wall_seconds": {
            key: float(
                value["result"].get("wall_seconds", value.get("wall_seconds", 0.0))
            )
            if isinstance(value, dict) and isinstance(value.get("result"), dict)
            else float(value.get("wall_seconds", 0.0))
            for key, value in source_manifest.get("stages", {}).items()
        },
    }


def _report_markdown(decision: dict[str, Any], cost: dict[str, Any]) -> str:
    lines = [
        "# INT-7 Value Target Comparison",
        "",
        f"Decision: **{decision['decision']}**",
        f"Point-estimate winner: `{decision['point_estimate_winner']}`",
        "",
        "This is one-corpus engineering-smoke evidence only. It is not an admission, promotion, rating, strength, or method claim.",
        "",
        "## Complete-player results",
        "",
        "| Method | Mean diagnostic rating | Anchor paired score | Competency correct / 30 | Mean p95 (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARM_ORDER:
        row = decision["methods"][arm]
        lines.append(
            f"| `{arm}` | {row['mean_diagnostic_rating']:.3f} | "
            f"{row['mean_paired_score_against_anchors']:.3f} | "
            f"{row['competency_correct']} | {row['mean_isolated_p95_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"Inherited labels: {cost['inherited']['labels']}.",
            f"Marginal training seconds: {cost['marginal']['training_seconds']:.3f}.",
            f"Evaluation wall seconds: {cost['marginal']['evaluation_wall_seconds']:.3f}.",
            "",
            "Calibration and teacher-target agreement are retained as subordinate diagnostics and did not select the winner.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    contract, contract_sha, arena, runtime, preregistration = preflight_contract()
    input_path = args.input_manifest.resolve()
    if input_path != INPUT_MANIFEST.resolve():
        raise Int7Error("INT-7 accepts only the preregistered retained input manifest")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise Int7Error("INT-7 output directory already exists")
    out_dir.mkdir(parents=True)
    ledger = CumulativeResourceLedger(out_dir, started=started, caps=RESOURCE_CAPS)
    before = verify_retained_input(input_path)
    write_json(out_dir / "input-before.json", before)
    payload = input_path.parent / "payload"
    dataset = load_shards(sorted((payload / "dataset").glob("shard_*.npz")))
    if len(dataset["action"]) != 507 or len(np.unique(dataset["game_index"])) != 4:
        raise Int7Error("retained dataset cardinality drift")
    receipts, checkpoint_source_paths, validation_indices = train_students(
        dataset, out_dir, ledger
    )
    write_json(out_dir / "training.json", {"checkpoints": receipts})
    primary, neutral_variants, checkpoint_paths, aliases = build_registrations(
        receipts, runtime, checkpoint_source_paths
    )
    players = [*arena.anchors, *primary]
    write_json(out_dir / "players.json", [row.model_dump() for row in players])
    write_json(
        out_dir / "profile-variants.json",
        [row.model_dump() for row in neutral_variants],
    )
    calibrations = {
        f"{row['arm']}-seed-{row['seed']}": checkpoint_calibration(
            out_dir / row["checkpoint_path"], dataset, validation_indices
        )
        for row in receipts
    }
    write_json(out_dir / "calibration.json", calibrations)

    schedule = arena.schedules["smoke"]
    pairs = list(combinations(players, 2))
    if len(pairs) != 136:
        raise Int7Error("complete INT-7 matrix must contain 136 cells")
    for worker in range(1, 5):
        ledger.check(
            "arena",
            projected_wall_seconds=14_400.0,
            projected_workers=4,
            projected_artifact_bytes=512 * 1024 * 1024,
            worker_launch=worker,
        )
    arena_started = time.perf_counter()
    cell_results = play_int7_cells(
        key=arena.key,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=out_dir,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
        workers=4,
    )
    arena_elapsed = time.perf_counter() - arena_started
    ledger.finish("arena", elapsed_seconds=arena_elapsed, workers=4)
    rows = [row for cell_rows, _trace, _replay in cell_results for row in cell_rows]
    traces = [trace for _rows, trace, _replay in cell_results]
    replay_cells = [replay for _rows, _trace, replay in cell_results]
    if len(rows) != 544:
        raise Int7Error("complete INT-7 matrix must contain 544 games")
    write_jsonl(out_dir / "matches.jsonl", rows)

    ledger.check("competencies", projected_wall_seconds=1800.0, projected_workers=1)
    competency_started = time.perf_counter()
    anchor_competencies = run_competencies(
        list(arena.anchors), seeds=schedule.competency_seeds
    )
    candidate_competencies = run_int7_competencies(
        primary,
        seeds=schedule.competency_seeds,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
    )
    competencies = _merge_competencies(anchor_competencies, candidate_competencies)
    ledger.finish(
        "competencies",
        elapsed_seconds=time.perf_counter() - competency_started,
        workers=1,
    )
    write_json(out_dir / "competencies.json", competencies)

    fit, rating = rating_payload(rows, schedule, arena)
    matrix = payoff_matrix(rows)
    write_json(out_dir / "rating.json", rating)
    write_json(out_dir / "payoff-matrix.json", matrix)
    source_trace_receipt = next(
        trace
        for trace in traces
        if Path(trace["artifact_path"]).name
        == "random-v1__scripted-greedy-v1.commands.jsonl.gz"
    )
    source_games = read_trace(out_dir / source_trace_receipt["artifact_path"])
    ledger.check("profile", projected_wall_seconds=3600.0, projected_workers=1)
    profile_started = time.perf_counter()
    matched_root = profile_int7_players(
        [*players, *neutral_variants],
        source_games=source_games,
        profile_roots=arena.profile_roots,
        checkpoint_paths=checkpoint_paths,
        comparison_seed_aliases=aliases,
    )
    profile_elapsed = time.perf_counter() - profile_started
    ledger.finish("profile", elapsed_seconds=profile_elapsed, workers=1)
    profile = {
        "native_gameplay": native_gameplay_profiles(rows, worker_count=4),
        "matched_root": matched_root,
    }
    write_json(out_dir / "profile.json", profile)
    mechanism = mechanism_payload(matched_root, [*primary, *neutral_variants])
    write_json(out_dir / "mechanism.json", mechanism)
    efficiency = search_efficiency_payload(
        rows,
        primary,
        fit.ratings,
        matched_root,
        {row.player_id for row in arena.anchors},
    )
    write_json(out_dir / "search-efficiency.json", efficiency)
    replay = {
        "passed": all(row["passed"] for row in replay_cells),
        "cells": replay_cells,
        "games": len(rows),
        "decisions": sum(int(row["decisions"]) for row in replay_cells),
    }
    write_json(out_dir / "replay.json", replay)
    decision = decision_payload(
        rows=rows,
        registrations=primary,
        ratings=fit.ratings,
        competencies=competencies,
        profile=matched_root,
        replay=replay,
        anchor_ids={row.player_id for row in arena.anchors},
    )
    write_json(out_dir / "decision.json", decision)
    inherited = _inherited_cost()
    cost = {
        "inherited": inherited,
        "marginal": {
            "training_seconds": sum(float(row["elapsed_seconds"]) for row in receipts),
            "training_examples": sum(int(row["examples"]) for row in receipts),
            "arena_wall_seconds": arena_elapsed,
            "profile_wall_seconds": profile_elapsed,
            "evaluation_wall_seconds": time.perf_counter() - started,
            "artifact_bytes_before_manifest": sum(
                path.stat().st_size for path in out_dir.rglob("*") if path.is_file()
            ),
        },
    }
    write_json(out_dir / "cost.json", cost)
    (out_dir / "report.md").write_text(_report_markdown(decision, cost))
    after = verify_retained_input(input_path)
    write_json(out_dir / "input-after.json", after)
    if before != after:
        raise Int7Error("retained input identity changed during INT-7")
    ledger.check(
        "finalize",
        projected_wall_seconds=300.0,
        projected_workers=1,
        projected_artifact_bytes=8 * 1024 * 1024,
    )
    ledger.finish("finalize", elapsed_seconds=0.0, workers=1)
    ledger.complete({"games": 544, "cells": 136, "measured_roots": 128})
    trace_receipts = [
        {
            "path": trace["artifact_path"],
            "sha256": trace["sha256"],
            "games": trace["games"],
        }
        for trace in traces
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "int-7-value-target-comparison-v1",
        "evidence_class": EVIDENCE_CLASS,
        "contract_path": EXPERIMENT_CONTRACT.relative_to(REPO_ROOT).as_posix(),
        "contract_sha256": contract_sha,
        "preregistration_commit": preregistration,
        "frozen_int6_contract_sha256": EXPECTED_INT6_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_SHA256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "arena_key": arena.key.model_dump(),
        "runtime": runtime,
        "candidates": [row.model_dump() for row in primary],
        "profile_variants": [row.model_dump() for row in neutral_variants],
        "comparison_seed_aliases": aliases,
        "traces": trace_receipts,
        "files": _tree_receipts(out_dir),
        "decision": decision["decision"],
        "point_estimate_winner": decision["point_estimate_winner"],
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
        "contract_prediction": contract["prediction"],
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    write_json(out_dir / "manifest.json", manifest)
    verification = verify_output(out_dir)
    retention = retain_output(out_dir, manifest)
    return {
        "state": "complete",
        "artifact": str(out_dir / "manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "decision": decision["decision"],
        "point_estimate_winner": decision["point_estimate_winner"],
        "verification": verification,
        "retention": retention,
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
    }


def verify_output(out_dir: Path) -> dict[str, Any]:
    contract, contract_sha, arena, runtime, preregistration = preflight_contract()
    manifest = load_json(out_dir / "manifest.json")
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise Int7Error("INT-7 manifest digest mismatch")
    expected_manifest = {
        "contract_sha256": contract_sha,
        "preregistration_commit": preregistration,
        "frozen_int6_contract_sha256": EXPECTED_INT6_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_SHA256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "arena_key": arena.key.model_dump(),
        "runtime": runtime,
        "evidence_class": EVIDENCE_CLASS,
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise Int7Error("INT-7 manifest dependency binding mismatch")
    expected_files = {
        row["path"]: (int(row["bytes"]), row["sha256"]) for row in manifest["files"]
    }
    actual_paths = {
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != set(expected_files):
        raise Int7Error("INT-7 artifact closed set mismatch")
    for relative, (size, sha) in expected_files.items():
        path = out_dir / relative
        if path.stat().st_size != size or file_sha256(path) != sha:
            raise Int7Error(f"INT-7 artifact digest mismatch: {relative}")
    before = load_json(out_dir / "input-before.json")
    after = load_json(out_dir / "input-after.json")
    current_input = verify_retained_input(INPUT_MANIFEST)
    if before != after or after != current_input:
        raise Int7Error("INT-7 retained input verification drift")
    ledger = verify_resource_ledger(out_dir / "resource-ledger.jsonl", RESOURCE_CAPS)
    training = load_json(out_dir / "training.json")["checkpoints"]
    if len(training) != 12:
        raise Int7Error("INT-7 requires exactly twelve checkpoints")
    for row in training:
        checkpoint = out_dir / row["checkpoint_path"]
        if (
            file_sha256(checkpoint) != row["checkpoint_sha256"]
            or checkpoint.stat().st_size != row["checkpoint_bytes"]
        ):
            raise Int7Error("INT-7 checkpoint identity mismatch")
        agent, _ = load_checkpoint_agent(str(checkpoint))
        if sum(parameter.numel() for parameter in agent.parameters()) != 102722:
            raise Int7Error("INT-7 checkpoint architecture mismatch")
    candidates = [
        Int7PlayerRegistration.model_validate(row) for row in manifest["candidates"]
    ]
    variants = [
        Int7PlayerRegistration.model_validate(row)
        for row in manifest["profile_variants"]
    ]
    if len(candidates) != 12 or len(variants) != 9:
        raise Int7Error("INT-7 registration cohort mismatch")
    rows = read_jsonl(out_dir / "matches.jsonl")
    validated = [MatchRow.model_validate(row) for row in rows]
    if len(validated) != 544 or len({row.cell_id for row in validated}) != 136:
        raise Int7Error("INT-7 match schedule cardinality mismatch")
    if {row.deal_seed for row in validated} != set(arena.schedules["smoke"].deal_seeds):
        raise Int7Error("INT-7 match schedule seed mismatch")
    if len({(row.cell_id, row.deal_block, row.leg) for row in validated}) != 544:
        raise Int7Error("INT-7 match schedule duplicate or missing seat leg")
    registrations = {row.player_id: row for row in (*arena.anchors, *candidates)}
    if any(
        row.player_a_registration_sha256 != registrations[row.player_a].identity_sha256
        or row.player_b_registration_sha256
        != registrations[row.player_b].identity_sha256
        for row in validated
    ):
        raise Int7Error("INT-7 match registration binding mismatch")
    traced: dict[str, str] = {}
    replay_receipts = []
    for trace in manifest["traces"]:
        path = out_dir / trace["path"]
        games = read_trace(path)
        if len(games) != 4 or file_sha256(path) != trace["sha256"]:
            raise Int7Error("INT-7 trace identity mismatch")
        receipt = replay_games(games).to_dict()
        replay_receipts.append(receipt)
        for game in games:
            traced[game["game_trace_sha256"]] = trace["sha256"]
    if len(traced) != 544 or not all(row["passed"] for row in replay_receipts):
        raise Int7Error("INT-7 exact Command replay mismatch")
    if any(
        row.trace_shard_sha256 != traced.get(row.game_trace_sha256) for row in validated
    ):
        raise Int7Error("INT-7 trace-to-match binding mismatch")
    _fit, recomputed_rating = rating_payload(rows, arena.schedules["smoke"], arena)
    if recomputed_rating != load_json(out_dir / "rating.json"):
        raise Int7Error("INT-7 rating recomputation mismatch")
    if payoff_matrix(rows) != load_json(out_dir / "payoff-matrix.json"):
        raise Int7Error("INT-7 payoff matrix recomputation mismatch")
    profile = load_json(out_dir / "profile.json")
    verify_profile(profile["matched_root"])
    if (
        profile["matched_root"]["measured_roots"] != 128
        or len(profile["matched_root"]["players"]) != 26
    ):
        raise Int7Error("INT-7 matched-root cohort mismatch")
    if mechanism_payload(
        profile["matched_root"], [*candidates, *variants]
    ) != load_json(out_dir / "mechanism.json"):
        raise Int7Error("INT-7 mechanism recomputation mismatch")
    competencies = load_json(out_dir / "competencies.json")
    if any(_competency_correct(competencies, row.player_id) > 10 for row in candidates):
        raise Int7Error("INT-7 competency row count mismatch")
    replay = load_json(out_dir / "replay.json")
    recomputed_decision = decision_payload(
        rows=rows,
        registrations=candidates,
        ratings=recomputed_rating["ratings"],
        competencies=competencies,
        profile=profile["matched_root"],
        replay=replay,
        anchor_ids={row.player_id for row in arena.anchors},
    )
    if recomputed_decision != load_json(out_dir / "decision.json"):
        raise Int7Error("INT-7 decision recomputation mismatch")
    if manifest["decision"] != recomputed_decision["decision"]:
        raise Int7Error("INT-7 manifest decision mismatch")
    return {
        "state": "verified",
        "manifest_sha256": manifest["manifest_sha256"],
        "games": 544,
        "cells": 136,
        "checkpoints": 12,
        "measured_roots": 128,
        "replayed_decisions": sum(row["decisions"] for row in replay_receipts),
        "ledger": ledger,
        "decision": recomputed_decision["decision"],
        "point_estimate_winner": recomputed_decision["point_estimate_winner"],
        "no_generation": True,
    }


def retain_output(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    destination = RETENTION_ROOT / manifest["manifest_sha256"]
    if destination.exists():
        raise Int7Error("INT-7 content-addressed retention destination already exists")
    result_dir = destination / "result"
    result_dir.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(out_dir, result_dir)
    verification = verify_output(result_dir)
    retention = {
        "schema_version": 1,
        "artifact_id": "int-7-value-target-comparison-v1",
        "evidence_class": EVIDENCE_CLASS,
        "result_path": "result",
        "manifest_file_sha256": file_sha256(result_dir / "manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "decision": manifest["decision"],
        "point_estimate_winner": manifest["point_estimate_winner"],
        "frozen_int6_contract_sha256": EXPECTED_INT6_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_SHA256,
        "verification": verification,
        "promotion_eligible": False,
        "admission_eligible": False,
        "method_level_claim": False,
    }
    write_json(destination / "retention.json", retention)
    return {"path": str(destination), **retention}


def verify_only(out_dir: Path) -> dict[str, Any]:
    before = _tree_snapshot(out_dir)
    verification = verify_output(out_dir)
    after = _tree_snapshot(out_dir)
    if before != after:
        raise Int7Error("INT-7 verify-only changed output bytes")
    return verification


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        if args.input_manifest is not None:
            parser.error("--verify-only does not accept --input-manifest")
    elif args.input_manifest is None:
        parser.error("--input-manifest is required unless --verify-only is used")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = verify_only(args.out_dir.resolve()) if args.verify_only else run(args)
        print(json.dumps(result, sort_keys=True))
    except (
        Int7Error,
        RetainedInputError,
        ResourceCapExceeded,
        ValueError,
        FileNotFoundError,
    ) as error:
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
