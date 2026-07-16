"""Run controlled search-supervised policy/value experiments.

The default Teacher-0 arm uses flat determinized Monte Carlo. Teacher-1 uses
multi-world PUCT with uniform priors, random leaf playouts, and real root visit
counts. Teacher-1 is MCTS, but it is not yet neural PUCT or information-set
consistent search.

Teacher-0's two arms share the policy target and isolate value supervision:

- ``policy_only``: policy loss only;
- ``policy_value``: the same policy loss plus terminal value BCE.

Teacher-1's arms share data, initialization, split, optimizer, capacity, and
root-value supervision while isolating the policy label:

- ``chosen_action``: one-hot supervision on the teacher's selected action;
- ``visit_distribution``: the complete normalized root-visit distribution.

Usage (smoke):
    uv run experiments/runners/run_search_supervised.py \
        --out-dir .runs/search-supervised-smoke --games 8 --workers 2 \
        --sims 2 --epochs 2 --quick-games 8 --teacher-probe-games 4 \
        --student-teacher-games 2 --teacher-min-win-rate 0.0 --device mps

Usage (overnight):
    uv run experiments/runners/run_search_supervised.py \
        --out-dir .runs/search-supervised-overnight --games 2000 --workers 12 \
        --sims 128 --epochs 25 --batch-size 1024 --quick-games 400 \
        --teacher-probe-games 40 --student-teacher-games 40 --device mps

Usage (Teacher-1 PUCT smoke):
    uv run experiments/runners/run_search_supervised.py \
        --teacher-kind determinized_puct --worlds 2 \
        --out-dir .runs/puct-supervised-smoke --games 8 --workers 2 \
        --sims 8 --epochs 2 --quick-games 8 --teacher-probe-games 4 \
        --student-teacher-games 2 --teacher-min-win-rate 0.0 --device mps
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.distill import (
    ROOT_VALUE_KEY,
    SCORE_KEY,
    VISIT_COUNT_KEY,
    _git_commit,
    load_shards,
    save_bc_checkpoint,
    soft_targets_from_scores,
)
from manabot.sim.flat_mc import aggregate_records, load_checkpoint_agent, play_games
from manabot.sim.rollout import BatchedSampler, RandomBatchController, run_vector_games
from manabot.sim.search_supervised import (
    CHOSEN_ACTION_TARGET,
    ROOT_VALUE_TARGET,
    SCORE_SOFTMAX_TARGET,
    TERMINAL_OUTCOME_TARGET,
    VISIT_DISTRIBUTION_TARGET,
    train_search_supervised,
)
from manabot.verify.util import INTERACTIVE_DECK

EXPERIMENT_CONTRACT = {
    "question": (
        "Does adding terminal-value supervision to the existing search-score "
        "policy target improve representation learning without weakening the policy?"
    ),
    "prediction": (
        "The joint arm will beat a constant-0.5 value predictor while retaining "
        "policy accuracy and gameplay strength within the pre-registered margin."
    ),
    "success_gates": {
        "teacher_signal": "teacher win rate versus random >= configured threshold",
        "policy_loss_improved": "held-out policy CE below the untrained baseline",
        "policy_beats_uniform": (
            "nontrivial held-out top-1 accuracy above uniform-over-legal"
        ),
        "value_beats_coin_brier": "joint held-out Brier score < 0.25",
        "joint_policy_noninferior": (
            "joint win rate versus random no more than 0.10 below policy-only"
        ),
    },
    "failure_branches": {
        "teacher_signal_fails": "improve the search teacher before training",
        "policy_fails": "inspect target entropy, representation, and optimization",
        "value_only_fails": "change value targets, loss weighting, or calibration",
        "validation_only": "treat gameplay distribution shift as leading diagnosis",
        "student_teacher_gap": (
            "increase data/capacity or improve target formulation; do not infer "
            "teacher recovery from validation accuracy"
        ),
    },
}

PUCT_EXPERIMENT_CONTRACT = {
    "question": (
        "At matched PUCT data, initialization, capacity, and optimization, do "
        "root visit-distribution targets produce a more useful student than the "
        "teacher's chosen action alone?"
    ),
    "prediction": (
        "Visit supervision will retain more search information than one-hot "
        "actions while the value head reproduces root values below a 0.25 "
        "held-out Brier/MSE. Outcome calibration is a separate evaluation."
    ),
    "success_gates": {
        "teacher_signal": "teacher win rate versus random >= configured threshold",
        "policy_loss_improved": "held-out policy CE below the untrained baseline",
        "policy_beats_uniform": (
            "nontrivial held-out top-1 accuracy above uniform-over-legal"
        ),
        "value_beats_coin_brier": (
            "held-out Brier/MSE to teacher root values < 0.25; this does not "
            "establish outcome calibration"
        ),
        "visit_policy_noninferior": (
            "visit-target win rate versus random no more than 0.10 below chosen-action"
        ),
    },
    "failure_branches": {
        "teacher_signal_fails": "improve tree selection or leaves before training",
        "visits_fail_only": "inspect visit temperature, search budget, and target entropy",
        "both_policy_arms_fail": "inspect representation, optimization, and data diversity",
        "value_fails": "separate noisy root values from terminal-outcome supervision",
        "validation_only": "treat gameplay distribution shift as leading diagnosis",
    },
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _runtime_contract(seed: int, device: str) -> dict[str, Any]:
    """Pin the engine content, observation ABI, matchup, and host runtime."""

    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="teacher-a",
            villain="teacher-b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed, auto_reset=False)
    env.reset(seed=seed)
    content_manifest = env.content_pack_manifest()
    observation_schema = {
        name: {"shape": list(shape), "dtype": "float32"}
        for name, shape in sorted(obs_space.shapes.items())
    }
    matchup = match.hypers.model_dump()
    return {
        "content_pack": {
            "schema_version": content_manifest.get("schema_version"),
            "content_digest": content_manifest.get("content_digest"),
            "manifest_sha256": _json_sha256(content_manifest),
            "definition_count": len(content_manifest.get("definitions", [])),
        },
        "observation_abi": {
            "schema": observation_schema,
            "sha256": _json_sha256(observation_schema),
        },
        "matchup": matchup,
        "matchup_sha256": _json_sha256(matchup),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "requested_device": device,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        },
    }


def _wall_cap_exceeded(started: float, cap_hours: float) -> bool:
    return time.perf_counter() - started > cap_hours * 3600.0


def _stop_at_wall_cap(
    manifest: dict[str, Any], manifest_path: Path, started: float
) -> None:
    manifest["status"] = "stopped_wall_cap"
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_json(manifest_path, manifest)
    raise SystemExit(
        f"wall cap reached after {manifest['wall_seconds']:.0f}s; "
        "partial artifacts and manifest were retained"
    )


def _matchup(
    hero: dict[str, Any], villain: dict[str, Any], *, games: int, seed: int
) -> dict[str, Any]:
    if games <= 0:
        return {"skipped": True, "num_games": 0}
    result = play_games(hero, villain, num_games=games, seed=seed)
    metrics: dict[str, Any] = aggregate_records(result.records)
    metrics.update(
        wall_seconds=result.wall_seconds,
        hero=result.hero,
        villain=result.villain,
    )
    if result.hero_search is not None:
        metrics["hero_search"] = result.hero_search.to_dict()
    if result.villain_search is not None:
        metrics["villain_search"] = result.villain_search.to_dict()
    return metrics


def _student_vs_random(
    checkpoint: Path, *, games: int, seed: int, device: str
) -> dict[str, Any]:
    agent, _ = load_checkpoint_agent(str(checkpoint))
    records, runtime = run_vector_games(
        BatchedSampler(agent, deterministic=False, seed=seed, device=device),
        RandomBatchController(seed=seed + 1),
        num_games=games,
        num_streams=min(128, max(2, games)),
        seed=seed,
    )
    metrics: dict[str, Any] = aggregate_records(records)
    metrics["runtime"] = runtime
    return metrics


def _dataset_diagnostics(
    dataset: dict[str, np.ndarray], *, policy_target_kind: str, temperature: float
) -> dict[str, Any]:
    if policy_target_kind == SCORE_SOFTMAX_TARGET:
        scores = torch.as_tensor(dataset[SCORE_KEY], dtype=torch.float32)
        targets = soft_targets_from_scores(scores, temperature)
        sorted_signal = np.sort(
            np.where(scores.numpy() >= 0, scores.numpy(), -np.inf), axis=1
        )
        signal_name = "score"
    else:
        visits = torch.as_tensor(dataset[VISIT_COUNT_KEY], dtype=torch.float32)
        valid = torch.as_tensor(dataset["actions_valid"] > 0, dtype=torch.bool)
        visits = visits.masked_fill(~valid, 0.0)
        targets = visits / visits.sum(dim=-1, keepdim=True).clamp_min(1.0)
        sorted_signal = np.sort(
            np.where(valid.numpy(), targets.numpy(), -np.inf), axis=1
        )
        signal_name = "visit_probability"
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(dim=-1)
    actions = np.asarray(dataset["action"], dtype=np.int64)
    valid = np.asarray(dataset["actions_valid"]) > 0
    in_range = (actions >= 0) & (actions < valid.shape[1])
    legal = np.zeros(len(actions), dtype=bool)
    legal[in_range] = valid[np.arange(len(actions))[in_range], actions[in_range]]
    if sorted_signal.shape[1] > 1:
        margins = sorted_signal[:, -1] - sorted_signal[:, -2]
        margins = margins[np.isfinite(margins)]
    else:
        margins = np.asarray([], dtype=np.float32)
    return {
        "decisions": int(len(dataset["action"])),
        "games": int(len(np.unique(dataset["game_index"]))),
        "mean_valid_actions": float(np.mean(dataset["num_valid"])),
        "policy_target_kind": policy_target_kind,
        "policy_temperature": temperature,
        "mean_target_entropy": float(entropy.mean().item()),
        f"mean_top_two_{signal_name}_margin": (
            float(np.mean(margins)) if len(margins) else None
        ),
        "winnerless_rows": int(np.sum(dataset["winner"] < 0)),
        "invalid_teacher_actions": int(np.sum(~legal)),
        "action_mask_width": int(valid.shape[1]),
        "mean_root_value": (
            float(np.mean(dataset[ROOT_VALUE_KEY]))
            if ROOT_VALUE_KEY in dataset
            else None
        ),
    }


def _generate_dataset(
    args: argparse.Namespace,
    dataset_dir: Path,
    teacher_spec: dict[str, Any],
    *,
    resume: bool = False,
) -> None:
    command = [
        sys.executable,
        "experiments/runners/run_distill_datagen.py",
        "--games",
        str(args.games),
        "--workers",
        str(args.workers),
        "--sims",
        str(args.sims),
        "--games-per-shard",
        str(args.games_per_shard),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(dataset_dir),
    ]
    if teacher_spec["kind"] != "search":
        command.extend(["--teacher-json", json.dumps(teacher_spec, sort_keys=True)])
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def _validate_dataset_manifest(
    generator_manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    policy_target_kind: str,
    value_target_kind: str,
    shard_paths: list[Path],
) -> None:
    """Fail closed when an existing dataset does not match this experiment.

    A stale shard in a reused output directory can otherwise change the data,
    teacher, or split while the top-level manifest records the new command.
    That is evidence corruption, not a resumable run.
    """

    expected_teacher = {k: v for k, v in teacher_spec.items() if k != "device"}
    provenance = generator_manifest.get("provenance") or {}
    actual_teacher = provenance.get("teacher_spec")
    actual_policy_target = generator_manifest.get("policy_target_kind")
    actual_value_target = generator_manifest.get("value_target_kind")
    # Teacher-0 shards generated before schema v2 did not copy target kinds to
    # the manifest. Their exact flat-search teacher spec makes both targets
    # unambiguous; tree teachers must always declare the new fields.
    if actual_teacher and actual_teacher.get("kind") == "search":
        actual_policy_target = actual_policy_target or SCORE_SOFTMAX_TARGET
        actual_value_target = actual_value_target or TERMINAL_OUTCOME_TARGET

    expected_scalars = {
        "games": args.games,
        "workers": args.workers,
        "seed": args.seed,
        "sims": args.sims,
        "games_per_shard": args.games_per_shard,
    }
    mismatches = [
        f"{key}={generator_manifest.get(key)!r} (expected {expected!r})"
        for key, expected in expected_scalars.items()
        if generator_manifest.get(key) != expected
    ]
    if actual_policy_target != policy_target_kind:
        mismatches.append(
            f"policy_target_kind={actual_policy_target!r} "
            f"(expected {policy_target_kind!r})"
        )
    if actual_value_target != value_target_kind:
        mismatches.append(
            f"value_target_kind={actual_value_target!r} "
            f"(expected {value_target_kind!r})"
        )
    if actual_teacher != expected_teacher:
        mismatches.append(
            f"teacher_spec={actual_teacher!r} (expected {expected_teacher!r})"
        )

    declared_shards = generator_manifest.get("shards") or []
    declared_names = sorted(
        Path(str(item.get("out_path"))).name
        for item in declared_shards
        if item.get("out_path")
    )
    actual_names = sorted(path.name for path in shard_paths)
    if declared_names != actual_names:
        mismatches.append(
            f"shards={actual_names!r} (manifest declares {declared_names!r})"
        )

    if mismatches:
        raise SystemExit(
            "dataset manifest does not match the requested experiment: "
            + "; ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument(
        "--games-per-shard",
        type=int,
        default=8,
        help="durable datagen checkpoint granularity",
    )
    parser.add_argument(
        "--teacher-kind",
        choices=("flat_mc", "determinized_puct"),
        default="flat_mc",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=4,
        help="hidden-information worlds for determinized_puct",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy-temperature", type=float, default=0.05)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick-games", type=int, default=400)
    parser.add_argument("--teacher-probe-games", type=int, default=40)
    parser.add_argument("--student-teacher-games", type=int, default=40)
    parser.add_argument("--teacher-min-win-rate", type=float, default=0.6)
    parser.add_argument(
        "--wall-cap-hours",
        type=float,
        default=8.0,
        help="declared end-to-end cap, checked between experiment stages",
    )
    parser.add_argument(
        "--resume-dataset",
        action="store_true",
        help="resume an exact partial dataset or reuse an exact complete dataset",
    )
    args = parser.parse_args()

    if args.games < 2 or args.workers < 1 or args.sims < 1:
        raise SystemExit("games >= 2, workers >= 1, and sims >= 1 are required")
    if args.games_per_shard < 1:
        raise SystemExit("games-per-shard must be positive")
    if args.worlds < 1 or (
        args.teacher_kind == "determinized_puct" and args.worlds > args.sims
    ):
        raise SystemExit("worlds must be positive and no greater than PUCT sims")
    if args.quick_games < 2:
        raise SystemExit("quick-games must be >= 2 for a seat-balanced evaluation")
    if args.wall_cap_hours <= 0:
        raise SystemExit("wall-cap-hours must be positive")

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_dir = Path(args.out_dir).resolve()
    dataset_dir = out_dir / "dataset"
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = vars(args).copy()
    is_puct = args.teacher_kind == "determinized_puct"
    if is_puct:
        teacher_spec = {
            "kind": "determinized_puct",
            "sims": args.sims,
            "worlds": args.worlds,
            "c_puct": args.c_puct,
        }
        experiment = "puct-visit-vs-chosen-action-ablation-v1"
        teacher_algorithm = "determinized_puct_uniform_prior_random_leaf"
        policy_target_kind = VISIT_DISTRIBUTION_TARGET
        value_target_kind = ROOT_VALUE_TARGET
        explicit_limit = (
            "neural PUCT, public-belief search, or information-set consistency"
        )
        contract = PUCT_EXPERIMENT_CONTRACT
        arms = {
            "chosen_action": {
                "policy_target_kind": CHOSEN_ACTION_TARGET,
                "value_target_kind": ROOT_VALUE_TARGET,
                "value_weight": args.value_weight,
            },
            "visit_distribution": {
                "policy_target_kind": VISIT_DISTRIBUTION_TARGET,
                "value_target_kind": ROOT_VALUE_TARGET,
                "value_weight": args.value_weight,
            },
        }
    else:
        teacher_spec = {"kind": "search", "sims": args.sims}
        experiment = "search-supervised-policy-value-ablation-v1"
        teacher_algorithm = "flat_determinized_monte_carlo"
        policy_target_kind = SCORE_SOFTMAX_TARGET
        value_target_kind = TERMINAL_OUTCOME_TARGET
        explicit_limit = "MCTS visit-count supervision"
        contract = EXPERIMENT_CONTRACT
        arms = {
            "policy_only": {
                "policy_target_kind": SCORE_SOFTMAX_TARGET,
                "value_target_kind": TERMINAL_OUTCOME_TARGET,
                "value_weight": 0.0,
            },
            "policy_value": {
                "policy_target_kind": SCORE_SOFTMAX_TARGET,
                "value_target_kind": TERMINAL_OUTCOME_TARGET,
                "value_weight": args.value_weight,
            },
        }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment": experiment,
        "teacher_algorithm": teacher_algorithm,
        "target_kind": f"{policy_target_kind}_and_{value_target_kind}",
        "explicitly_not": explicit_limit,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": _git_commit(),
        "contract": contract,
        "caps": {
            "wall_hours": args.wall_cap_hours,
            "teacher_selfplay_games": args.games,
            "teacher_simulations": args.sims,
            "teacher_budget_semantics": (
                "total tree traversals per decision"
                if is_puct
                else "independent playouts per legal action"
            ),
            "teacher_worlds": args.worlds if is_puct else None,
            "training_epochs_per_arm": args.epochs,
            "evaluation_games_per_arm": args.quick_games,
            "teacher_gap_games_per_arm": args.student_teacher_games,
        },
        "trajectory_contract": {
            "kind": "encoded_decision_shard",
            "viewer_safe_observation": True,
            "legal_action_mask": True,
            "teacher_action": True,
            "per_action_search_scores": True,
            "root_visit_counts": is_puct,
            "root_value": is_puct,
            "terminal_outcome": True,
            "replayable_engine_trajectory": False,
            "limitation": (
                "Decision shards are not replayable engine trajectories. Teacher-1 "
                "adds real MCTS visits but remains determinization-based and uses "
                "uniform priors plus random leaves."
            ),
        },
        "pinned_runtime": _runtime_contract(args.seed, args.device),
        "config": config,
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)

    manifest["stages"]["teacher_probe"] = _matchup(
        teacher_spec,
        {"kind": "random"},
        games=args.teacher_probe_games,
        seed=args.seed + 10_000,
    )
    teacher_win_rate = float(manifest["stages"]["teacher_probe"].get("win_rate", 0.0))
    manifest["gates"] = {
        "teacher_signal": teacher_win_rate >= args.teacher_min_win_rate
    }
    _atomic_json(manifest_path, manifest)
    if not manifest["gates"]["teacher_signal"]:
        manifest["status"] = "stopped_teacher_gate"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
        raise SystemExit(
            f"teacher gate failed: {teacher_win_rate:.3f} < {args.teacher_min_win_rate:.3f}"
        )

    if _wall_cap_exceeded(started, args.wall_cap_hours):
        _stop_at_wall_cap(manifest, manifest_path, started)

    dataset_manifest_path = dataset_dir / "manifest.json"
    shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    if args.resume_dataset:
        if not dataset_manifest_path.exists():
            raise SystemExit("--resume-dataset requires dataset/manifest.json")
        _generate_dataset(args, dataset_dir, teacher_spec, resume=True)
        shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    elif shard_paths or dataset_manifest_path.exists():
        raise SystemExit(
            "dataset directory already contains artifacts; use a new --out-dir or "
            "--resume-dataset with an exactly matching manifest"
        )
    else:
        _generate_dataset(args, dataset_dir, teacher_spec)
        shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise SystemExit(f"dataset generation produced no shards under {dataset_dir}")
    generator_manifest = json.loads(dataset_manifest_path.read_text())
    if generator_manifest.get("status") != "completed":
        raise SystemExit(
            "dataset generator did not complete; rerun with --resume-dataset"
        )
    _validate_dataset_manifest(
        generator_manifest,
        args=args,
        teacher_spec=teacher_spec,
        policy_target_kind=policy_target_kind,
        value_target_kind=value_target_kind,
        shard_paths=shard_paths,
    )
    dataset = load_shards([str(path) for path in shard_paths])
    manifest["stages"]["dataset"] = {
        "generator_manifest": generator_manifest,
        "diagnostics": _dataset_diagnostics(
            dataset,
            policy_target_kind=policy_target_kind,
            temperature=args.policy_temperature,
        ),
        "shards": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in shard_paths
        ],
    }
    _atomic_json(manifest_path, manifest)

    if _wall_cap_exceeded(started, args.wall_cap_hours):
        _stop_at_wall_cap(manifest, manifest_path, started)

    manifest["stages"]["arms"] = {}
    for arm_index, (name, arm_spec) in enumerate(arms.items()):
        if _wall_cap_exceeded(started, args.wall_cap_hours):
            _stop_at_wall_cap(manifest, manifest_path, started)
        arm_policy_target = str(arm_spec["policy_target_kind"])
        arm_value_target = str(arm_spec["value_target_kind"])
        value_weight = float(arm_spec["value_weight"])
        print(
            f"[{name}] policy={arm_policy_target} value={arm_value_target} "
            f"value_weight={value_weight}",
            flush=True,
        )
        arm_started = time.perf_counter()
        agent, obs_space, initial_validation, history = train_search_supervised(
            dataset,
            policy_temperature=args.policy_temperature,
            policy_target_kind=arm_policy_target,
            value_target_kind=arm_value_target,
            value_weight=value_weight,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            log=True,
        )
        checkpoint = out_dir / f"{name}.pt"
        save_bc_checkpoint(
            agent,
            obs_space,
            checkpoint,
            extra={
                "experiment": manifest["experiment"],
                "arm": name,
                "value_weight": value_weight,
                "policy_temperature": args.policy_temperature,
                "policy_target_kind": arm_policy_target,
                "value_target_kind": arm_value_target,
                "dataset_shards": [
                    item["sha256"] for item in manifest["stages"]["dataset"]["shards"]
                ],
                "source_commit": manifest["source_commit"],
            },
        )
        final = history[-1].validation
        gameplay = _student_vs_random(
            checkpoint,
            games=args.quick_games,
            seed=args.seed + 20_000 + arm_index * 1_000,
            device=args.device,
        )
        teacher_gap = _matchup(
            {"kind": "checkpoint", "path": str(checkpoint), "name": name},
            teacher_spec,
            games=args.student_teacher_games,
            seed=args.seed + 30_000 + arm_index * 1_000,
        )
        arm_result = {
            "policy_target_kind": arm_policy_target,
            "value_target_kind": arm_value_target,
            "value_weight": value_weight,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "seconds": time.perf_counter() - arm_started,
            "initial_validation": asdict(initial_validation),
            "history": [asdict(epoch) for epoch in history],
            "gameplay_vs_random": gameplay,
            "gameplay_vs_teacher": teacher_gap,
            "gates": {
                "policy_loss_improved": final.policy_loss
                < initial_validation.policy_loss,
                "policy_beats_uniform": final.policy_accuracy_nontrivial
                > final.uniform_policy_probability,
                "value_beats_coin_brier": (
                    final.value_brier < 0.25 if value_weight > 0 else None
                ),
            },
        }
        manifest["stages"]["arms"][name] = arm_result
        _atomic_json(manifest_path, manifest)

    baseline_name, challenger_name = tuple(arms)
    baseline_win = manifest["stages"]["arms"][baseline_name]["gameplay_vs_random"][
        "win_rate"
    ]
    challenger_win = manifest["stages"]["arms"][challenger_name]["gameplay_vs_random"][
        "win_rate"
    ]
    comparison_gate = (
        "visit_policy_noninferior" if is_puct else "joint_policy_noninferior"
    )
    manifest["gates"][comparison_gate] = challenger_win >= baseline_win - 0.10
    required_arm_gates = [
        value
        for arm in manifest["stages"]["arms"].values()
        for value in arm["gates"].values()
        if value is not None
    ]
    manifest["status"] = (
        "completed_pass"
        if all(manifest["gates"].values()) and all(required_arm_gates)
        else "completed_diagnostic_failure"
    )
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_json(manifest_path, manifest)
    print(
        f"done: {manifest['status']} in {manifest['wall_seconds']:.0f}s -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
