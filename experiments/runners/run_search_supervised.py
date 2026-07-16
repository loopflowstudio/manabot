"""Run the first diagnostic search-supervised policy/value experiment.

This runner deliberately says ``search``, not ``MCTS``. The current teacher is
flat determinized Monte Carlo with equal rollouts per root action. It provides
a score distribution and game outcome suitable for proving the joint
policy/value learning substrate that a later PUCT/MCTS teacher can reuse.

The two arms share data, initialization, split, optimizer, and policy targets:

- ``policy_only``: policy loss only;
- ``policy_value``: the same policy loss plus terminal value BCE.

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
    SCORE_KEY,
    _git_commit,
    load_shards,
    save_bc_checkpoint,
    soft_targets_from_scores,
)
from manabot.sim.flat_mc import aggregate_records, load_checkpoint_agent, play_games
from manabot.sim.rollout import BatchedSampler, RandomBatchController, run_vector_games
from manabot.sim.search_supervised import (
    SCORE_SOFTMAX_TARGET,
    TERMINAL_OUTCOME_TARGET,
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
    dataset: dict[str, np.ndarray], *, temperature: float
) -> dict[str, Any]:
    scores = torch.as_tensor(dataset[SCORE_KEY], dtype=torch.float32)
    targets = soft_targets_from_scores(scores, temperature)
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(dim=-1)
    actions = np.asarray(dataset["action"], dtype=np.int64)
    valid = np.asarray(dataset["actions_valid"]) > 0
    in_range = (actions >= 0) & (actions < valid.shape[1])
    legal = np.zeros(len(actions), dtype=bool)
    legal[in_range] = valid[np.arange(len(actions))[in_range], actions[in_range]]
    sorted_scores = np.sort(
        np.where(scores.numpy() >= 0, scores.numpy(), -np.inf), axis=1
    )
    if scores.shape[1] > 1:
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        margins = margins[np.isfinite(margins)]
    else:
        margins = np.asarray([], dtype=np.float32)
    return {
        "decisions": int(len(dataset["action"])),
        "games": int(len(np.unique(dataset["game_index"]))),
        "mean_valid_actions": float(np.mean(dataset["num_valid"])),
        "policy_temperature": temperature,
        "mean_target_entropy": float(entropy.mean().item()),
        "mean_top_two_score_margin": float(np.mean(margins)) if len(margins) else None,
        "winnerless_rows": int(np.sum(dataset["winner"] < 0)),
        "invalid_teacher_actions": int(np.sum(~legal)),
        "action_mask_width": int(valid.shape[1]),
    }


def _generate_dataset(args: argparse.Namespace, dataset_dir: Path) -> None:
    command = [
        sys.executable,
        "experiments/runners/run_distill_datagen.py",
        "--games",
        str(args.games),
        "--workers",
        str(args.workers),
        "--sims",
        str(args.sims),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(dataset_dir),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--sims", type=int, default=128)
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
        help="reuse complete shard data already present under OUT_DIR/dataset",
    )
    args = parser.parse_args()

    if args.games < 2 or args.workers < 1 or args.sims < 1:
        raise SystemExit("games >= 2, workers >= 1, and sims >= 1 are required")
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
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment": "search-supervised-policy-value-ablation-v1",
        "teacher_algorithm": "flat_determinized_monte_carlo",
        "target_kind": "masked_score_softmax_and_terminal_outcome",
        "explicitly_not": "MCTS visit-count supervision",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": _git_commit(),
        "contract": EXPERIMENT_CONTRACT,
        "caps": {
            "wall_hours": args.wall_cap_hours,
            "teacher_selfplay_games": args.games,
            "teacher_simulations_per_legal_action": args.sims,
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
            "terminal_outcome": True,
            "replayable_engine_trajectory": False,
            "limitation": (
                "Teacher-0 proves the runner and learner substrate; W2-234 still "
                "requires offer/command trajectories and real MCTS visit targets."
            ),
        },
        "pinned_runtime": _runtime_contract(args.seed, args.device),
        "config": config,
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)

    teacher_spec = {"kind": "search", "sims": args.sims}
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

    shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    if not (
        args.resume_dataset and shard_paths and (dataset_dir / "manifest.json").exists()
    ):
        _generate_dataset(args, dataset_dir)
        shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise SystemExit(f"dataset generation produced no shards under {dataset_dir}")
    dataset = load_shards([str(path) for path in shard_paths])
    manifest["stages"]["dataset"] = {
        "generator_manifest": json.loads((dataset_dir / "manifest.json").read_text()),
        "diagnostics": _dataset_diagnostics(
            dataset, temperature=args.policy_temperature
        ),
        "shards": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in shard_paths
        ],
    }
    _atomic_json(manifest_path, manifest)

    if _wall_cap_exceeded(started, args.wall_cap_hours):
        _stop_at_wall_cap(manifest, manifest_path, started)

    arms: dict[str, float] = {
        "policy_only": 0.0,
        "policy_value": args.value_weight,
    }
    manifest["stages"]["arms"] = {}
    for arm_index, (name, value_weight) in enumerate(arms.items()):
        if _wall_cap_exceeded(started, args.wall_cap_hours):
            _stop_at_wall_cap(manifest, manifest_path, started)
        print(f"[{name}] value_weight={value_weight}", flush=True)
        arm_started = time.perf_counter()
        agent, obs_space, initial_validation, history = train_search_supervised(
            dataset,
            policy_temperature=args.policy_temperature,
            policy_target_kind=SCORE_SOFTMAX_TARGET,
            value_target_kind=TERMINAL_OUTCOME_TARGET,
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
                "policy_target_kind": SCORE_SOFTMAX_TARGET,
                "value_target_kind": TERMINAL_OUTCOME_TARGET,
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

    policy_win = manifest["stages"]["arms"]["policy_only"]["gameplay_vs_random"][
        "win_rate"
    ]
    joint_win = manifest["stages"]["arms"]["policy_value"]["gameplay_vs_random"][
        "win_rate"
    ]
    manifest["gates"]["joint_policy_noninferior"] = joint_win >= policy_win - 0.10
    required_arm_gates = [
        manifest["stages"]["arms"][name]["gates"][gate]
        for name, gate in (
            ("policy_only", "policy_loss_improved"),
            ("policy_only", "policy_beats_uniform"),
            ("policy_value", "policy_loss_improved"),
            ("policy_value", "policy_beats_uniform"),
            ("policy_value", "value_beats_coin_brier"),
        )
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
