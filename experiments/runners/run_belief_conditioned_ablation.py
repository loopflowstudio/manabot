"""Matched belief-conditioned vs unconditioned policy/value ablation (INT-14).

A 2×2 factorial ablation through the existing arena path. All four arms share
ONE conditioned ``Agent`` architecture (``AgentHypers.max_conditions = K``) and
train on the SAME D×K expanded conditional rows with the SAME per-row policy
targets. The only differences are:

- **conditioning dimension**: unconditioned arms see a NEUTRAL condition input
  (``condition_index=0, condition_weight=1.0``); conditioned arms see the REAL
  per-row condition (``condition_index=k, condition_weight=1/K``).
- **value dimension**: ``value_weight`` 0 (policy_only) vs 1 (policy_value).

Inference uses the neutral condition: the arena ``ObservationSpace`` does not
produce condition keys, so the ``Agent`` defaults to the uninformative prior
(``condition_index=0, condition_weight=1.0``). All arms evaluate under this
neutral condition — the train/test mismatch the ~0 prediction covers.

The toy condition is a uniform determinization — an uninformative belief. The
pre-registered prediction is **~0 strength gap** vs the non-conditioned
baseline. The deliverable is a plumbing + measurement-integrity receipt, NOT a
strength claim. ``CLAIM_BOUNDARY.strength_claim`` is False; no learned belief
head, range net, or per-hand value vector is added; the policy and scalar value
heads are unchanged. Hidden truth never enters policy/value inputs: the
condition is a public tag (``condition_index``), not the opponent's hand.

Usage
-----
``--frozen`` loads shards from a frozen conditional snapshot directory (the
checked-in evidence under ``experiments/data/``). Without ``--frozen`` a tiny
toy dataset is generated on the fly via
``conditional_distill.build_conditional_dataset`` and optionally frozen.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

from experiments.runners.run_belief_conditioned_snapshot import (
    CLAIM_BOUNDARY,
    freeze_conditional_snapshot,
    verify_conditional_snapshot,
)
from manabot.infra.hypers import AgentHypers
from manabot.sim.conditional_distill import (
    CONDITION_ROLES,
    build_conditional_dataset,
    load_conditional_shards,
    with_neutral_condition,
)
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import aggregate_records, load_checkpoint_agent, play_games
from manabot.sim.rollout import BatchedSampler, RandomBatchController, run_vector_games
from manabot.sim.search_supervised import (
    SCORE_SOFTMAX_TARGET,
    TERMINAL_OUTCOME_TARGET,
    train_search_supervised,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = (
    REPO_ROOT / "experiments" / "data" / "int-14-belief-conditioned-snapshot-v1"
)

ARMS: dict[str, dict[str, Any]] = {
    "policy_only": {"conditioned": False, "value_weight": 0.0},
    "policy_value": {"conditioned": False, "value_weight": 1.0},
    "belief_conditioned_policy_only": {"conditioned": True, "value_weight": 0.0},
    "belief_conditioned_policy_value": {"conditioned": True, "value_weight": 1.0},
}

PREREGISTERED_PREDICTION = (
    "~0 strength gap: the toy condition is a uniform determinization "
    "(uninformative belief); conditioning on it is expected to show no "
    "strength change versus the non-conditioned baseline within Wilson-CI "
    "noise. A non-zero gap is a flagged falsification candidate, not a win."
)


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


def _matchup(
    hero: dict[str, Any], villain: dict[str, Any], *, games: int, seed: int
) -> dict[str, Any]:
    if games <= 0:
        return {"skipped": True, "num_games": 0}
    result = play_games(hero, villain, num_games=games, seed=seed)
    metrics: dict[str, Any] = aggregate_records(result.records)
    metrics["wall_seconds"] = result.wall_seconds
    return metrics


def _load_frozen_shards(snapshot_dir: Path) -> list[str]:
    npz_paths = sorted(snapshot_dir.glob("shard_*.npz"))
    if not npz_paths:
        raise SystemExit(f"frozen snapshot has no shards: {snapshot_dir}")
    return [str(p) for p in npz_paths]


def _gap(conditioned: dict[str, Any], unconditioned: dict[str, Any]) -> dict[str, Any]:
    c_wr = float(conditioned["win_rate"])
    u_wr = float(unconditioned["win_rate"])
    return {
        "win_rate_delta": c_wr - u_wr,
        "conditioned_win_rate": c_wr,
        "unconditioned_win_rate": u_wr,
        "conditioned_ci": [
            conditioned["win_ci_lower"],
            conditioned["win_ci_upper"],
        ],
        "unconditioned_ci": [
            unconditioned["win_ci_lower"],
            unconditioned["win_ci_upper"],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--frozen", action="store_true", help="load the checked-in snapshot"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--freeze-snapshot", action="store_true", help="freeze the generated dataset"
    )
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--games-per-shard", type=int, default=8)
    parser.add_argument("--sims", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy-temperature", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--cross-games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    started = time.perf_counter()
    k = len(CONDITION_ROLES)

    if args.frozen:
        snapshot_dir = args.snapshot_dir.resolve()
        if not snapshot_dir.exists():
            raise SystemExit(
                f"--frozen requires a checked-in snapshot at {snapshot_dir}"
            )
        shard_paths = _load_frozen_shards(snapshot_dir)
        snapshot_manifest = json.loads((snapshot_dir / "snapshot.json").read_text())
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=snapshot_manifest["snapshot_identity_sha256"],
        )
        dataset_source = {"frozen_snapshot": str(snapshot_dir)}
    else:
        dataset_dir = out_dir / "dataset"
        build_conditional_dataset(
            dataset_dir,
            num_games=args.games,
            games_per_shard=args.games_per_shard,
            sims=args.sims,
            seed=args.seed,
        )
        shard_paths = sorted(str(p) for p in sorted(dataset_dir.glob("shard_*.npz")))
        dataset_source = {"generated_dataset": str(dataset_dir)}
        if args.freeze_snapshot:
            snap_dir = out_dir / "snapshot"
            snap = freeze_conditional_snapshot(
                dataset_dir, snap_dir, shard_count=len(shard_paths)
            )
            dataset_source["frozen_snapshot"] = {
                "path": str(snap_dir),
                "identity": snap["snapshot_identity_sha256"],
            }

    real_dataset = load_conditional_shards(shard_paths)
    neutral_dataset = with_neutral_condition(real_dataset)
    decisions = len(real_dataset["action"]) // k
    label_cost = len(real_dataset["action"])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "belief_conditioned_policy_value_ablation_v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistered_prediction": PREREGISTERED_PREDICTION,
        "strength_claim": False,
        "condition_roles": list(CONDITION_ROLES),
        "condition_count": k,
        "condition_source": CLAIM_BOUNDARY["condition_source"],
        "policy_target_kind": SCORE_SOFTMAX_TARGET,
        "value_target_kind": TERMINAL_OUTCOME_TARGET,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_source": dataset_source,
        "decisions": decisions,
        "label_cost": label_cost,
        "label_cost_note": (
            "D×K expanded rows (D decisions × K conditions); the toy "
            "condition is uninformative so label cost measures plumbing, "
            "not information gain."
        ),
        "config": vars(args).copy(),
        "arms": {},
    }
    _atomic_write(manifest_path, manifest)

    checkpoints: dict[str, Path] = {}
    for arm_index, (name, spec) in enumerate(ARMS.items()):
        dataset = real_dataset if spec["conditioned"] else neutral_dataset
        value_weight = float(spec["value_weight"])
        print(
            f"[{name}] conditioned={spec['conditioned']} value_weight={value_weight}",
            flush=True,
        )
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
            agent_hypers=AgentHypers(max_conditions=k),
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
                "conditioned": bool(spec["conditioned"]),
                "value_weight": value_weight,
                "max_conditions": k,
            },
        )
        checkpoints[name] = checkpoint
        gameplay = _student_vs_random(
            checkpoint,
            games=args.arena_games,
            seed=args.seed + 20_000 + arm_index * 1_000,
            device=args.device,
        )
        manifest["arms"][name] = {
            "conditioned": bool(spec["conditioned"]),
            "value_weight": value_weight,
            "checkpoint": str(checkpoint),
            "initial_validation": _asdict_safe(initial_validation),
            "history": [_asdict_safe(h) for h in history],
            "gameplay_vs_random": gameplay,
            "seconds": time.perf_counter() - arm_started,
        }
        _atomic_write(manifest_path, manifest)
        print(
            f"  win_rate_vs_random={gameplay['win_rate']:.3f} "
            f"ci=[{gameplay['win_ci_lower']:.3f},{gameplay['win_ci_upper']:.3f}]",
            flush=True,
        )

    # Cross-arm matchups: conditioned vs unconditioned at each value dimension.
    cross: dict[str, Any] = {}
    for value_label, pair in (
        ("policy_only", ("policy_only", "belief_conditioned_policy_only")),
        ("policy_value", ("policy_value", "belief_conditioned_policy_value")),
    ):
        uncond_name, cond_name = pair
        if args.cross_games > 0:
            result = _matchup(
                {
                    "kind": "checkpoint",
                    "path": str(checkpoints[cond_name]),
                    "name": cond_name,
                },
                {
                    "kind": "checkpoint",
                    "path": str(checkpoints[uncond_name]),
                    "name": uncond_name,
                },
                games=args.cross_games,
                seed=args.seed + 40_000,
            )
            cross[value_label] = {
                "conditioned_vs_unconditioned": result,
                "gap_vs_random": _gap(
                    manifest["arms"][cond_name]["gameplay_vs_random"],
                    manifest["arms"][uncond_name]["gameplay_vs_random"],
                ),
            }
        else:
            cross[value_label] = {
                "gap_vs_random": _gap(
                    manifest["arms"][cond_name]["gameplay_vs_random"],
                    manifest["arms"][uncond_name]["gameplay_vs_random"],
                )
            }
    manifest["cross_arm"] = cross

    # Summary gap table + pre-registered statement.
    gaps = {
        label: data["gap_vs_random"]["win_rate_delta"] for label, data in cross.items()
    }
    manifest["gaps"] = gaps
    manifest["preregistered_statement"] = (
        f"Pre-registered prediction: {PREREGISTERED_PREDICTION} "
        f"Measured gaps (conditioned - unconditioned win rate vs random): "
        f"{gaps}. No strength claim is made (CLAIM_BOUNDARY.strength_claim=False)."
    )
    manifest["status"] = "completed"
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_write(manifest_path, manifest)
    print(
        f"done: {manifest['status']} in {manifest['wall_seconds']:.0f}s -> {manifest_path}",
        flush=True,
    )
    print(f"gaps: {gaps}", flush=True)
    print(f"preregistered: {PREREGISTERED_PREDICTION}", flush=True)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"not JSON serializable: {type(obj).__name__}")

    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_default)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _asdict_safe(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


if __name__ == "__main__":
    main()
