"""Exp-07 (wave/search C7) Tasks 3/4: distill a student from search self-play.

Sweeps the soft-target temperature (plus a hard-argmax control), trains one
student per config, quick-evaluates each vs random with the batched vector
driver (seat-balanced), and keeps the best by win rate (val accuracy as the
tiebreak). All sweep wall-clock is billed to the round's training cost.

Usage:
    python -m manabot.verify.run_exp07_bc --data-dir .runs/exp07/dataset_r0 \
        --out .runs/exp07/student_r0.pt --log .runs/exp07/bc_r0.json \
        --taus 0.03,0.05,0.1 --include-hard --quick-games 200 --device mps
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import time


def quick_eval_vs_random(
    checkpoint_path: str,
    *,
    num_games: int,
    seed: int,
    device: str,
    num_streams: int = 128,
) -> dict[str, float]:
    """Seat-balanced student (stochastic) vs random via the batched driver."""

    from manabot.sim.flat_mc import aggregate_records, load_checkpoint_agent
    from manabot.sim.rollout import (
        BatchedSampler,
        RandomBatchController,
        run_vector_games,
    )

    agent, _ = load_checkpoint_agent(checkpoint_path)
    sampler = BatchedSampler(agent, deterministic=False, seed=seed, device=device)
    records, stats = run_vector_games(
        sampler,
        RandomBatchController(seed=seed + 1),
        num_games=num_games,
        num_streams=num_streams,
        seed=seed,
    )
    metrics = aggregate_records(records)
    metrics["wall_seconds"] = stats["wall_seconds"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--round", type=int, default=0, help="round tag for --data-dir shards"
    )
    parser.add_argument(
        "--extra-data-dir",
        type=str,
        default=None,
        help="optional second shard dir (multi-round aggregate training)",
    )
    parser.add_argument("--extra-round", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--taus", type=str, default="0.03,0.05,0.1")
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick-games", type=int, default=200)
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="checkpoint to warm-start from (fine-tune) instead of fresh init",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch

    from manabot.sim.distill import load_shards, save_bc_checkpoint, train_bc

    shard_paths = sorted(glob.glob(str(Path(args.data_dir) / "shard_*.npz")))
    if not shard_paths:
        raise SystemExit(f"no shards found under {args.data_dir}")
    rounds = [args.round] * len(shard_paths)
    if args.extra_data_dir:
        extra = sorted(glob.glob(str(Path(args.extra_data_dir) / "shard_*.npz")))
        if not extra:
            raise SystemExit(f"no shards found under {args.extra_data_dir}")
        shard_paths = shard_paths + extra
        rounds = rounds + [args.extra_round] * len(extra)

    sweep_start = time.perf_counter()
    dataset = load_shards(shard_paths, rounds=rounds)
    print(
        f"loaded {len(dataset['action'])} decisions from {len(shard_paths)} shards "
        f"(rounds {sorted(set(rounds))})",
        flush=True,
    )

    initial_state = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        initial_state = checkpoint["model_state_dict"]
        print(f"warm-starting from {args.init_from}", flush=True)

    configs: list[tuple[str, float | None]] = []
    if args.include_hard:
        configs.append(("hard", None))
    configs.extend((f"tau{tau}", float(tau)) for tau in args.taus.split(",") if tau)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    best = None
    for name, tau in configs:
        print(f"[bc] config={name} lr={args.lr} epochs={args.epochs}", flush=True)
        run_start = time.perf_counter()
        agent, obs_space, history = train_bc(
            dataset,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            soft_temperature=tau,
            initial_agent_state=initial_state,
            log=True,
        )
        train_seconds = time.perf_counter() - run_start
        final = history[-1]

        candidate_path = str(out_dir / f"bc_{name}.pt")
        save_bc_checkpoint(
            agent,
            obs_space,
            candidate_path,
            extra={
                "config": name,
                "soft_temperature": tau,
                "lr": args.lr,
                "epochs": args.epochs,
                "val_accuracy": final.val_accuracy,
                "dataset_decisions": len(dataset["action"]),
                "init_from": args.init_from,
            },
        )
        print(f"[eval] {name}: {args.quick_games} games vs random", flush=True)
        quick = quick_eval_vs_random(
            candidate_path,
            num_games=args.quick_games,
            seed=args.seed + 4242,
            device=args.device,
        )
        run_seconds = time.perf_counter() - run_start
        print(
            f"       win {quick['win_rate']:.3f} "
            f"[{quick['win_ci_lower']:.3f},{quick['win_ci_upper']:.3f}] "
            f"val_acc {final.val_accuracy:.4f} | {run_seconds:.0f}s",
            flush=True,
        )
        run_entry = {
            "config": name,
            "soft_temperature": tau,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_seconds": train_seconds,
            "wall_seconds": run_seconds,
            "checkpoint": candidate_path,
            "quick_eval": quick,
            "history": [asdict(h) for h in history],
        }
        runs.append(run_entry)
        key = (quick["win_rate"], final.val_accuracy)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "config": name,
                "soft_temperature": tau,
                "checkpoint": candidate_path,
                "win_rate": quick["win_rate"],
                "val_accuracy": final.val_accuracy,
            }

    # Promote the winner to the requested output path.
    import shutil

    shutil.copyfile(best["checkpoint"], args.out)
    total_seconds = time.perf_counter() - sweep_start
    best_out = {k: v for k, v in best.items() if k != "key"}
    log = {
        "dataset_dir": args.data_dir,
        "decisions": int(len(dataset["action"])),
        "device": args.device,
        "seed": args.seed,
        "init_from": args.init_from,
        "sweep_wall_seconds": total_seconds,
        "runs": runs,
        "best": best_out,
        "student": args.out,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(
        f"done: best={best['config']} win={best['win_rate']:.3f} "
        f"val_acc={best['val_accuracy']:.4f} | sweep wall {total_seconds:.0f}s "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
