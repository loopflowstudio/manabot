"""Exp-03 (wave/search C4) Task 2: behavior-clone a fresh Agent on search data.

Runs a small (lr x epochs) sweep — ALL of it is billed to the BC cost — picks
the config with the best validation accuracy, saves that policy as a
trainer-format checkpoint, and writes a JSON log with per-epoch curves and
exact wall-clock.

Usage:
    python -m manabot.verify.run_distill_bc --data-dir .runs/exp03/dataset \
        --out .runs/exp03/bc_policy.pt --log .runs/exp03/bc_log.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=".runs/exp03/dataset")
    parser.add_argument("--out", type=str, default=".runs/exp03/bc_policy.pt")
    parser.add_argument("--log", type=str, default=".runs/exp03/bc_log.json")
    parser.add_argument("--lrs", type=str, default="1e-3,3e-4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max-game-index",
        type=int,
        default=None,
        help="train on games with index < N only (cost-fraction studies)",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch

    from manabot.sim.distill import load_shards, save_bc_checkpoint, train_bc

    shard_paths = sorted(glob.glob(str(Path(args.data_dir) / "shard_*.npz")))
    if not shard_paths:
        raise SystemExit(f"no shards found under {args.data_dir}")

    sweep_start = time.perf_counter()
    dataset = load_shards(shard_paths)
    if args.max_game_index is not None:
        keep = dataset["game_index"] < args.max_game_index
        dataset = {key: value[keep] for key, value in dataset.items()}
    load_seconds = time.perf_counter() - sweep_start
    print(
        f"loaded {len(dataset['action'])} decisions from {len(shard_paths)} shards "
        f"in {load_seconds:.0f}s",
        flush=True,
    )

    lrs = [float(x) for x in args.lrs.split(",")]
    runs = []
    best = None
    for lr in lrs:
        print(f"[bc] lr={lr} epochs={args.epochs}", flush=True)
        run_start = time.perf_counter()
        agent, obs_space, history = train_bc(
            dataset,
            lr=lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            log=True,
        )
        run_seconds = time.perf_counter() - run_start
        final = history[-1]
        runs.append(
            {
                "lr": lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "wall_seconds": run_seconds,
                "history": [asdict(h) for h in history],
            }
        )
        if best is None or final.val_accuracy > best["val_accuracy"]:
            best = {
                "lr": lr,
                "val_accuracy": final.val_accuracy,
                "val_accuracy_nontrivial": final.val_accuracy_nontrivial,
                "val_loss": final.val_loss,
            }
            num_params = sum(p.numel() for p in agent.parameters())
            save_bc_checkpoint(
                agent,
                obs_space,
                args.out,
                extra={
                    "lr": lr,
                    "epochs": args.epochs,
                    "val_accuracy": final.val_accuracy,
                    "dataset_decisions": len(dataset["action"]),
                },
            )

    total_seconds = time.perf_counter() - sweep_start
    log = {
        "dataset_dir": args.data_dir,
        "decisions": int(len(dataset["action"])),
        "num_params": int(num_params),
        "device": args.device,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "sweep_wall_seconds": total_seconds,
        "load_seconds": load_seconds,
        "runs": runs,
        "best": best,
        "checkpoint": args.out,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(
        f"done: best lr={best['lr']} val_acc={best['val_accuracy']:.4f} "
        f"(nontrivial {best['val_accuracy_nontrivial']:.4f}) | "
        f"sweep wall {total_seconds:.0f}s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
