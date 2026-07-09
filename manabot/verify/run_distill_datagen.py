"""Exp-03 (wave/search C4) Task 1: search-64 self-play dataset generation.

Plays search-vs-search self-play games in parallel worker processes; each
worker writes one .npz shard of (observation, search action) decisions plus a
manifest entry. Wall-clock and per-worker engine seconds are logged — that is
the teacher cost.

Usage:
    python -m manabot.verify.run_distill_datagen --games 480 --workers 4 \
        --sims 64 --out-dir .runs/exp03/dataset
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


def _worker(args: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from manabot.sim.distill import generate_selfplay_shard

    return generate_selfplay_shard(
        num_games=args["num_games"],
        sims=args["sims"],
        seed=args["seed"],
        game_offset=args["game_offset"],
        out_path=args["out_path"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=".runs/exp03/dataset")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    per_worker = args.games // args.workers
    remainder = args.games % args.workers
    offset = 0
    for w in range(args.workers):
        chunk = per_worker + (1 if w < remainder else 0)
        if chunk == 0:
            continue
        chunks.append(
            {
                "num_games": chunk,
                "sims": args.sims,
                "seed": args.seed + w * 1_000_000,
                "game_offset": offset,
                "out_path": str(out_dir / f"shard_{w:02d}.npz"),
            }
        )
        offset += chunk

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        summaries = list(pool.map(_worker, chunks))
    wall_seconds = time.perf_counter() - start

    engine_seconds = float(sum(s["wall_seconds"] for s in summaries))
    decisions = int(sum(s["decisions"] for s in summaries))
    steps = [step for s in summaries for step in s["steps_per_game"]]
    winners = [w for s in summaries for w in s["winners"]]
    manifest = {
        "teacher": summaries[0]["teacher"] if summaries else None,
        "games": int(sum(s["num_games"] for s in summaries)),
        "decisions": decisions,
        "wall_seconds": wall_seconds,
        "engine_seconds": engine_seconds,
        "engine_core_hours": engine_seconds / 3600.0,
        "workers": args.workers,
        "seed": args.seed,
        "sims": args.sims,
        "mean_steps_per_game": float(np.mean(steps)) if steps else 0.0,
        "seat0_win_rate": (
            float(np.mean([w == 0 for w in winners])) if winners else 0.0
        ),
        "search": {
            key: float(sum(s["search"][key] for s in summaries))
            for key in ("decisions", "seconds", "simulations", "cap_hits")
        },
        "shards": summaries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"done: {manifest['games']} games, {decisions} decisions, "
        f"wall {wall_seconds:.0f}s, engine {engine_seconds:.0f}s "
        f"({manifest['engine_core_hours']:.2f} core-hours) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
