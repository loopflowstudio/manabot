"""Generate attributable search self-play shards in parallel.

Each worker writes viewer observations, legal masks, chosen actions, outcomes,
and teacher-specific targets. Flat Monte Carlo emits action scores; tree search
also emits root visits and values. Wall-clock, engine time, simulations, and
tree-growth diagnostics make label cost explicit.

Usage:
    uv run experiments/runners/run_distill_datagen.py --games 480 --workers 4 \
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
        teacher_spec=args.get("teacher_spec"),
        seed=args["seed"],
        game_offset=args["game_offset"],
        out_path=args["out_path"],
        round_index=args.get("round_index", 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument(
        "--teacher-json",
        type=str,
        default=None,
        help='full teacher spec as JSON, e.g. \'{"kind": "policy_search", '
        '"sims": 16, "checkpoint": "/abs/x.pt"}\' (overrides --sims)',
    )
    parser.add_argument(
        "--round", type=int, default=0, help="expert-iteration round tag"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=".runs/exp03/dataset")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_spec = json.loads(args.teacher_json) if args.teacher_json else None

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
                "teacher_spec": teacher_spec,
                "round_index": args.round,
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
        "provenance": summaries[0].get("provenance") if summaries else None,
        "policy_target_kind": (
            summaries[0].get("policy_target_kind") if summaries else None
        ),
        "value_target_kind": (
            summaries[0].get("value_target_kind") if summaries else None
        ),
        "round": args.round,
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
    if summaries and "tree_nodes" in summaries[0]["search"]:
        for key in ("tree_nodes", "worlds_sampled", "max_depth_sum"):
            manifest["search"][key] = float(
                sum(s["search"][key] for s in summaries)
            )
        manifest["search"]["max_depth_max"] = float(
            max(s["search"]["max_depth_max"] for s in summaries)
        )
        manifest["search"]["mean_max_depth"] = (
            manifest["search"]["max_depth_sum"] / max(1.0, decisions)
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"done: {manifest['games']} games, {decisions} decisions, "
        f"wall {wall_seconds:.0f}s, engine {engine_seconds:.0f}s "
        f"({manifest['engine_core_hours']:.2f} core-hours) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
