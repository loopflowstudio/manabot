"""Exp-02 (wave/search C3) runner: flat determinized MC vs baselines.

Runs the pre-registered matchup matrix seat-balanced, in parallel across
processes, and writes one JSON blob per matchup so partial progress survives.

Usage:
    python -m manabot.verify.run_flat_mc --games 200 --workers 8 \
        --out reports/data/exp-02-flat-mc.json
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

CHECKPOINT_TEMPLATE = (
    "/Users/jack/src/manabot/.runs/first-light-c1-interactive-dev2-s{seed}-final/"
    "step_65536.pt"
)


def _worker(args: dict[str, Any]) -> dict[str, Any]:
    """Play a chunk of games in a subprocess; returns picklable results."""

    import torch

    torch.set_num_threads(1)
    from manabot.sim.flat_mc import play_games

    result = play_games(
        args["hero_spec"],
        args["villain_spec"],
        num_games=args["num_games"],
        seed=args["seed"],
        game_offset=args["game_offset"],
    )
    return {
        "records": [
            {
                "game_index": r.game_index,
                "hero_seat": r.hero_seat,
                "hero_won": r.hero_won,
                "winner": r.winner,
                "steps": r.steps,
            }
            for r in result.records
        ],
        "hero_search": (
            result.hero_search.to_dict() if result.hero_search else None
        ),
        "hero_decision_seconds": (
            result.hero_search.decision_seconds if result.hero_search else []
        ),
        "villain_search": (
            result.villain_search.to_dict() if result.villain_search else None
        ),
        "villain_decision_seconds": (
            result.villain_search.decision_seconds if result.villain_search else []
        ),
        "wall_seconds": result.wall_seconds,
    }


def run_matchup(
    hero_spec: dict[str, Any],
    villain_spec: dict[str, Any],
    *,
    num_games: int,
    workers: int,
    base_seed: int,
) -> dict[str, Any]:
    from manabot.sim.flat_mc import GameRecord, aggregate_records, spec_name

    chunks = []
    per_worker = num_games // workers
    remainder = num_games % workers
    offset = 0
    for w in range(workers):
        chunk = per_worker + (1 if w < remainder else 0)
        if chunk == 0:
            continue
        chunks.append(
            {
                "hero_spec": hero_spec,
                "villain_spec": villain_spec,
                "num_games": chunk,
                "seed": base_seed + w * 1_000_000,
                "game_offset": offset,
            }
        )
        offset += chunk

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        outputs = list(pool.map(_worker, chunks))
    wall = time.perf_counter() - start

    records = [
        GameRecord(**record) for out in outputs for record in out["records"]
    ]
    metrics = aggregate_records(records)

    def merge_search(key: str, ds_key: str) -> dict[str, Any] | None:
        stats = [out[key] for out in outputs if out[key] is not None]
        if not stats:
            return None
        seconds = [s for out in outputs for s in out[ds_key]]
        total = {
            name: float(sum(s[name] for s in stats))
            for name in ("decisions", "seconds", "simulations", "cap_hits")
        }
        total["mean_seconds_per_decision"] = (
            total["seconds"] / total["decisions"] if total["decisions"] else 0.0
        )
        total["median_seconds_per_decision"] = (
            float(np.median(seconds)) if seconds else 0.0
        )
        total["mean_sims_per_decision"] = (
            total["simulations"] / total["decisions"] if total["decisions"] else 0.0
        )
        total["cap_hit_rate"] = (
            total["cap_hits"] / total["simulations"] if total["simulations"] else 0.0
        )
        return total

    return {
        "hero": spec_name(hero_spec),
        "villain": spec_name(villain_spec),
        "hero_spec": hero_spec,
        "villain_spec": villain_spec,
        "metrics": metrics,
        "hero_search": merge_search("hero_search", "hero_decision_seconds"),
        "villain_search": merge_search("villain_search", "villain_decision_seconds"),
        "wall_seconds": wall,
        "engine_seconds": float(sum(out["wall_seconds"] for out in outputs)),
    }


def build_matchups(games: int) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    search_ns = [16, 64, 256]
    checkpoints = [
        {
            "kind": "checkpoint",
            "path": CHECKPOINT_TEMPLATE.format(seed=s),
            "name": f"c1v2-s{s}",
        }
        for s in (1, 2, 3)
    ]
    matchups: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    # 1. N-scaling vs random.
    for n in search_ns:
        matchups.append(({"kind": "search", "sims": n}, {"kind": "random"}, games))
    # 2. vs trained checkpoints.
    for n in search_ns:
        for ckpt in checkpoints:
            matchups.append(({"kind": "search", "sims": n}, ckpt, games))
    # 3. Monotonicity head-to-head.
    matchups.append(
        ({"kind": "search", "sims": 16}, {"kind": "search", "sims": 256}, games)
    )
    return matchups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default="reports/data/exp-02-flat-mc.json"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated list of matchup names (hero__villain) to run",
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    from manabot.sim.flat_mc import spec_name

    only = set(args.only.split(",")) if args.only else None
    matchups = build_matchups(args.games)
    for index, (hero_spec, villain_spec, games) in enumerate(matchups):
        name = f"{spec_name(hero_spec)}__{spec_name(villain_spec)}"
        if only is not None and name not in only:
            continue
        if name in results:
            print(f"[skip] {name} (already in {out_path})")
            continue
        print(f"[run ] {name}: {games} games, {args.workers} workers")
        result = run_matchup(
            hero_spec,
            villain_spec,
            num_games=games,
            workers=args.workers,
            base_seed=args.seed + index * 97,
        )
        results[name] = result
        out_path.write_text(json.dumps(results, indent=2))
        m = result["metrics"]
        search = result["hero_search"] or {}
        print(
            f"       win {m['win_rate']:.3f} "
            f"[{m['win_ci_lower']:.3f},{m['win_ci_upper']:.3f}] "
            f"play {m['win_rate_on_play']:.3f} draw {m['win_rate_on_draw']:.3f} "
            f"| {search.get('mean_seconds_per_decision', 0.0) * 1000:.0f} ms/dec "
            f"| wall {result['wall_seconds']:.0f}s"
        )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
