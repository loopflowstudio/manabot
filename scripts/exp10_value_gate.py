"""exp-10 (wave/search C10): the goal-4 gate — search-with-V vs V-greedy.

Subcommands:
  bc           Distill a BC student from search-256 shards (exp-07 recipe).
  train-value  Fit the value head on terminal outcomes from the shards.
  assess       V vs rollout ground truth: Spearman, calibration, buckets.
  match        Seat-balanced head-to-head between any two player specs.

All Python through uv:  uv run scripts/exp10_value_gate.py <cmd> ...
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np


def _shard_paths(shards: str) -> list[str]:
    if Path(shards).is_dir():
        paths = sorted(glob.glob(str(Path(shards) / "*.npz")))
    else:
        paths = sorted(glob.glob(shards))
    if not paths:
        raise SystemExit(f"no shards found at {shards}")
    return paths


def cmd_bc(args: argparse.Namespace) -> None:
    import torch

    from manabot.sim.distill import load_shards, save_bc_checkpoint, train_bc

    torch.manual_seed(args.seed)
    dataset = load_shards(_shard_paths(args.shards))
    print(f"dataset: {len(dataset['action'])} decisions")
    agent, obs_space, history = train_bc(
        dataset,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        log=True,
    )
    save_bc_checkpoint(
        agent,
        obs_space,
        args.out,
        extra={"history": [asdict(h) for h in history], "shards": args.shards},
    )
    print(f"saved {args.out}")


def cmd_train_value(args: argparse.Namespace) -> None:
    import torch

    from manabot.sim.distill import load_shards
    from manabot.sim.value import save_value_checkpoint, train_value

    dataset = load_shards(_shard_paths(args.shards))
    init_state = None
    if args.init:
        init_state = torch.load(args.init, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
    agent, obs_space, history = train_value(
        dataset,
        init_state=init_state,
        freeze_encoder=args.freeze_encoder,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        log=True,
    )
    save_value_checkpoint(
        agent,
        obs_space,
        args.out,
        extra={
            "history": [asdict(h) for h in history],
            "shards": args.shards,
            "init": args.init,
            "freeze_encoder": args.freeze_encoder,
        },
    )
    print(f"saved {args.out}")


def cmd_assess(args: argparse.Namespace) -> None:
    from manabot.sim.value import bucket_report, collect_value_assessment, spearman

    t0 = time.perf_counter()
    assessment = collect_value_assessment(
        value_checkpoint=args.value,
        behavior_checkpoint=args.behavior,
        num_games=args.games,
        sample_rate=args.sample_rate,
        gt_worlds=args.gt_worlds,
        gt_rollouts=args.gt_rollouts,
        seed=args.seed,
        device=args.device,
    )
    wall = time.perf_counter() - t0
    v, gt = assessment["v"], assessment["gt"]
    bins = np.clip((v * 10).astype(int), 0, 9)
    calibration = [
        {
            "bin": f"[{b/10:.1f},{(b+1)/10:.1f})",
            "n": int((bins == b).sum()),
            "mean_v": float(v[bins == b].mean()) if (bins == b).any() else None,
            "mean_gt": float(gt[bins == b].mean()) if (bins == b).any() else None,
        }
        for b in range(10)
    ]
    summary = {
        "states": int(len(v)),
        "games": args.games,
        "wall_seconds": wall,
        "spearman_v_gt": spearman(v, gt),
        "spearman_v_outcome": spearman(v, assessment["won"]),
        "pearson_v_gt": float(np.corrcoef(v, gt)[0, 1]),
        "mean_bias": float((v - gt).mean()),
        "mae": float(np.abs(v - gt).mean()),
        "calibration": calibration,
        "buckets": bucket_report(assessment),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        np.savez_compressed(Path(args.out).with_suffix(".npz"), **assessment)
        print(f"saved {args.out}")


def _match_worker(payload: tuple) -> dict:
    hero_spec, villain_spec, num_games, seed, game_offset = payload
    from manabot.sim.flat_mc import play_games

    result = play_games(
        hero_spec,
        villain_spec,
        num_games=num_games,
        seed=seed,
        game_offset=game_offset,
    )
    return {
        "records": [asdict(r) for r in result.records],
        "hero_search": result.hero_search.to_dict() if result.hero_search else None,
        "villain_search": (
            result.villain_search.to_dict() if result.villain_search else None
        ),
        "wall_seconds": result.wall_seconds,
    }


def cmd_match(args: argparse.Namespace) -> None:
    from manabot.sim.flat_mc import GameRecord, aggregate_records, spec_name

    hero_spec = json.loads(args.hero)
    villain_spec = json.loads(args.villain)
    workers = max(1, args.workers)
    per_worker = [args.games // workers] * workers
    for i in range(args.games % workers):
        per_worker[i] += 1
    offsets = np.cumsum([0] + per_worker[:-1]).tolist()
    payloads = [
        (hero_spec, villain_spec, games, args.seed + 1000 * w, offset)
        for w, (games, offset) in enumerate(zip(per_worker, offsets))
        if games > 0
    ]
    t0 = time.perf_counter()
    if workers == 1:
        chunks = [_match_worker(payloads[0])]
    else:
        with mp.Pool(workers) as pool:
            chunks = pool.map(_match_worker, payloads)
    wall = time.perf_counter() - t0

    records = [
        GameRecord(**r) for chunk in chunks for r in chunk["records"]
    ]

    def _sum_stats(key: str) -> dict | None:
        stats = [c[key] for c in chunks if c[key]]
        if not stats:
            return None
        out = {k: sum(s[k] for s in stats) for k in stats[0]}
        out["ms_per_decision"] = 1000.0 * out["seconds"] / max(1.0, out["decisions"])
        return out

    summary = {
        "hero": spec_name(hero_spec),
        "villain": spec_name(villain_spec),
        "hero_spec": hero_spec,
        "villain_spec": villain_spec,
        "wall_seconds": wall,
        "workers": workers,
        "metrics": aggregate_records(records),
        "hero_search": _sum_stats("hero_search"),
        "villain_search": _sum_stats("villain_search"),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"saved {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bc", help="distill BC student from shards")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_bc)

    p = sub.add_parser("train-value", help="fit value head on outcomes")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_train_value)

    p = sub.add_parser("assess", help="V vs rollout ground truth + buckets")
    p.add_argument("--value", required=True)
    p.add_argument("--behavior", required=True)
    p.add_argument("--games", type=int, default=80)
    p.add_argument("--sample-rate", type=float, default=0.06)
    p.add_argument("--gt-worlds", type=int, default=16)
    p.add_argument("--gt-rollouts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_assess)

    p = sub.add_parser("match", help="seat-balanced head-to-head")
    p.add_argument("--hero", required=True, help="player spec JSON")
    p.add_argument("--villain", required=True, help="player spec JSON")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_match)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
