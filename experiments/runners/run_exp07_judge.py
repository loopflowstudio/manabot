"""Exp-07 (wave/intelligence C7): judge a distilled student (standard protocol).

Matrix per student:
    1. student vs random — seat-balanced, batched vector driver;
    2. ladder rungs: student vs search-{8,16,32,64} (process-parallel);
    3. behavioral profile (cast_when_able / passed_when_able, exp-01 metrics);
    4. optional head-to-head vs another checkpoint (batched).

Policies sample stochastically from masked softmax — the exp-00c/01/02/03
protocol.

Usage:
    uv run experiments/runners/run_exp07_judge.py --student .runs/exp07/student_r0.pt \
        --name r0 --out experiments/data/exp-07-expert-iteration.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def batched_matchup(
    hero_path: str,
    villain_path: str | None,
    *,
    num_games: int,
    seed: int,
    device: str,
    num_streams: int = 128,
) -> dict[str, Any]:
    """Seat-balanced checkpoint (stochastic) vs random or another checkpoint."""

    from manabot.sim.flat_mc import aggregate_records, load_checkpoint_agent
    from manabot.sim.rollout import (
        BatchedSampler,
        RandomBatchController,
        run_vector_games,
    )

    hero_agent, _ = load_checkpoint_agent(hero_path)
    hero = BatchedSampler(hero_agent, deterministic=False, seed=seed, device=device)
    if villain_path is None:
        villain: Any = RandomBatchController(seed=seed + 1)
    else:
        villain_agent, _ = load_checkpoint_agent(villain_path)
        villain = BatchedSampler(
            villain_agent, deterministic=False, seed=seed + 1, device=device
        )
    records, stats = run_vector_games(
        hero, villain, num_games=num_games, num_streams=num_streams, seed=seed
    )
    metrics = aggregate_records(records)
    return {"metrics": metrics, "wall_seconds": stats["wall_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--name", type=str, required=True, help="e.g. r0 / r1")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--ladder-games", type=int, default=200)
    parser.add_argument("--rungs", type=str, default="8,16,32,64")
    parser.add_argument("--head-to-head", type=str, default=None)
    parser.add_argument("--head-to-head-name", type=str, default=None)
    parser.add_argument("--profile-games", type=int, default=400)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out", type=str, default="experiments/data/exp-07-expert-iteration.json"
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
    section = results.setdefault(args.name, {})

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2))

    def show(tag: str, metrics: dict[str, float], wall: float) -> None:
        print(
            f"       {tag}: win {metrics['win_rate']:.3f} "
            f"[{metrics['win_ci_lower']:.3f},{metrics['win_ci_upper']:.3f}] "
            f"play {metrics['win_rate_on_play']:.3f} "
            f"draw {metrics['win_rate_on_draw']:.3f} | wall {wall:.0f}s",
            flush=True,
        )

    # 1. vs random (batched driver).
    if "vs_random" not in section:
        print(f"[run ] {args.name} vs random: {args.games} games", flush=True)
        result = batched_matchup(
            args.student,
            None,
            num_games=args.games,
            seed=args.seed,
            device=args.device,
        )
        section["vs_random"] = result
        save()
        show("vs random", result["metrics"], result["wall_seconds"])

    # 2. Ladder rungs (search villains need engine-side playouts; process pool).
    from run_flat_mc import run_matchup

    student_spec = {
        "kind": "checkpoint",
        "path": args.student,
        "name": args.name,
        "deterministic": False,
    }
    ladder = section.setdefault("ladder", {})
    for index, rung in enumerate(int(n) for n in args.rungs.split(",")):
        key = f"search-{rung}"
        if key in ladder:
            continue
        print(f"[run ] {args.name} vs {key}: {args.ladder_games} games", flush=True)
        result = run_matchup(
            student_spec,
            {"kind": "search", "sims": rung},
            num_games=args.ladder_games,
            workers=args.workers,
            base_seed=args.seed + 101 * (index + 1),
        )
        ladder[key] = result
        save()
        show(f"vs {key}", result["metrics"], result["wall_seconds"])

    # 3. Behavioral profile.
    if not args.skip_profile and "profile" not in section:
        from run_distill_judge import behavior_profile

        print(f"[run ] {args.name} profile: {args.profile_games} games", flush=True)
        section["profile"] = behavior_profile(
            args.student, num_games=args.profile_games, seed=args.seed + 31337
        )
        save()
        p = section["profile"]
        print(
            f"       cast_when_able {p['cast_when_able']:.3f} "
            f"passed_when_able {p['passed_when_able']:.3f} "
            f"win {p['win_rate']:.3f}",
            flush=True,
        )

    # 4. Head-to-head vs another student (both argmax).
    if args.head_to_head:
        key = f"vs_{args.head_to_head_name or Path(args.head_to_head).stem}"
        if key not in section:
            print(f"[run ] {args.name} {key}: {args.games} games", flush=True)
            result = batched_matchup(
                args.student,
                args.head_to_head,
                num_games=args.games,
                seed=args.seed + 5150,
                device=args.device,
            )
            section[key] = result
            save()
            show(key, result["metrics"], result["wall_seconds"])

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
