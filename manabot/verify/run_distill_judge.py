"""Exp-03 (wave/search C4) Task 4: judge the BC policy vs matched-cost PPO.

Matchup matrix (seat-balanced, Wilson CIs, per-seat rates; resumable JSON):
    1. bc vs random
    2. ppo-matched vs random
    3. bc vs each C1v2 checkpoint
    4. bc vs search-{16,64}  (ladder placement)
Plus behavioral profiles (cast_when_able / passed_when_able) for the BC and
PPO policies via the same capture_evaluation instrument used in exp-01.

Usage:
    python -m manabot.verify.run_distill_judge \
        --bc .runs/exp03/bc_policy.pt --ppo .runs/<ppo>/step_N.pt \
        --games 400 --workers 8 --out reports/data/exp-03-distillation.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

C1V2_CHECKPOINT_TEMPLATE = (
    "/Users/jack/src/manabot/.runs/first-light-c1-interactive-dev2-s{seed}-final/"
    "step_65536.pt"
)


def build_matchups(
    bc_path: str,
    ppo_path: str | None,
    games: int,
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    bc = {"kind": "checkpoint", "path": bc_path, "name": "bc-search64"}
    matchups: list[tuple[dict[str, Any], dict[str, Any], int]] = [
        (bc, {"kind": "random"}, games),
    ]
    if ppo_path:
        ppo = {"kind": "checkpoint", "path": ppo_path, "name": "ppo-matched"}
        matchups.append((ppo, {"kind": "random"}, games))
        matchups.append((bc, ppo, games))
    for seed in (1, 2, 3):
        matchups.append(
            (
                bc,
                {
                    "kind": "checkpoint",
                    "path": C1V2_CHECKPOINT_TEMPLATE.format(seed=seed),
                    "name": f"c1v2-s{seed}",
                },
                games,
            )
        )
    for n in (16, 64):
        matchups.append((bc, {"kind": "search", "sims": n}, games))
    return matchups


def behavior_profile(
    checkpoint_path: str,
    *,
    num_games: int,
    seed: int,
) -> dict[str, float]:
    """cast_when_able / passed_when_able etc. vs random, seat-balanced."""

    from manabot.env import Match, Reward
    from manabot.infra.hypers import MatchHypers, RewardHypers
    from manabot.sim.flat_mc import load_checkpoint_agent
    from manabot.verify.util import INTERACTIVE_DECK, run_evaluation

    agent, obs_space = load_checkpoint_agent(checkpoint_path)
    match = Match(
        MatchHypers(
            hero="hero",
            villain="villain",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    metrics = run_evaluation(
        agent,
        obs_space,
        match,
        Reward(RewardHypers()),
        num_games=num_games,
        opponent_policy="random",
        deterministic=False,
        seed=seed,
        seat_balanced=True,
    )
    keys = (
        "num_games",
        "win_rate",
        "win_rate_on_play",
        "win_rate_on_draw",
        "cast_when_able",
        "could_spell",
        "spell_casts",
        "passed_when_able",
        "could_pass",
        "pass_count",
        "attacked_when_able",
        "landed_when_able",
        "mean_steps",
    )
    return {key: float(metrics[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=str, required=True)
    parser.add_argument("--ppo", type=str, default=None)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--profile-games", type=int, default=400)
    parser.add_argument(
        "--out", type=str, default="reports/data/exp-03-distillation.json"
    )
    parser.add_argument("--skip-profiles", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    from manabot.sim.flat_mc import spec_name
    from manabot.verify.run_flat_mc import run_matchup

    matchups = build_matchups(args.bc, args.ppo, args.games)
    for index, (hero_spec, villain_spec, games) in enumerate(matchups):
        name = f"{spec_name(hero_spec)}__{spec_name(villain_spec)}"
        if name in results.get("matchups", {}):
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name}: {games} games, {args.workers} workers", flush=True)
        result = run_matchup(
            hero_spec,
            villain_spec,
            num_games=games,
            workers=args.workers,
            base_seed=args.seed + index * 97,
        )
        results.setdefault("matchups", {})[name] = result
        out_path.write_text(json.dumps(results, indent=2))
        m = result["metrics"]
        print(
            f"       win {m['win_rate']:.3f} "
            f"[{m['win_ci_lower']:.3f},{m['win_ci_upper']:.3f}] "
            f"play {m['win_rate_on_play']:.3f} draw {m['win_rate_on_draw']:.3f} "
            f"| wall {result['wall_seconds']:.0f}s",
            flush=True,
        )

    if not args.skip_profiles:
        profiles = results.setdefault("profiles", {})
        targets = {"bc-search64": args.bc}
        if args.ppo:
            targets["ppo-matched"] = args.ppo
        for name, path in targets.items():
            if name in profiles:
                print(f"[skip] profile {name}")
                continue
            print(f"[run ] profile {name}: {args.profile_games} games", flush=True)
            profiles[name] = behavior_profile(
                path, num_games=args.profile_games, seed=args.seed + 31337
            )
            out_path.write_text(json.dumps(results, indent=2))
            p = profiles[name]
            print(
                f"       cast_when_able {p['cast_when_able']:.3f} "
                f"passed_when_able {p['passed_when_able']:.3f} "
                f"win {p['win_rate']:.3f}",
                flush=True,
            )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
