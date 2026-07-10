"""Exp-11 (wave/search C8): cross-judge the curriculum/exploitability arms.

Matrix per checkpoint (all seat-balanced, stochastic sampling, Wilson CIs,
INTERACTIVE_DECK mirror — the exp-00c/01/02/03 protocol):

    1. vs random               — batched vector driver, 400 games;
    2. vs frozen student_r0    — batched head-to-head, 400 games
                                 (for the vs-student arm this IS the
                                 exploitability probe reading);
    3. ladder rungs search-{4,8,16} — process-parallel, 400 games each;
    4. behavioral profile      — cast/passed_when_able etc., 400 games.

Results accumulate in a JSON keyed by checkpoint name; finished cells are
skipped on re-run.

Usage:
    python -m manabot.verify.run_exp11_judge \
        --checkpoint .runs/exp11/random-s1/step_262144.pt --name random-s1 \
        --student .runs/exp11/student_r0_ported.pt \
        --out reports/data/exp-11-curriculum-exploitability.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--student", type=str, required=True,
                        help="ported frozen student_r0 checkpoint")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--ladder-games", type=int, default=400)
    parser.add_argument("--rungs", type=str, default="4,8,16")
    parser.add_argument("--profile-games", type=int, default=400)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out", type=str, default="reports/data/exp-11-curriculum-exploitability.json"
    )
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    from manabot.verify.run_exp07_judge import batched_matchup

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
    section = results.setdefault(args.name, {})
    section["checkpoint"] = args.checkpoint

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

    # 1. vs random.
    if "vs_random" not in section:
        print(f"[run ] {args.name} vs random: {args.games} games", flush=True)
        result = batched_matchup(
            args.checkpoint, None,
            num_games=args.games, seed=args.seed, device=args.device,
        )
        section["vs_random"] = result
        save()
        show("vs random", result["metrics"], result["wall_seconds"])

    # 2. vs the frozen student (exploitability probe for the student arm).
    if "vs_student" not in section:
        print(f"[run ] {args.name} vs student_r0: {args.games} games", flush=True)
        result = batched_matchup(
            args.checkpoint, args.student,
            num_games=args.games, seed=args.seed + 5150, device=args.device,
        )
        section["vs_student"] = result
        save()
        show("vs student_r0", result["metrics"], result["wall_seconds"])

    # 3. Ladder rungs.
    if not args.skip_ladder:
        from manabot.verify.run_flat_mc import run_matchup

        spec = {
            "kind": "checkpoint",
            "path": args.checkpoint,
            "name": args.name,
            "deterministic": False,
        }
        ladder = section.setdefault("ladder", {})
        for index, rung in enumerate(int(n) for n in args.rungs.split(",")):
            key = f"search-{rung}"
            if key in ladder:
                continue
            print(
                f"[run ] {args.name} vs {key}: {args.ladder_games} games", flush=True
            )
            result = run_matchup(
                spec,
                {"kind": "search", "sims": rung},
                num_games=args.ladder_games,
                workers=args.workers,
                base_seed=args.seed + 101 * (index + 1),
            )
            ladder[key] = result
            save()
            show(f"vs {key}", result["metrics"], result["wall_seconds"])

    # 4. Behavioral profile.
    if not args.skip_profile and "profile" not in section:
        from manabot.verify.run_distill_judge import behavior_profile

        print(f"[run ] {args.name} profile: {args.profile_games} games", flush=True)
        section["profile"] = behavior_profile(
            args.checkpoint, num_games=args.profile_games, seed=args.seed + 31337
        )
        save()
        p = section["profile"]
        print(
            f"       cast_when_able {p['cast_when_able']:.3f} "
            f"passed_when_able {p['passed_when_able']:.3f} "
            f"win {p['win_rate']:.3f}",
            flush=True,
        )

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
