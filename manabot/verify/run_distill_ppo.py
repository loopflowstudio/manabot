"""Exp-03 (wave/search C4) Task 3: matched-cost PPO baseline.

Runs the first_light_shaped_v1 recipe on INTERACTIVE_DECK (the exp-01 deck
rebind) with total_timesteps sized by the caller so PPO's training wall-clock
cost matches the BC pipeline's total cost. Training wall-clock is measured
around the training chunks only (harness evals are measurement, not training,
per the exp-00 accounting convention); in-loop periodic eval is disabled so
every training second is learning.

Usage:
    python -m manabot.verify.run_distill_ppo --total-timesteps 950272 \
        --seed 1 --label exp03-ppo-matched --log .runs/exp03/ppo_log.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", type=str, default="exp03-ppo-matched")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--eval-num-games", type=int, default=50)
    parser.add_argument("--db", type=str, default=".runs/verify.sqlite")
    parser.add_argument("--log", type=str, default=".runs/exp03/ppo_log.json")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    import manabot.verify.first_light as fl
    from manabot.verify.util import INTERACTIVE_DECK

    # exp-01's deck rebind: the recipe is unchanged, the deck is interactive.
    fl.STANDARD_DECK = INTERACTIVE_DECK

    # Bill training only: measure wall-clock around the training chunks and
    # disable the trainer's in-loop periodic eval.
    train_wall = {"seconds": 0.0, "steps": 0}
    original_build = fl.build_first_light_hypers
    original_chunk = fl._train_chunk

    def build_with_eval_disabled(config, *, total_timesteps, seed, exp_name_suffix):
        hypers = original_build(
            config,
            total_timesteps=total_timesteps,
            seed=seed,
            exp_name_suffix=exp_name_suffix,
        )
        hypers.train.eval_interval = 10**9
        return hypers

    def timed_chunk(agent, hypers):
        start = time.perf_counter()
        result = original_chunk(agent, hypers)
        train_wall["seconds"] += time.perf_counter() - start
        train_wall["steps"] += int(hypers.train.total_timesteps)
        return result

    fl.build_first_light_hypers = build_with_eval_disabled
    fl._train_chunk = timed_chunk

    config = fl.resolve_run_config(
        db=args.db,
        seed=args.seed,
        mode="dev",
        label=args.label,
        opponent_policy="random",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        eval_interval=args.total_timesteps,  # no intermediate harness evals
        eval_num_games=args.eval_num_games,
        baseline=False,
        report=False,
        report_path=None,
        notes="exp-03 matched-cost PPO baseline (C4)",
    )

    harness_start = time.perf_counter()
    summary = fl.run_first_light(config)
    harness_seconds = time.perf_counter() - harness_start

    sps = train_wall["steps"] / train_wall["seconds"] if train_wall["seconds"] else 0.0
    final = summary.get("final") or {}
    log = {
        "run_id": summary["run_id"],
        "label": args.label,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "total_timesteps": args.total_timesteps,
        "train_wall_seconds": train_wall["seconds"],
        "train_sps": sps,
        "harness_wall_seconds": harness_seconds,
        "checkpoint": str(
            Path(".runs") / f"first-light-{args.label}-final" / f"step_{args.total_timesteps}.pt"
        ),
        "final_eval": {
            key: final.get(key)
            for key in (
                "win_rate",
                "cast_when_able",
                "passed_when_able",
                "landed_when_able",
            )
        },
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(
        json.dumps(
            {
                "run_id": log["run_id"],
                "train_wall_seconds": round(train_wall["seconds"], 1),
                "train_sps": round(sps, 1),
                "checkpoint": log["checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
