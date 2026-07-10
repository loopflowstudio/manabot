"""Exp-11 (wave/search C8): PPO arms with seat-routed opponents.

Trains one arm of the curriculum/exploitability experiment with the proven
terminal-only recipe (RewardHypers defaults: win +1 / lose -1, no shaping)
on the INTERACTIVE_DECK mirror, dev-preset scale (4 envs x 128 steps x
262,144 timesteps), through the seat-routed collector — the *only*
difference between arms is who plays the opponent seat:

    --arm random    uniform-over-valid opponent (lineage control, seat-balanced)
    --arm student   frozen checkpoint plays the opponent seat (--student)
    --arm self      the live learner plays both seats (true self-play;
                    mirror-seat transitions discarded to keep batch identical)

Usage:
    python -m manabot.verify.run_exp11_train --arm student --seed 1 \
        --student .runs/exp11/student_r0_ported.pt \
        --label exp11-student-s1 --log .runs/exp11/train_student_s1.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("random", "student", "self"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--student", type=str, default=None,
                        help="frozen opponent checkpoint (required for --arm student)")
    parser.add_argument("--total-timesteps", type=int, default=262_144)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=128,
                        help="periodic eval every N updates (0 disables)")
    parser.add_argument("--eval-num-games", type=int, default=50)
    parser.add_argument("--log", type=str, required=True)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    from manabot.env import Match, ObservationSpace, Reward
    from manabot.infra import Experiment
    from manabot.model.agent import Agent
    from manabot.sim.flat_mc import load_checkpoint_agent
    from manabot.sim.net_opponent import NetOpponentTrainer, SeatRoutedCollector
    from manabot.verify.util import INTERACTIVE_DECK, build_hypers

    hypers = build_hypers(
        experiment={"seed": args.seed, "exp_name": args.label},
        train={
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "total_timesteps": args.total_timesteps,
            # Scripted policy used only by the periodic eval, all arms alike.
            "opponent_policy": "random",
            "eval_interval": args.eval_interval if args.eval_interval > 0 else 10**9,
            "eval_num_games": args.eval_num_games,
        },
        # Terminal-only reward: RewardHypers defaults (win +1 / lose -1,
        # zero shaping, potential off) — the exp-04 proven recipe.
        match={"hero_deck": INTERACTIVE_DECK, "villain_deck": INTERACTIVE_DECK},
    )

    observation_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    experiment = Experiment(hypers.experiment, hypers)

    opponent_agent = None
    if args.arm == "student":
        if not args.student:
            parser.error("--arm student requires --student")
        opponent_agent, _ = load_checkpoint_agent(args.student)
    opponent_mode = {"random": "random", "student": "frozen", "self": "self"}[args.arm]

    collector = SeatRoutedCollector(
        observation_space,
        match,
        reward,
        num_envs=args.num_envs,
        seed=args.seed,
        opponent_mode=opponent_mode,
        opponent_agent=opponent_agent,
        device=experiment.device,
    )
    agent = Agent(observation_space, hypers.agent)
    trainer = NetOpponentTrainer(agent, experiment, collector, hypers.train)

    start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start

    stats = collector.stats.to_dict()
    log = {
        "arm": args.arm,
        "label": args.label,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "total_timesteps": args.total_timesteps,
        "student": args.student,
        "wall_seconds": wall_seconds,
        "train_sps": args.total_timesteps / wall_seconds if wall_seconds else 0.0,
        "checkpoint": str(experiment.runs_dir / f"step_{trainer.global_step}.pt"),
        "collector": stats,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(json.dumps(
        {
            "arm": args.arm,
            "seed": args.seed,
            "wall_seconds": round(wall_seconds, 1),
            "rollout_win_rate": round(stats["learner_win_rate"], 4),
            "games": stats["games"],
            "checkpoint": log["checkpoint"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
