"""Verification step 4: stretch goal against random opponent."""

import argparse

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.metrics import MetricsDB
from manabot.model.train import run_training

from .util import (
    STANDARD_DECK,
    build_hypers,
    print_result,
    run_evaluation,
    suppress_truncation_logs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    hypers = build_hypers(
        experiment={
            "seed": args.seed,
            "exp_name": "verify-step4-random",
            "wandb": args.wandb,
        },
        train={
            "num_envs": args.num_envs,
            "total_timesteps": args.total_timesteps,
            "opponent_policy": "random",
        },
        match={
            "hero_deck": STANDARD_DECK,
            "villain_deck": STANDARD_DECK,
        },
    )

    trainer = run_training(hypers)

    metrics_db = MetricsDB()
    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    eval_metrics = run_evaluation(
        trainer.agent,
        obs_space,
        match,
        reward,
        num_games=args.num_games,
        opponent_policy="random",
        deterministic=False,
        seed=args.seed + 1000,
        metrics_db=metrics_db,
        model_name="verify-step4-random",
        model_step=args.total_timesteps,
    )

    metrics = {
        **eval_metrics,
        "explained_variance": trainer.last_explained_variance,
        "total_timesteps": float(args.total_timesteps),
    }
    passed = eval_metrics["win_ci_lower"] > 0.60
    print_result("step4_beat_random", passed, metrics)


if __name__ == "__main__":
    main()
