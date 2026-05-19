"""Verification step 2: deterministic memorization against passive opponent."""

import argparse

from manabot.env import Match, ObservationSpace, Reward
from manabot.model.train import run_training

from .util import (
    build_hypers,
    print_result,
    run_evaluation,
    suppress_truncation_logs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-games", type=int, default=200)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    hypers = build_hypers(
        experiment={
            "seed": args.seed,
            "exp_name": "verify-step2-memorization",
        },
        train={
            "num_envs": 1,
            "total_timesteps": args.total_timesteps,
        },
        # Default deck (mixed creatures + lands) gives the agent real
        # decisions: play lands, cast elves/ogres, attack.
    )

    trainer = run_training(hypers)

    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    eval_metrics = run_evaluation(
        trainer.agent,
        obs_space,
        match,
        reward,
        num_games=args.num_games,
        opponent_policy="passive",
        deterministic=False,
        seed=args.seed + 1000,
    )

    metrics = {
        **eval_metrics,
        "explained_variance": trainer.last_explained_variance,
        "total_timesteps": float(args.total_timesteps),
    }
    passed = eval_metrics["win_ci_lower"] > 0.95
    print_result("step2_memorization", passed, metrics)


if __name__ == "__main__":
    main()
