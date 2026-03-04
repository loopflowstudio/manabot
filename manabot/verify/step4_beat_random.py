"""Verification step 4: stretch goal against random opponent."""

import argparse

from manabot.env import Match, ObservationSpace, Reward
from manabot.model.train import run_training

from .util import build_hypers, print_result, run_evaluation, suppress_truncation_logs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    standard_deck = {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    }

    hypers = build_hypers(
        experiment={
            "seed": args.seed,
            "exp_name": "verify-step4-random",
            "wandb": False,
            "device": "cpu",
        },
        train={
            "num_envs": args.num_envs,
            "total_timesteps": args.total_timesteps,
            "opponent_policy": "random",
        },
        match={
            "hero_deck": standard_deck,
            "villain_deck": standard_deck,
        },
        agent={"attention_on": False},
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
        opponent_policy="random",
        deterministic=True,
        seed=args.seed + 1000,
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
