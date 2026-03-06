"""Verification step 3: learn to beat passive opponent with trend checks."""

import argparse

from manabot.env import (
    Match,
    ObservationSpace,
    Reward,
    VectorEnv,
)
from manabot.infra import Experiment
from manabot.model import Agent, Trainer

from .util import (
    STANDARD_DECK,
    build_hypers,
    print_result,
    run_evaluation,
    suppress_truncation_logs,
)


def _train_chunk(agent: Agent | None, hypers):
    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    experiment = Experiment(hypers.experiment, hypers)
    env = VectorEnv(
        hypers.train.num_envs,
        match,
        obs_space,
        reward,
        device=experiment.device,
        opponent_policy=hypers.train.opponent_policy,
    )
    agent = agent or Agent(obs_space, hypers.agent)
    trainer = Trainer(agent, experiment, env, hypers.train)
    trainer.train()
    return agent, trainer, obs_space, match, reward


def _parse_checkpoints(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one checkpoint timestep")
    if values != sorted(values):
        raise ValueError("Checkpoint timesteps must be sorted ascending")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--checkpoints", default="1000000,3000000,5000000")
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    checkpoints = _parse_checkpoints(args.checkpoints)
    train_chunks = [
        checkpoints[0],
        *[b - a for a, b in zip(checkpoints, checkpoints[1:])],
    ]

    metrics: dict[str, float] = {}
    eval_by_ckpt: dict[int, dict[str, float]] = {}
    agent = None
    last_explained_variance = float("nan")

    for index, chunk_timesteps in enumerate(train_chunks):
        global_timestep = checkpoints[index]
        hypers = build_hypers(
            experiment={
                "seed": args.seed,
                "exp_name": f"verify-step3-passive-{global_timestep}",
            },
            train={
                "num_envs": args.num_envs,
                "total_timesteps": chunk_timesteps,
            },
            match={
                "hero_deck": STANDARD_DECK,
                "villain_deck": STANDARD_DECK,
            },
        )

        agent, trainer, obs_space, match, reward = _train_chunk(agent, hypers)
        last_explained_variance = trainer.last_explained_variance
        eval_metrics = run_evaluation(
            agent,
            obs_space,
            match,
            reward,
            num_games=args.num_games,
            opponent_policy="passive",
            deterministic=True,
            seed=args.seed + 1000 + global_timestep,
        )
        eval_by_ckpt[global_timestep] = eval_metrics

        prefix = f"ckpt_{global_timestep}"
        for key in (
            "win_rate",
            "win_ci_lower",
            "mean_steps",
            "attack_rate",
            "action_space_truncations",
            "card_space_truncations",
            "permanent_space_truncations",
        ):
            metrics[f"{prefix}_{key}"] = eval_metrics[key]

    first = eval_by_ckpt[checkpoints[0]]
    final = eval_by_ckpt[checkpoints[-1]]

    mean_steps_improved = final["mean_steps"] <= first["mean_steps"] * 0.9
    if first["attack_rate"] <= 0.0:
        attack_rate_improved = final["attack_rate"] > 0.0
    else:
        attack_rate_improved = final["attack_rate"] >= first["attack_rate"] * 1.1

    metrics["last_explained_variance"] = last_explained_variance
    metrics["mean_steps_improved"] = float(mean_steps_improved)
    metrics["attack_rate_improved"] = float(attack_rate_improved)

    passed = (
        final["win_ci_lower"] > 0.90
        and mean_steps_improved
        and attack_rate_improved
        and last_explained_variance > 0.5
    )
    print_result("step3_beat_passive", passed, metrics)


if __name__ == "__main__":
    main()
