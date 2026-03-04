"""Verification step 1: PPO optimizer sanity with trivial +1 reward."""

import argparse

import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.model.train import run_training

from .util import build_hypers, print_result, suppress_truncation_logs


def _estimate_value(agent, obs_space, match, reward, seed: int) -> float:
    env = Env(match, obs_space, reward, seed=seed, auto_reset=False)
    try:
        obs, _ = env.reset(seed=seed)
    finally:
        env.close()

    device = next(agent.parameters()).device
    tensor_obs = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }
    with torch.no_grad():
        value = agent.get_value(tensor_obs).mean().item()
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=500)
    args = parser.parse_args(argv)
    suppress_truncation_logs()

    total_timesteps = args.num_envs * args.num_steps * args.updates
    hypers = build_hypers(
        experiment={
            "seed": args.seed,
            "exp_name": "verify-step1-trivial-reward",
        },
        reward={"trivial": True},
        train={
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "total_timesteps": total_timesteps,
        },
    )

    trainer = run_training(hypers)

    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)

    value_target = 1.0 / (1.0 - hypers.train.gamma)
    value_estimate = _estimate_value(trainer.agent, obs_space, match, reward, args.seed)

    metrics = {
        "explained_variance": trainer.last_explained_variance,
        "value_target": value_target,
        "value_estimate": value_estimate,
        "total_timesteps": float(total_timesteps),
    }
    passed = trainer.last_explained_variance > 0.8
    print_result("step1_trivial_reward", passed, metrics)


if __name__ == "__main__":
    main()
