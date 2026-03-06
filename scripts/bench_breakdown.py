#!/usr/bin/env python3
"""
bench_breakdown.py
Profile the active Rust vector-env path, with or without model inference.
"""

import argparse

import torch

from manabot.env import Match, ObservationSpace, Reward, VectorEnv
from manabot.infra.hypers import AgentHypers, RewardHypers
from manabot.infra.profiler import Profiler
from manabot.model import Agent


def build_env(num_envs: int, seed: int) -> VectorEnv:
    return VectorEnv(
        num_envs=num_envs,
        match=Match(),
        observation_space=ObservationSpace(),
        reward=Reward(RewardHypers()),
        device="cpu",
        seed=seed,
        opponent_policy="passive",
    )


def run_breakdown(
    num_envs: int,
    total_steps: int,
    with_inference: bool,
    seed: int,
) -> tuple[dict[str, dict[str, float]], float]:
    profiler = Profiler(enabled=True)
    env = build_env(num_envs, seed)

    agent = None
    if with_inference:
        agent = Agent(env.observation_space, AgentHypers())
        agent.eval()

    obs, _ = env.reset()
    actions = torch.zeros(num_envs, dtype=torch.int64)

    for _ in range(20):
        env.step(actions)

    with profiler.track("update"):
        for _ in range(total_steps):
            with profiler.track("step"):
                if agent is not None:
                    with profiler.track("inference"):
                        with torch.no_grad():
                            action, _, _, _ = agent.get_action_and_value(obs)
                        actions = action

                with profiler.track("step_into_buffers"):
                    env._rust_env.step_into_buffers(actions.cpu().tolist())

                with profiler.track("sync_tensors"):
                    env._sync_tensors_from_buffers()

                with profiler.track("apply_reward_policy"):
                    env._apply_reward_policy()

                obs = env._obs_tensors

    env.close()
    stats = profiler.get_stats()
    wall_time = stats.get("update", {}).get("total_time", 0.0)
    sps = (total_steps * num_envs) / wall_time if wall_time > 0 else 0.0
    return stats, sps


def print_report(
    stats: dict[str, dict[str, float]],
    sps: float,
    num_envs: int,
    total_steps: int,
    with_inference: bool,
) -> None:
    wall_time = stats.get("update", {}).get("total_time", 0.0)
    total_env_steps = total_steps * num_envs

    print(f"\n{'=' * 72}")
    print(
        f"VectorEnv breakdown: num_envs={num_envs}, steps={total_steps}, "
        f"inference={'on' if with_inference else 'off'}"
    )
    print(f"Total: {total_env_steps} env steps in {wall_time:.2f}s = {sps:.0f} SPS")
    print(f"{'=' * 72}")
    print(
        f"\n{'Path':<32s} {'Total':>8s} {'%':>6s} {'Count':>7s} "
        f"{'Mean':>10s} {'p95':>10s}"
    )
    print(f"{'-' * 32} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 10} {'-' * 10}")

    for path in sorted(stats.keys()):
        stat = stats[path]
        total_s = stat["total_time"]
        pct = stat["pct_of_total"]
        count = int(stat["count"])
        mean_us = stat["mean"] * 1e6
        p95_us = stat["p95"] * 1e6
        depth = path.count("/")
        indent = "  " * depth
        label = path.split("/")[-1]
        print(
            f"{indent}{label:<{32 - 2 * depth}s} {total_s:>7.3f}s {pct:>5.1f}% "
            f"{count:>7d} {mean_us:>8.1f}us {p95_us:>8.1f}us"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-inference",
        action="store_true",
        help="Include a torch forward pass per step",
    )
    args = parser.parse_args()

    stats, sps = run_breakdown(
        num_envs=args.num_envs,
        total_steps=args.steps,
        with_inference=args.with_inference,
        seed=args.seed,
    )
    print_report(stats, sps, args.num_envs, args.steps, args.with_inference)


if __name__ == "__main__":
    main()
