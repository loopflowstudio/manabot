#!/usr/bin/env python3
"""
bench_breakdown.py
Profile where time goes in the env step + encode pipeline.

Decomposes the RustVectorEnv.step() into its constituent parts and
times each one separately using the existing Profiler.

Usage:
    python scripts/bench_breakdown.py
    python scripts/bench_breakdown.py --num-envs 16 --steps 2048
    python scripts/bench_breakdown.py --with-inference
"""

import argparse

import torch

from manabot.env import Match, ObservationSpace, Reward, RustVectorEnv
from manabot.env.env import add_truncation_flags, stack_encoded_observations
from manabot.infra.hypers import RewardHypers
from manabot.infra.profiler import Profiler


def run_breakdown(num_envs: int, total_steps: int, with_inference: bool, seed: int):
    profiler = Profiler(enabled=True)
    match = Match()
    obs_space = ObservationSpace()
    reward = Reward(RewardHypers())

    env = RustVectorEnv(
        num_envs, match, obs_space, reward,
        device="cpu", seed=seed, opponent_policy="passive",
    )

    agent = None
    if with_inference:
        from manabot.infra.hypers import AgentHypers
        from manabot.model import Agent
        agent = Agent(obs_space, AgentHypers())
        agent.eval()

    obs, _ = env.reset()
    actions = torch.zeros(num_envs, dtype=torch.int64)

    # Warm up
    for _ in range(20):
        env.step(actions)

    with profiler.track("update"):
        for _ in range(total_steps):
            with profiler.track("step"):
                # 1. Torch inference (if enabled)
                if agent is not None:
                    with profiler.track("inference"):
                        with torch.no_grad():
                            action, _, _, _ = agent.get_action_and_value(obs)
                        actions = action

                # 2. Rust env step (game simulation + PyO3 boundary)
                with profiler.track("rust_step"):
                    # Clamp actions to valid range (agent may pick out-of-bounds)
                    action_list = [min(a, 0) for a in actions.cpu().tolist()]
                    results = env._rust_env.step(action_list)

                # 3. Result unpacking (reward compute, truncation flags)
                with profiler.track("unpack"):
                    raw_obs = []
                    for env_index, (o, raw_reward, done, cut, info) in enumerate(results):
                        info_dict = dict(info)
                        env.reward.compute(raw_reward, env._last_raw_obs[env_index], o)
                        add_truncation_flags(o, info_dict, obs_space.encoder)
                        raw_obs.append(o)
                    env._last_raw_obs = raw_obs

                # 4. Python observation encoding
                with profiler.track("encode"):
                    encoded = [obs_space.encode(o) for o in raw_obs]

                # 5. Stack + tensorize
                with profiler.track("tensorize"):
                    obs = stack_encoded_observations(encoded, env.device)

    stats = profiler.get_stats()
    wall_time = stats.get("update", {}).get("total_time", 0)
    total_env_steps = total_steps * num_envs
    sps = total_env_steps / wall_time if wall_time > 0 else 0

    print(f"\n{'=' * 72}")
    print(f"Breakdown: num_envs={num_envs}, steps={total_steps}, "
          f"inference={'on' if with_inference else 'off'}")
    print(f"Total: {total_env_steps} env steps in {wall_time:.2f}s = {sps:.0f} SPS")
    print(f"{'=' * 72}")
    print(f"\n{'Path':<40s} {'Total':>8s} {'%':>6s} {'Count':>7s} "
          f"{'Mean':>10s} {'p95':>10s}")
    print(f"{'-'*40} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*10}")

    for path in sorted(stats.keys()):
        s = stats[path]
        total_s = s["total_time"]
        pct = s["pct_of_total"]
        count = int(s["count"])
        mean_us = s["mean"] * 1e6
        p95_us = s["p95"] * 1e6
        depth = path.count("/")
        indent = "  " * depth
        label = path.split("/")[-1]
        print(f"{indent}{label:<{40 - 2*depth}s} {total_s:>7.3f}s {pct:>5.1f}% "
              f"{count:>7d} {mean_us:>8.1f}us {p95_us:>8.1f}us")

    env.close()
    return stats, sps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--with-inference", action="store_true",
                        help="Include torch forward pass")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_breakdown(args.num_envs, args.steps, args.with_inference, args.seed)


if __name__ == "__main__":
    main()
