#!/usr/bin/env python3
"""
bench_ab.py
A/B comparison of env configurations with statistical rigor.

Runs two configurations back-to-back for multiple rounds, reports
mean +/- stddev SPS and a t-test p-value.

Usage:
    python scripts/bench_ab.py
    python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5
    python scripts/bench_ab.py --a rust --b rust-no-encode --steps 4096
"""

import argparse
import time

import math

import numpy as np
import torch

from manabot.env import Match, ObservationSpace, Reward, RustVectorEnv, VectorEnv
from manabot.env.single_agent_env import build_opponent_policy
from manabot.infra.hypers import RewardHypers


def build_env(mode: str, num_envs: int, match: Match, obs_space: ObservationSpace, reward: Reward, seed: int):
    if mode == "rust":
        return RustVectorEnv(
            num_envs, match, obs_space, reward,
            device="cpu", seed=seed, opponent_policy="passive",
        )
    elif mode == "async":
        return VectorEnv(
            num_envs, match, obs_space, reward,
            device="cpu", seed=seed,
            opponent_policy=build_opponent_policy("passive"),
        )
    elif mode == "rust-no-encode":
        return RustVectorEnv(
            num_envs, match, obs_space, reward,
            device="cpu", seed=seed, opponent_policy="passive",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def measure_sps(env, num_envs: int, total_steps: int, raw_step: bool = False) -> float:
    """Run total_steps iterations and return SPS."""
    actions = torch.zeros(num_envs, dtype=torch.int64)
    env.reset()

    # Warm up
    for _ in range(10):
        env.step(actions)

    start = time.perf_counter()
    if raw_step and isinstance(env, RustVectorEnv):
        for _ in range(total_steps):
            env._rust_env.step(actions.cpu().tolist())
    else:
        for _ in range(total_steps):
            env.step(actions)
    elapsed = time.perf_counter() - start

    return (total_steps * num_envs) / elapsed


def run_ab(mode_a: str, mode_b: str, num_envs: int, total_steps: int, rounds: int, seed: int):
    match = Match()
    obs_space = ObservationSpace()
    reward = Reward(RewardHypers())

    sps_a = []
    sps_b = []

    for r in range(rounds):
        round_seed = seed + r * 1000

        # Alternate order to reduce systematic bias
        if r % 2 == 0:
            first, second = mode_a, mode_b
            first_list, second_list = sps_a, sps_b
        else:
            first, second = mode_b, mode_a
            first_list, second_list = sps_b, sps_a

        env1 = build_env(first, num_envs, match, obs_space, reward, round_seed)
        s1 = measure_sps(env1, num_envs, total_steps, raw_step=(first == "rust-no-encode"))
        env1.close()
        first_list.append(s1)

        env2 = build_env(second, num_envs, match, obs_space, reward, round_seed)
        s2 = measure_sps(env2, num_envs, total_steps, raw_step=(second == "rust-no-encode"))
        env2.close()
        second_list.append(s2)

        print(f"  Round {r+1}/{rounds}: {mode_a}={sps_a[-1]:.0f}  {mode_b}={sps_b[-1]:.0f}")

    a_arr = np.array(sps_a)
    b_arr = np.array(sps_b)

    a_mean, a_std = np.mean(a_arr), np.std(a_arr, ddof=1)
    b_mean, b_std = np.mean(b_arr), np.std(b_arr, ddof=1)

    # Welch's t-test (manual, no scipy dependency)
    if rounds >= 3:
        n_a, n_b = len(a_arr), len(b_arr)
        var_a, var_b = np.var(a_arr, ddof=1), np.var(b_arr, ddof=1)
        se = math.sqrt(var_a / n_a + var_b / n_b)
        t_stat = (a_mean - b_mean) / se if se > 0 else float("inf")
        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1
        # Two-tailed p-value approximation using normal for simplicity
        # (good enough for df > 5, conservative otherwise)
        z = abs(t_stat)
        p_value = 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) if z < 10 else 0.0
    else:
        t_stat, p_value = float("nan"), float("nan")

    delta_pct = (a_mean - b_mean) / b_mean * 100 if b_mean > 0 else float("nan")

    print(f"\n{'=' * 60}")
    print(f"A/B Results: {num_envs} envs, {total_steps} steps/round, {rounds} rounds")
    print(f"{'=' * 60}")
    print(f"  A ({mode_a:>15s}): {a_mean:>8.0f} +/- {a_std:>6.0f} SPS")
    print(f"  B ({mode_b:>15s}): {b_mean:>8.0f} +/- {b_std:>6.0f} SPS")
    print(f"  Delta: {delta_pct:+.1f}% (A vs B)")
    if rounds >= 3:
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"  p-value: {p_value:.4f} ({sig})")
    else:
        print(f"  p-value: need >= 3 rounds for t-test")
    print()

    return {
        "a_mode": mode_a, "b_mode": mode_b,
        "a_mean": a_mean, "a_std": a_std,
        "b_mean": b_mean, "b_std": b_std,
        "delta_pct": delta_pct, "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="rust", choices=["rust", "async", "rust-no-encode"])
    parser.add_argument("--b", default="async", choices=["rust", "async", "rust-no-encode"])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ab(args.a, args.b, args.num_envs, args.steps, args.rounds, args.seed)


if __name__ == "__main__":
    main()
