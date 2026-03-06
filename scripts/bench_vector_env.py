#!/usr/bin/env python3
"""
bench_vector_env.py
Measure vector-env throughput for the active Rust path and the legacy baseline.
"""

import argparse
import statistics
import time
from typing import Any

import torch

from manabot.env import (
    LegacyVectorEnv,
    Match,
    ObservationSpace,
    Reward,
    RustVectorEnv,
    build_opponent_policy,
)
from manabot.infra.hypers import RewardHypers

VALID_BACKENDS = ("rust", "legacy")


def build_env(
    backend: str, num_envs: int, seed: int
) -> RustVectorEnv | LegacyVectorEnv:
    match = Match()
    observation_space = ObservationSpace()
    reward = Reward(RewardHypers())

    if backend == "rust":
        return RustVectorEnv(
            num_envs=num_envs,
            match=match,
            observation_space=observation_space,
            reward=reward,
            device="cpu",
            seed=seed,
            opponent_policy="passive",
        )

    return LegacyVectorEnv(
        num_envs=num_envs,
        match=match,
        observation_space=observation_space,
        reward=reward,
        device="cpu",
        seed=seed,
        opponent_policy=build_opponent_policy("passive"),
    )


def measure_sps(env: Any, total_steps: int) -> float:
    actions = torch.zeros(env.num_envs, dtype=torch.int64)
    env.reset()

    for _ in range(10):
        env.step(actions)

    start = time.perf_counter()
    for _ in range(total_steps):
        env.step(actions)
    elapsed = time.perf_counter() - start

    return (total_steps * env.num_envs) / elapsed


def run_backend(
    backend: str, num_envs: int, total_steps: int, rounds: int, seed: int
) -> None:
    samples = []
    for round_index in range(rounds):
        round_seed = seed + round_index * 1000
        env = build_env(backend, num_envs, round_seed)
        try:
            sps = measure_sps(env, total_steps)
        finally:
            env.close()
        samples.append(sps)
        print(f"{backend:>6s} round {round_index + 1}/{rounds}: {sps:9.0f} SPS")

    mean_sps = statistics.fmean(samples)
    std_sps = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(
        f"{backend:>6s} summary: num_envs={num_envs} steps={total_steps} "
        f"rounds={rounds} mean={mean_sps:.0f} std={std_sps:.0f} SPS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=VALID_BACKENDS,
        dest="backends",
        help="Backend to benchmark; repeat to run multiple backends",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    backends = args.backends or ["rust"]
    for backend in backends:
        run_backend(backend, args.num_envs, args.steps, args.rounds, args.seed)


if __name__ == "__main__":
    main()
