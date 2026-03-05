#!/usr/bin/env python3
"""
bench_sps.py
Compare env step throughput between RustVectorEnv and legacy AsyncVectorEnv.
Measures raw reset+step cycles, no training.
"""

import time

import torch

from manabot.env import Match, ObservationSpace, Reward, RustVectorEnv, VectorEnv
from manabot.env.single_agent_env import build_opponent_policy
from manabot.infra.hypers import RewardHypers

NUM_ENVS = 8
TOTAL_STEPS = 2048
SEED = 42


def bench(label: str, env):
    obs, _ = env.reset()
    actions = torch.zeros(env.num_envs, dtype=torch.int64)

    # Warm up
    for _ in range(10):
        env.step(actions)

    start = time.perf_counter()
    for _ in range(TOTAL_STEPS):
        env.step(actions)
    elapsed = time.perf_counter() - start

    total = TOTAL_STEPS * env.num_envs
    sps = total / elapsed
    print(f"{label:>20s}: {sps:8.0f} SPS  ({total} steps in {elapsed:.2f}s)")
    env.close()
    return sps


def main():
    match = Match()
    obs_space = ObservationSpace()
    reward = Reward(RewardHypers())

    rust_env = RustVectorEnv(
        num_envs=NUM_ENVS,
        match=match,
        observation_space=obs_space,
        reward=reward,
        device="cpu",
        seed=SEED,
        opponent_policy="passive",
    )
    rust_sps = bench("RustVectorEnv", rust_env)

    legacy_env = VectorEnv(
        num_envs=NUM_ENVS,
        match=match,
        observation_space=obs_space,
        reward=reward,
        device="cpu",
        seed=SEED,
        opponent_policy=build_opponent_policy("passive"),
    )
    legacy_sps = bench("AsyncVectorEnv", legacy_env)

    print()
    if legacy_sps > 0:
        print(f"Speedup: {rust_sps / legacy_sps:.2f}x")


if __name__ == "__main__":
    main()
