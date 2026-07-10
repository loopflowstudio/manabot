#!/usr/bin/env python3
"""
bench_policy_driver.py
Net-in-loop throughput of the batched rollout driver (exp-07, wave goal 1).

Every surfaced decision on K parallel streams is answered by the policy net
(both seats), so obs/sec here is directly comparable to the historical
inference-on numbers from scripts/bench_breakdown.py (2.0k SPS at 16 envs,
reports/sps-closeout.md).

Usage:
    python scripts/bench_policy_driver.py --streams 256 --steps 200 --device mps
"""

import argparse
import time

import numpy as np
import torch

import managym

from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.infra.profiler import Profiler
from manabot.model import Agent
from manabot.sim.rollout import BatchedSampler, _allocate_buffers
from manabot.verify.util import INTERACTIVE_DECK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    obs_space = ObservationSpace()
    agent = Agent(obs_space, AgentHypers())
    agent.eval()
    sampler = BatchedSampler(agent, seed=args.seed, device=args.device)

    from manabot.infra.hypers import MatchHypers

    match = Match(
        MatchHypers(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = managym.VectorEnv(
        num_envs=args.streams, seed=args.seed, skip_trivial=True, opponent_policy="none"
    )
    buffers = _allocate_buffers(obs_space, args.streams)
    env.set_buffers(buffers)
    env.reset_all_into_buffers(match.to_rust())

    profiler = Profiler(enabled=True)
    rows = np.arange(args.streams, dtype=np.int64)

    # Warmup (compiles MPS kernels, fills caches).
    for _ in range(10):
        actions = sampler.select(buffers, rows)
        env.step_into_buffers(actions.tolist())

    start = time.perf_counter()
    with profiler.track("update"):
        for _ in range(args.steps):
            with profiler.track("select"):
                actions = sampler.select(buffers, rows)
            with profiler.track("env_step"):
                env.step_into_buffers(actions.tolist())
    wall = time.perf_counter() - start

    total_obs = args.steps * args.streams
    stats = profiler.get_stats()
    print(
        f"streams={args.streams} device={args.device} steps={args.steps}: "
        f"{total_obs} obs in {wall:.2f}s = {total_obs / wall:.0f} obs/sec (net-in-loop)"
    )
    for path in sorted(stats.keys()):
        s = stats[path]
        print(
            f"  {path:<12s} total {s['total_time']:.3f}s  {s['pct_of_total']:.1f}%  "
            f"mean {s['mean'] * 1e3:.2f}ms"
        )


if __name__ == "__main__":
    main()
