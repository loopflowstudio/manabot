#!/usr/bin/env python3
"""
bench_breakdown.py
Profile the active Rust vector-env path, with or without model inference.
"""

import argparse
from contextlib import closing
import json

from bench_observation_support import identity, matched_breakdown
import numpy as np
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace, step_buffers
from manabot.infra.config import (
    POLICY_INPUT_VERSION,
    ModelConfig,
)
from manabot.infra.profiler import Profiler
from manabot.policy.spans import ObservationSelection
import managym


def run_breakdown(
    num_envs: int,
    total_steps: int,
    with_inference: bool,
    seed: int,
) -> tuple[dict[str, dict[str, float]], float]:
    profiler = Profiler(enabled=True)
    space = ObservationSpace()
    buffers = step_buffers(num_envs)
    agent = ActorCritic(space, ModelConfig()).eval() if with_inference else None
    with closing(
        RecordedVectorEnv(
            num_envs=num_envs, seed=seed, player_policies=("agent", "random")
        )
    ) as env:
        env.set_observation_layout(
            buffers, action_overflow=space.config.action_overflow, layout=space.layout
        )
        env.reset_all_into_buffers(Match().to_rust())
        actions = [0] * num_envs

        def step() -> None:
            env.step_into_buffers(actions)
            done = np.flatnonzero(buffers["terminated"] | buffers["truncated"])
            if done.size:
                env.load_slots_into_buffers(
                    [managym.SlotLoad.fresh(int(slot)) for slot in done]
                )

        for _ in range(20):
            step()
        with profiler.track("update"):
            for _ in range(total_steps):
                with profiler.track("step"):
                    if agent is not None:
                        with profiler.track("inference"):
                            rows = ObservationSelection(
                                space, tuple(env.observation_rows())
                            )
                            actions = agent.sample(
                                agent.prepare(rows, inference=True)
                            ).tolist()
                    with profiler.track("step_and_refill"):
                        step()
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Steady-game seed; the matched decision corpus is fixed at seed 23",
    )
    parser.add_argument(
        "--with-inference",
        action="store_true",
        help="Include a torch forward pass per step",
    )
    parser.add_argument(
        "--matched-corpus",
        action="store_true",
        help="Measure identical native priority/target/payment/large-action roots (benchmark feature adds Rust allocation counts)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit a portable timing receipt"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    if args.matched_corpus:
        if args.with_inference:
            parser.error(
                "matched corpus measures environment encoding, without inference"
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        space = ObservationSpace()
        with closing(RecordedVectorEnv(num_envs=args.num_envs, seed=args.seed)) as env:
            env.set_observation_layout(
                step_buffers(args.num_envs),
                action_overflow=space.config.action_overflow,
                layout=space.layout,
            )
            env.reset_all_into_buffers(Match().to_rust())
            before = identity()
            rows = matched_breakdown(env, space, args.num_envs, args.steps)
            print(
                json.dumps(
                    {
                        "identity": before,
                        "identity_after": identity(),
                        "num_envs": args.num_envs,
                        "steps": args.steps,
                        "policy_input_version": POLICY_INPUT_VERSION,
                        "cases": rows,
                    }
                )
            )
        return

    stats, sps = run_breakdown(
        num_envs=args.num_envs,
        total_steps=args.steps,
        with_inference=args.with_inference,
        seed=args.seed,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "num_envs": args.num_envs,
                    "steps": args.steps,
                    "warmup_steps": 20,
                    "policy_input_version": POLICY_INPUT_VERSION,
                    "inference": args.with_inference,
                    "stats": stats,
                    "steps_per_second": sps,
                }
            )
        )
    else:
        print_report(stats, sps, args.num_envs, args.steps, args.with_inference)


if __name__ == "__main__":
    main()
