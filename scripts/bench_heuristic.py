"""Bounded CPU collection cost with a native opponent, without model inference.

Run with `uv run python scripts/bench_heuristic.py --opponent heuristic`.
Compare random and heuristic in separate processes on the same host. They play
different trajectories: this measures collection cost, not selector speedup or
strength. Two slots advance 128 times through the training wrapper, including
encoding, tensor views, reward handling and automatic resets. Startup includes
construction and reset, excluding imports. There is no warmup: first-use opponent
knowledge initialization is included in whichever phase first needs it.
"""

import argparse
import json
import platform
import time

import torch

from manabot import managym
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.vector_env import VectorEnv
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=("random", "heuristic"), required=True)
    parser.add_argument("--setup", choices=("basic", "elemental"), default="elemental")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    match = (
        MatchConfig.authored("elemental-world-v2", "elemental_ur", "elemental_gw")
        if args.setup == "elemental"
        else MatchConfig(
            hero_deck={"Mountain": 20, "Gray Ogre": 12, "Lightning Bolt": 8},
            villain_deck={"Forest": 20, "Llanowar Elves": 20},
        )
    )
    slots, batch_steps, seed = 2, 128, 213
    started = time.perf_counter()
    env = VectorEnv(
        slots,
        Match(match),
        ObservationSpace(ObservationConfig(policy_input_version=6, max_facts=512)),
        Reward(RewardConfig()),
        device="cpu",
        seed=seed,
        opponent_policy=args.opponent,
    )
    try:
        env.reset()
        startup_seconds = time.perf_counter() - started
        # The first retained legal offer is a bounded stand-in for a learner.
        actions = torch.zeros(slots, dtype=torch.long)
        terminals = 0
        started = time.perf_counter()
        for _ in range(batch_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            terminals += int((terminated | truncated).sum())
        step_seconds = time.perf_counter() - started
    finally:
        env.close()
    print(
        json.dumps(
            {
                "opponent": args.opponent,
                "setup": args.setup,
                "world": managym.WORLD_VERSION,
                "platform": platform.platform(),
                "seed": seed,
                "slots": slots,
                "batch_steps": batch_steps,
                "learner_transitions": slots * batch_steps,
                "terminal_transitions": terminals,
                "startup_seconds": startup_seconds,
                "step_seconds": step_seconds,
                "learner_transitions_per_second": slots * batch_steps / step_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
