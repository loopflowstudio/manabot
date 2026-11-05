#!/usr/bin/env python3
"""Collect one real self-play batch and print how credit reaches each decision.

This is a learning aid for docs/start-here.md, not a training entry point. It
uses the production collector and target code on an untrained or briefly trained
model, so every number is what the update would see. It plays a handful of small
games on one CPU thread and finishes in seconds.

    uv run python scripts/inspect_batch.py
    uv run python scripts/inspect_batch.py --policy-lambda 1
    uv run python scripts/inspect_batch.py --updates 20
"""

import argparse

import numpy as np
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.baseline import baseline
from manabot.training.learning.damped_policy import _targets
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector

STREAMS = 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-lambda", type=float, default=0.5)
    parser.add_argument("--value-lambda", type=float, default=0.8)
    parser.add_argument(
        "--updates", type=int, default=0, help="real updates to run before inspecting"
    )
    parser.add_argument(
        "--rows", type=int, default=12, help="decisions to print from one trajectory"
    )
    return parser.parse_args()


def _trajectories(ends: np.ndarray) -> list[range]:
    """Split packed rows into one range per seat trajectory, using end flags."""
    stops = (np.flatnonzero(ends) + 1).tolist()
    return [range(a, b) for a, b in zip([0, *stops[:-1]], stops, strict=True)]


def _report(
    batch: RolloutBatch, learner: Learner, learning: DampedPolicyLearning, rows: int
) -> None:
    targets = _targets(learner, batch, learning)
    advantages = targets.advantages.flatten().numpy()
    value_targets = targets.values.flatten().numpy()
    kept = targets.selected.flatten().numpy()
    values = batch.values.flatten()
    rewards = batch.rewards.flatten()
    ends = batch.dones.flatten()
    spans = _trajectories(ends)

    print(f"batch: {len(values)} rows, {len(spans)} seat trajectories")
    print(f"lengths: {[len(span) for span in spans]}")
    print(f"filter kept {int(kept.sum())} of {len(kept)} rows ({kept.mean():.0%})")
    print(f"bootstrapped tail fraction: {float((~batch.dones[-1]).mean()):.0f}")

    span = spans[0]
    outcome = {1.0: "won", -1.0: "lost", 0.0: "drew"}[float(rewards[span[-1]])]
    shown = list(span)[-rows:]
    print(f"\nlast {len(shown)} decisions of the first trajectory, which {outcome}:")
    print(f"{'to end':>6} {'V':>7} {'reward':>7} {'advantage':>10} {'target':>7}  kept")
    for row in shown:
        print(
            f"{span[-1] - row:>6} {values[row]:>7.3f} {rewards[row]:>7.1f} "
            f"{advantages[row]:>10.3f} {value_targets[row]:>7.3f}  "
            f"{'yes' if kept[row] else ''}"
        )


def main() -> None:
    args = _arguments()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    regime = baseline(steps=2).regime
    space = ObservationSpace(regime.observation)
    agent = ActorCritic(space, regime.agent)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        policy_lambda=args.policy_lambda,
        value_lambda=args.value_lambda,
    )
    collector = SeatRoutedCollector(
        space,
        Match(regime.match),
        Reward(RewardConfig()),
        num_envs=STREAMS,
        seed=args.seed,
        opponent_mode="self",
    )
    learner = Learner(agent, "cpu", minibatch_rows=STREAMS)
    rng = np.random.default_rng(args.seed)
    try:
        for iteration in range(1, args.updates + 1):
            batch = collector.collect(agent, 64)
            update_iteration(learner, batch, learning, 0.0, rng, iteration=iteration)
        _report(collector.collect(agent, 64), learner, learning, args.rows)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
