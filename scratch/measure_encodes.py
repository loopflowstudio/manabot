"""Count catalog encodes per optimizer step on real learner updates (MTG-300).

Single process, one thread. Counts SemanticCardEncoder.definitions calls by
kind and grad mode, Adam steps, and the CPU profiler share of the encode.
"""

from collections import Counter
from copy import deepcopy

import numpy as np
import managym
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import DampedPolicyLearning, Learning, Schedule
from manabot.training.rollout.collection import SeatRoutedCollector

torch.set_num_threads(1)
torch.manual_seed(0)
STREAMS = 8
counts: Counter[str] = Counter()
rows_encoded: Counter[str] = Counter()
original = SemanticCardEncoder.definitions


def counted(self: SemanticCardEncoder, rows: torch.Tensor | None = None) -> torch.Tensor:
    key = ("referenced" if rows is not None else "full") + (
        "/grad" if torch.is_grad_enabled() else "/no_grad"
    )
    counts[key] += 1
    rows_encoded[key] += len(self.tokens) if rows is None else len(rows)
    with record_function("catalog_encode"), record_function(key):
        return original(self, rows)


SemanticCardEncoder.definitions = counted  # type: ignore[method-assign]

agent = ActorCritic(ObservationSpace(), ModelConfig(world=managym.WORLD_SETUP))
assert agent.semantic_cards is not None
print("catalog programs:", len(agent.semantic_cards.tokens))
collector = SeatRoutedCollector(
    agent.observation_space,
    Match(),
    Reward(RewardConfig()),
    num_envs=STREAMS,
    opponent_mode="self",
    learning_seats="both",
)
batch = collector.collect(agent, 16)
counts.clear()
rows_encoded.clear()
print("collection encodes (PolicyInference per scope) cleared; rows", batch.actions.size)

rules = {
    "damped": DampedPolicyLearning(
        gradient="damped_policy", advantage_quantile=0, min_advantage=0
    ),
    "ppo": Learning(
        retained_fraction=1,
        epochs=2,
        minibatches=4,
        learning_rate=Schedule(initial=0.01),
        tau=Schedule(initial=0),
        collection_kl=0,
    ),
}
for name, learning in rules.items():
    learner = Learner(deepcopy(agent), "cpu", minibatch_rows=STREAMS)
    steps = 0
    step = learner.optimizer.step

    def counting_step(*args: object, **kwargs: object) -> object:
        global steps
        steps += 1
        return step(*args, **kwargs)

    learner.optimizer.step = counting_step  # type: ignore[method-assign]
    counts.clear()
    rows_encoded.clear()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("update"):
            update_iteration(
                learner, batch, learning, 0.5, np.random.default_rng(3), iteration=1
            )
    events = {e.key: e for e in prof.key_averages()}
    update_us = events["update"].cpu_time_total
    encode_us = events["catalog_encode"].cpu_time_total
    backward = [
        e for e in prof.key_averages() if "Gru" in e.key and "Backward" in e.key
    ]
    print(f"\n[{name}] optimizer steps: {steps}")
    print(f"[{name}] encodes: {dict(counts)}")
    print(f"[{name}] rows encoded: {dict(rows_encoded)}")
    print(
        f"[{name}] encode forward CPU share of update: {encode_us / update_us:.1%}"
    )
    for e in backward:
        print(f"[{name}]   {e.key}: {e.cpu_time_total / update_us:.1%}")
    for key in counts:
        print(f"[{name}]   {key}: {events[key].cpu_time_total / update_us:.1%}")
    grads = sorted(
        (e for e in prof.key_averages() if "ackward" in e.key),
        key=lambda e: -e.cpu_time_total,
    )[:8]
    for e in grads:
        print(f"[{name}]   bw {e.key}: {e.cpu_time_total / update_us:.1%}")
