"""Capture forward/learner outputs for the MTG-299 bitwise check.

uv run python scratch/bench/capture.py <out.pt>
Works before and after admission exists (admits when available).
"""

from copy import deepcopy
import sys

import numpy as np
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.damped_policy import update_damped_iteration
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import SeatRoutedCollector
from tests.training.test_recovery import recipe

try:
    from manabot.model.admission import admit_observation
except ImportError:
    admit_observation = None


def admit(agent: ActorCritic, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return obs if admit_observation is None else admit_observation(agent, obs)


def build(variant: str) -> tuple[ActorCritic, SeatRoutedCollector]:
    torch.set_num_threads(1)
    torch.manual_seed(819)
    config = recipe()
    config.agent.value_aggregation = "masked_mean"
    if variant == "history":
        config.observation.policy_history_version = 1
        config.agent.recent_events = True
    if variant == "generic":
        config.agent.world = None
    if variant == "wdl":
        config.agent.value_kind = "categorical_wdl"
        config.agent.value_aggregation = "value_token"
    space = ObservationSpace(config.observation)
    agent = ActorCritic(space, config.agent).eval()
    collector = SeatRoutedCollector(
        space,
        Match(config.match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=91,
        opponent_mode="self",
        recovery_max_microsteps=10000,
    )
    return agent, collector


def main(path: str) -> None:
    out: dict[str, torch.Tensor] = {}
    for variant in ("semantic", "generic", "history", "wdl"):
        agent, collector = build(variant)
        batch = collector.collect(agent, 16)
        rows = batch.observations(np.arange(batch.actions.size))
        obs = {
            key: torch.from_numpy(value).float()
            if key == "action_focus"
            else torch.from_numpy(value)
            for key, value in rows.observation.items()
        }
        with torch.no_grad():
            logits, value = agent.forward_distribution(admit(agent, obs))
        out[f"{variant}/nograd/logits"] = logits
        out[f"{variant}/nograd/value"] = value
        trained = deepcopy(agent).train()
        logits, value = trained.forward_distribution(admit(trained, obs))
        out[f"{variant}/grad/logits"] = logits.detach()
        out[f"{variant}/grad/value"] = value.detach()
        (logits[obs["actions_valid"].bool()].square().mean() + value.square().mean()).backward()
        for name, parameter in trained.named_parameters():
            if parameter.grad is not None:
                out[f"{variant}/grad/{name}"] = parameter.grad.clone()
        if variant in ("semantic", "wdl"):
            for rule_name, rule in (
                ("ppo", Learning(epochs=2, minibatches=2)),
                ("damped", DampedPolicyLearning(gradient="damped_policy")),
            ):
                if rule_name == "ppo" and variant == "wdl":
                    continue
                learner_agent = deepcopy(agent)
                learner = Learner(learner_agent, "cpu", minibatch_rows=2)
                rng = np.random.default_rng(5)
                update_iteration(learner, batch, rule, 0.5, rng)
                for name, parameter in learner.agent.named_parameters():
                    out[f"{variant}/{rule_name}/{name}"] = parameter.detach().clone()
    torch.save(out, path)
    print(len(out), "tensors")


if __name__ == "__main__":
    main(sys.argv[1])
