"""Ordinary rollout and learning through naming group and name decisions.

Naming adds ordinary decisions: each group choice and the final name choice is
one learner transition with the standard policy, value, discount and credit.
This is a bounded wiring check on real games, not a learning-quality claim.
"""

import numpy as np
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import Learning
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector

SOZIN = {"Swamp": 17, "The Rise of Sozin": 23}


def _naming_rows(batch: RolloutBatch) -> tuple[np.ndarray, np.ndarray]:
    """Flat rows whose chosen action was a naming group or a name."""
    obs = batch.observations().observation["name_candidates"]
    steps, envs = np.meshgrid(
        np.arange(batch.actions.shape[0]),
        np.arange(batch.actions.shape[1]),
        indexing="ij",
    )
    chosen = obs[steps, envs, batch.actions].reshape(-1, obs.shape[-1])
    return np.flatnonzero(chosen[:, 2] > 0), np.flatnonzero(chosen[:, 0] > 0)


def test_rollout_and_ppo_update_cover_group_and_name_transitions() -> None:
    torch.manual_seed(243)
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    collector = SeatRoutedCollector(
        space,
        Match(
            MatchConfig(
                hero="hero", villain="villain", hero_deck=SOZIN, villain_deck=SOZIN
            )
        ),
        Reward(RewardConfig()),
        num_envs=4,
        seed=243,
        opponent_mode="self",
        learning_seats="both",
    )
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, attention_on=False))
    batch = None
    for _ in range(20):
        batch = collector.collect(agent, num_steps=64)
        groups, names = _naming_rows(batch)
        if len(groups) and len(names):
            break
    assert batch is not None
    groups, names = _naming_rows(batch)
    assert len(groups) and len(names), "self-play reached no naming decision"
    obs = batch.observations().observation
    # Bounded prompts at ordinary capacity: no row offers more than 64 choices.
    flat_valid = obs["actions_valid"].reshape(-1, obs["actions_valid"].shape[-1])
    assert (flat_valid[np.concatenate([groups, names])].sum(-1) <= 64).all()
    # Every naming transition retains complete behavior evidence.
    batch.action_data.check(batch.actions, batch.logprobs)

    learner = Learner(agent, "cpu", minibatch_rows=batch.actions.size)
    assert agent.name_candidates is not None
    before = agent.name_candidates.embedding.weight.detach().clone()
    update_iteration(
        learner,
        batch,
        Learning(retained_fraction=1, epochs=1, minibatches=1, collection_kl=0),
        0,
        np.random.default_rng(0),
    )
    # The ordinary learner updates public name identities through both kinds
    # of transition; no separate route or joint-path objective exists.
    assert not torch.equal(agent.name_candidates.embedding.weight, before)
