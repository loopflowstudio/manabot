"""A self-play PPO loop written from the pieces alone, with no recipe or executor."""

from copy import deepcopy

import numpy as np
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.policy_distribution import legal_mask, policy_logs
from manabot.rl.losses import clipped_surrogate, entropy, squared_value_loss
from manabot.rl.returns import standardized_advantages, transition_gae
from manabot.rl.step import gradient_step, update_ema
from manabot.training.rollout.collection import close_collector, collect, open_collector
from manabot.training.updates.learner import AdmittedRollout


def train(
    iterations: int,
    *,
    streams: int = 4,
    decisions: int = 64,
    seed: int = 0,
    model: ModelConfig | None = None,
) -> ActorCritic:
    """Return the EMA of a policy trained `iterations` collect/update rounds."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = ActorCritic(ObservationSpace(), model or ModelConfig(hidden_dim=64))
    averaged = deepcopy(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
    collector = open_collector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",  # one model plays and learns from both seats
    )
    try:
        for _ in range(iterations):
            # Whole games, at least `decisions` learner rows per stream.
            batch = collect(collector, agent, decisions)
            ends = torch.as_tensor(batch.dones)
            values = torch.as_tensor(batch.values)
            # Complete games end every trajectory, so the tail value is never read.
            tail = torch.zeros(ends.shape[1])
            advantages, returns = transition_gae(
                torch.as_tensor(batch.rewards), values, ends, tail, gamma=1.0, lam=0.95
            )
            # Time-major [T,E] rows, the order `batch.observations` indexes.
            advantages, returns = advantages.flatten(), returns.flatten()
            advantages = standardized_advantages(advantages)
            actions = torch.as_tensor(batch.actions).flatten()
            behavior = torch.as_tensor(batch.logprobs).flatten()

            # Admission checks each row's policy input once, before any epoch.
            admitted = AdmittedRollout(agent, batch)
            admitted.admit_rows(np.arange(len(actions)), 256)
            agent.train()
            for _epoch in range(2):
                order = torch.as_tensor(rng.permutation(len(actions)))
                for rows in order.split(256):
                    obs, _ = admitted.rows(rows.numpy(), torch.device("cpu"))
                    valid = legal_mask(obs["actions_valid"])
                    logits, value = agent(obs)
                    logs = policy_logs(logits, valid)
                    chosen = logs.gather(-1, actions[rows, None]).squeeze(-1)
                    loss = (
                        clipped_surrogate(
                            chosen - behavior[rows], advantages[rows], 0.2
                        ).mean()
                        + 0.5 * squared_value_loss(value, returns[rows]).mean()
                        - 0.01 * entropy(logs, valid).mean()
                    )
                    gradient_step(loss, agent, optimizer, max_grad_norm=0.5)
            update_ema(averaged, agent, rate=0.99)
    finally:
        close_collector(collector)
    return averaged
