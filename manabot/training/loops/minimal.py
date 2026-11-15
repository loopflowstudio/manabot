"""A self-play PPO loop written from the pieces alone, with no recipe or executor."""

import numpy as np

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.policy.protocol import BackendPolicy
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.backend import TrainingBackend


def train(
    iterations: int,
    *,
    backend: TrainingBackend,
    streams: int = 4,
    decisions: int = 64,
    seed: int = 0,
    model: ModelConfig | None = None,
) -> BackendPolicy:
    """Return the EMA of a policy trained `iterations` collect/update rounds."""
    backend.runtime.require_execution()
    backend.seed(seed)
    rl = backend.rl
    array = backend.arrays.from_numpy
    rng = np.random.default_rng(seed)
    agent = backend.runtime.create_policy(
        ObservationSpace(), model or ModelConfig(hidden_dim=64)
    )
    averaged = backend.state.clone_policy(agent)
    learner = backend.learner.Learner(agent, "cpu", minibatch_rows=256)
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",
        numerics=backend.collection.CollectionSession(
            device="cpu", seed=seed, precision="float32"
        ),  # one model plays and learns from both seats
    )
    try:
        for _ in range(iterations):
            # Whole games, at least `decisions` learner rows per stream.
            batch = collector.collect(agent, decisions)
            ends = array(batch.dones)
            values = array(batch.values)
            # Complete games end every trajectory, so the tail value is never read.
            tail = array(np.zeros(ends.shape[1], dtype=np.float32))
            advantages, returns = rl.transition_gae(
                array(batch.rewards), values, ends, tail, gamma=1.0, lam=0.95
            )
            # Time-major [T,E] rows, the order `batch.observations` indexes.
            advantages, returns = advantages.flatten(), returns.flatten()
            advantages = rl.standardized_advantages(advantages)
            actions = array(batch.actions).flatten()
            behavior = array(batch.logprobs).flatten()

            # Admission checks each row's policy input once, before any epoch.
            admitted = backend.rollout.AdmittedRollout(agent, batch)
            admitted.admit_rows(np.arange(len(actions)), 256)
            for _epoch in range(2):
                order = rng.permutation(len(actions))
                for start in range(0, len(order), 256):
                    indexes = order[start : start + 256]
                    rows = array(indexes)
                    obs, _ = admitted.rows(indexes, learner.device)
                    valid = admitted.legal(obs)
                    logits, value = agent(obs)
                    logs = rl.policy_logs(logits, valid)
                    chosen = rl.action_logs(logs, actions[rows])
                    loss = (
                        rl.clipped_surrogate(
                            chosen - behavior[rows], advantages[rows], 0.2
                        ).mean()
                        + 0.5 * rl.squared_value_loss(value, returns[rows]).mean()
                        - 0.01 * rl.entropy(logs, valid).mean()
                    )
                    learner.step(loss, max_grad_norm=0.5)
            backend.learner.update_ema(averaged, agent, rate=0.99)
    finally:
        collector.close()
    return averaged
