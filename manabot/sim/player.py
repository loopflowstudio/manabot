"""Scalar simulation players; model players receive an admitted backend policy.

`Player` is the interface `manabot.sim.sim` plays: a name and one action per
viewer observation. `load_player` in that module picks the implementation.
"""

from typing import Protocol

import numpy as np

from manabot.env.observation import encode_policy_batch
from manabot.policy.protocol import BackendPolicy
import managym


class Player(Protocol):
    name: str

    def get_action(self, obs: managym.Observation) -> int:
        """One action index for an viewer observation."""
        ...


class ModelPlayer:
    """Player that uses a trained model for inference."""

    def __init__(
        self,
        name: str,
        agent: BackendPolicy,
        deterministic: bool = False,
    ) -> None:
        self.name = name
        self.agent = agent
        self.deterministic = deterministic

    def get_action(self, obs: managym.Observation) -> int:
        batch = self.agent.prepare(
            encode_policy_batch(self.agent.observation_space, [obs]),
            inference=True,
        )
        return int(self.agent.sample(batch, deterministic=self.deterministic)[0])


class RandomPlayer:
    """Player that selects a uniformly random valid action from numpy's global RNG."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_action(self, obs: managym.Observation) -> int:
        valid_actions = np.arange(len(obs.action_space.actions))
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        return int(np.random.choice(valid_actions))


class DefaultPlayer:
    """Player that always plays action 0."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_action(self, obs: managym.Observation) -> int:
        return 0
