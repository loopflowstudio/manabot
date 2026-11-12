"""Inference players for scalar simulation; model admission lives in model.checkpoint.

`Player` is the interface `manabot.sim.sim` plays: a name and one action per
encoded observation. `load_player` in that module picks the implementation.
"""

from typing import Protocol

import numpy as np
import torch

from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import admit_on_device


class Player(Protocol):
    name: str

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        """One action index for an encoded observation."""
        ...


class ModelPlayer:
    """Player that uses a trained model for inference."""

    def __init__(
        self,
        name: str,
        agent: ActorCritic,
        deterministic: bool = False,
    ) -> None:
        self.name = name
        self.agent = agent
        self.deterministic = deterministic

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        tensor_obs = admit_on_device(
            self.agent,
            {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in obs.items()
            },
        )
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                tensor_obs, deterministic=self.deterministic
            )
            return int(action.item())


class RandomPlayer:
    """Player that selects a uniformly random valid action from numpy's global RNG."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        valid_actions = np.where(obs["actions_valid"] > 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        return int(np.random.choice(valid_actions))


class DefaultPlayer:
    """Player that always plays action 0."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        return 0
