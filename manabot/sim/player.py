"""Inference players for scalar simulation; model admission lives in model.checkpoint."""

from enum import Enum
from typing import Dict

import numpy as np
import torch

# Local imports
from manabot.model.actor_critic import ActorCritic

# -----------------------------------------------------------------------------
# Player Classes
# -----------------------------------------------------------------------------


class PlayerType(Enum):
    """Types of players for evaluation."""

    MODEL = "model"
    RANDOM = "random"
    DEFAULT = "default"


class Player:
    """Base player class for evaluation."""

    def __init__(self, name: str, player_type: PlayerType):
        self.name = name
        self.player_type = player_type
        self.device = "cpu"

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from observation."""
        raise NotImplementedError

    def to(self, device: str) -> "Player":
        """Move player to specified device."""
        self.device = device
        return self


class ModelPlayer(Player):
    """Player that uses a trained model for inference."""

    def __init__(
        self,
        name: str,
        agent: ActorCritic,
        deterministic: bool = False,
    ):
        super().__init__(name, PlayerType.MODEL)
        self.agent = agent
        self.deterministic = deterministic
        self.device = next(agent.parameters()).device

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from model."""
        # Convert numpy arrays to tensors with batch dimension
        tensor_obs = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }

        # Get action from model
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                tensor_obs, deterministic=self.deterministic
            )
            return action.item()

    def to(self, device: str) -> "ModelPlayer":
        """Move player and model to device."""
        super().to(device)
        self.agent = self.agent.to(device)
        return self


class RandomPlayer(Player):
    """Player that selects random valid actions."""

    def __init__(self, name: str):
        super().__init__(name, PlayerType.RANDOM)

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get random valid action."""
        valid_actions = np.where(obs["actions_valid"] > 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        return int(np.random.choice(valid_actions))


class DefaultPlayer(Player):
    """Player that always plays 0."""

    def __init__(self, name: str):
        super().__init__(name, PlayerType.DEFAULT)

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        return 0
