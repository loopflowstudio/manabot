"""Scalar simulation players; model players receive an admitted backend policy."""

from enum import Enum

import numpy as np

from manabot.env.observation import ObservationBuffers
from manabot.policy.protocol import BackendPolicy

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

    def __init__(self, name: str, player_type: PlayerType) -> None:
        self.name = name
        self.player_type = player_type
        self.device = "cpu"

    def get_action(self, obs: ObservationBuffers) -> int:
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
        agent: BackendPolicy,
        deterministic: bool = False,
    ) -> None:
        super().__init__(name, PlayerType.MODEL)
        self.agent = agent
        self.device = agent.device_type
        self.deterministic = deterministic

    def get_action(self, obs: ObservationBuffers) -> int:
        """Get action from model."""
        batch = self.agent.prepare(
            {
                key: np.asarray(value, dtype=np.float32)[None]
                for key, value in obs.items()
            },
            inference=True,
        )
        return int(self.agent.sample(batch, deterministic=self.deterministic)[0])

    def to(self, device: str) -> "ModelPlayer":
        """Move player and model to device."""
        super().to(device)
        self.agent = self.agent.to(device)
        return self


class RandomPlayer(Player):
    """Player that selects random valid actions."""

    def __init__(self, name: str) -> None:
        super().__init__(name, PlayerType.RANDOM)

    def get_action(self, obs: ObservationBuffers) -> int:
        """Get random valid action."""
        valid_actions = np.where(obs["actions_valid"] > 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        return int(np.random.choice(valid_actions))


class DefaultPlayer(Player):
    """Player that always plays 0."""

    def __init__(self, name: str) -> None:
        super().__init__(name, PlayerType.DEFAULT)

    def get_action(self, obs: ObservationBuffers) -> int:
        return 0
