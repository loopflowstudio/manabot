"""Fixed opponent policies for bounded verification diagnostics."""

import numpy as np

from .observation import ActionEnum


class PassivePolicy:
    """Always pass priority when possible."""

    def __call__(self, obs: dict[str, np.ndarray]) -> int:
        valid_indices = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")

        pass_priority_type = int(ActionEnum.PRIORITY_PASS_PRIORITY)
        for idx in valid_indices:
            if obs["actions"][idx, pass_priority_type] > 0:
                return int(idx)
        return int(valid_indices[0])


class RandomPolicy:
    """Choose uniformly from valid actions."""

    def __call__(self, obs: dict[str, np.ndarray]) -> int:
        valid_indices = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")
        return int(np.random.choice(valid_indices))


def build_opponent_policy(name: str):
    if name == "none":
        return None
    if name == "passive":
        return PassivePolicy()
    if name == "random":
        return RandomPolicy()
    raise ValueError(f"Unsupported opponent_policy: {name}")
