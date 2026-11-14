"""Fixed opponent policies for bounded verification diagnostics."""

import numpy as np


class RandomPolicy:
    """Choose uniformly from valid actions."""

    def __call__(self, obs: dict[str, np.ndarray]) -> int:
        valid_indices = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")
        return int(np.random.choice(valid_indices))


def build_opponent_policy(name: str) -> RandomPolicy | None:
    if name == "none":
        return None
    if name == "random":
        return RandomPolicy()
    raise ValueError(f"Unsupported opponent_policy: {name}")
