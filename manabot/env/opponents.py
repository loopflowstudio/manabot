"""Fixed opponent policies for bounded verification diagnostics."""

import numpy as np

import managym


class RandomPolicy:
    """Choose uniformly from valid actions."""

    def __call__(self, obs: managym.Observation) -> int:
        valid_indices = np.arange(len(obs.action_space.actions))
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")
        return int(np.random.choice(valid_indices))


def build_opponent_policy(name: str) -> RandomPolicy | None:
    if name == "none":
        return None
    if name == "random":
        return RandomPolicy()
    raise ValueError(f"Unsupported opponent_policy: {name}")
