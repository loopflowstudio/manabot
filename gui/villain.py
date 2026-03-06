"""
villain.py
Simple villain policies for raw managym observations.
"""

import random
from typing import Callable

# Local imports
from manabot.env.observation import ActionEnum
import managym


def passive_policy(obs: managym.Observation) -> int:
    """Pass priority when possible, otherwise pick the first available action."""
    actions = obs.action_space.actions
    if not actions:
        raise ValueError("No available actions for villain policy.")

    pass_priority = int(ActionEnum.PRIORITY_PASS_PRIORITY)
    for index, action in enumerate(actions):
        if int(action.action_type) == pass_priority:
            return index

    return 0


def random_policy(obs: managym.Observation) -> int:
    """Select a random legal action."""
    actions = obs.action_space.actions
    if not actions:
        raise ValueError("No available actions for villain policy.")
    return random.randrange(len(actions))


POLICIES: dict[str, Callable[[managym.Observation], int]] = {
    "passive": passive_policy,
    "random": random_policy,
}


def build_villain_policy(name: str) -> Callable[[managym.Observation], int]:
    """Build a villain policy by name."""
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported villain_type: {name}") from exc
