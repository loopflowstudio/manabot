"""
test_single_agent_env.py
Tests for single-agent wrapper and fixed-opponent policies.
"""

import numpy as np

# Local imports
from manabot.env.single_agent_env import (
    PassivePolicy,
    RandomPolicy,
    build_opponent_policy,
)


def make_obs(
    player_index: int,
    actions_valid: list[float] | None = None,
    pass_priority_slots: list[int] | None = None,
):
    actions_valid = actions_valid or [1.0, 1.0, 0.0, 0.0]
    pass_priority_slots = pass_priority_slots or []

    actions = np.zeros((len(actions_valid), 6), dtype=np.float32)
    for slot in pass_priority_slots:
        actions[slot, 2] = 1.0  # PRIORITY_PASS_PRIORITY

    agent_player = np.zeros((1, 26), dtype=np.float32)
    agent_player[0, 0] = 1.0
    return {
        "agent_player": agent_player,
        "actions": actions,
        "actions_valid": np.array(actions_valid, dtype=np.float32),
        "_player_index": player_index,
    }


def test_passive_policy_picks_pass_priority():
    obs = make_obs(player_index=1, actions_valid=[1, 1, 0, 0], pass_priority_slots=[1])
    action = PassivePolicy()(obs)
    assert action == 1


def test_random_policy_picks_only_valid_actions():
    obs = make_obs(player_index=1, actions_valid=[1, 0, 1, 0], pass_priority_slots=[])
    policy = RandomPolicy()
    chosen = {policy(obs) for _ in range(100)}
    assert chosen.issubset({0, 2})
    assert chosen


def test_build_opponent_policy_none():
    assert build_opponent_policy("none") is None
