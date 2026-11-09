"""Frozen v1 code anchors: scripted greedy and uniform random players.

These players read only the acting viewer's offers or encoded observation and
stay Torch-free, so an evaluation game child that plays them never loads Torch
(`players.build_code_player` builds them). Arena registrations pin this file's
bytes as the scripted and random source identity; `heuristic` builds on
`ScriptedGreedyPlayer`.
"""

from __future__ import annotations

import numpy as np

from manabot.env.env import Env
import managym


class ScriptedGreedyPlayer:
    """Deterministic viewer-safe baseline over the current offered actions."""

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        raw = env.last_raw_obs
        actions = list(raw.action_space.actions)
        kind = int(raw.action_space.action_space_type)
        if not actions:
            raise RuntimeError("scripted player received no legal offers")
        if kind == int(managym.ActionSpaceEnum.PRIORITY):
            for desired in (
                managym.ActionEnum.PRIORITY_PLAY_LAND,
                managym.ActionEnum.PRIORITY_CAST_SPELL,
                managym.ActionEnum.PRIORITY_PASS_PRIORITY,
            ):
                for index, action in enumerate(actions):
                    if int(action.action_type) == int(desired):
                        return index
        if kind == int(managym.ActionSpaceEnum.DECLARE_ATTACKER):
            for index, action in enumerate(actions):
                if int(action.action_type) == int(managym.ActionEnum.DECLARE_ATTACKER):
                    return index
        if kind == int(managym.ActionSpaceEnum.DECLARE_BLOCKER):
            for index, action in enumerate(actions):
                if int(action.action_type) == int(managym.ActionEnum.DECLARE_BLOCKER):
                    return index
        for index, action in enumerate(actions):
            if int(action.action_type) != int(managym.ActionEnum.DECLINE_CHOICE):
                return index
        return len(actions) - 1


class RandomMatchupPlayer:
    """Uniform random over valid actions in the encoded observation.

    Matches the RandomPolicy used for every historical vs-random baseline.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del env
        valid = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid) == 0:
            return 0
        return int(self._rng.choice(valid))
