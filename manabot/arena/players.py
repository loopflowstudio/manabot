"""Stable arena player factory over current executable player seams."""

from __future__ import annotations

from typing import Any

import numpy as np

from manabot.env import Env, ObservationSpace
from manabot.sim.flat_mc import MatchupPlayer, make_player
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


def build_player(
    registration: Any, *, seed: int, checkpoint_path: str | None = None
) -> tuple[MatchupPlayer, ObservationSpace | None]:
    spec = dict(registration.player_spec)
    if registration.runner_kind == "checkpoint":
        if checkpoint_path is None:
            raise FileNotFoundError("checkpoint candidate bytes are unavailable")
        spec["path"] = checkpoint_path
    if spec["kind"] == "scripted_greedy":
        return ScriptedGreedyPlayer(), None
    player, obs_space = make_player(spec, seed=seed)
    attribute_names = {
        "sims": "simulations" if spec["kind"] == "determinized_puct" else "sims",
        "rollouts_per_world": "rollouts",
        "worlds": "worlds",
        "c_puct": "c_puct",
        "max_steps": "max_steps",
        "branch_driver_id": "branch_driver_id",
    }
    for field, attribute in attribute_names.items():
        if field in spec and getattr(player, attribute, None) != spec[field]:
            raise RuntimeError(f"constructed player did not echo {field}")
    if spec["kind"] == "determinized_puct":
        semantics = registration.search_semantics
        if semantics is None:
            raise RuntimeError("determinized PUCT has no registered search semantics")
        if player.branch_audit != semantics.branch_audit:
            raise RuntimeError("constructed player did not echo branch_audit")
        if type(player.evaluator).__name__ != "UniformRandomLeafEvaluator":
            raise RuntimeError(
                "constructed player did not use the registered leaf evaluator"
            )
    return player, obs_space
