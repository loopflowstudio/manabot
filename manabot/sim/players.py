"""Registered code and checkpoint players for bounded simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from manabot.env import Env, ObservationSpace
from manabot.infra.identity import file_sha256
from manabot.sim.search.flat import FlatSearchPolicy
from manabot.sim.search.flat_mc import MatchupPlayer, make_player
from manabot.sim.search.puct import puct_source_sha256
import managym


class DemoSearchPlayer:
    """Run the exact demo policy, including its forced-offer seed behavior."""

    def __init__(self, spec: dict[str, Any], seed: int):
        self.policy = FlatSearchPolicy(
            sims=spec["sims"],
            rollouts_per_world=spec["rollouts_per_world"],
            max_steps=spec["max_steps"],
            seed=seed,
        )

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        return self.policy(env._engine, env.last_raw_obs)


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
    if spec["kind"] == "demo_search":
        return DemoSearchPlayer(spec, seed), None
    if registration.runner_kind == "checkpoint":
        if checkpoint_path is None:
            raise FileNotFoundError("checkpoint candidate bytes are unavailable")
        spec["path"] = checkpoint_path
        if spec["kind"] == "agent_puct":
            if puct_source_sha256() != spec["implementation_source_sha256"]:
                raise ValueError("agent PUCT implementation source drift")
            if (
                file_sha256(checkpoint_path) != registration.checkpoint_sha256
                or Path(checkpoint_path).stat().st_size != registration.checkpoint_bytes
            ):
                raise ValueError("agent PUCT checkpoint byte identity drift")
            spec["checkpoint"] = checkpoint_path
        if spec["kind"] == "local_update":
            from manabot.sim.search import local_update

            if (
                local_update.local_search_source_sha256()
                != spec["implementation_source_sha256"]
            ):
                raise ValueError("local-update implementation source drift")
            spec["checkpoint"] = checkpoint_path
            spec["checkpoint_sha256"] = registration.checkpoint_sha256
    if spec["kind"] == "scripted_greedy":
        return ScriptedGreedyPlayer(), None
    player, obs_space = make_player(spec, seed=seed)
    attribute_names = {
        "sims": "simulations"
        if spec["kind"] in {"determinized_puct", "agent_puct"}
        else "sims",
        "rollouts_per_world": "rollouts",
        "worlds": "worlds",
        "c_puct": "c_puct",
        "max_steps": "max_steps",
        "branch_driver_id": "branch_driver_id",
        "branch_audit": "branch_audit",
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
    if spec["kind"] == "agent_puct":
        if player.evaluator.value_mode != spec["value_mode"]:
            raise RuntimeError("constructed player did not echo value_mode")
        if (
            sum(p.numel() for p in player.evaluator.agent.parameters())
            != registration.parameter_count
        ):
            raise ValueError("agent PUCT checkpoint parameter count drift")
    return player, obs_space
