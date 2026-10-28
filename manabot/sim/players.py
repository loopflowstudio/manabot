"""Torch-free code players for bounded simulation and their one builder.

`build_code_player` constructs every registered code player that needs no
policy network: the v1 anchors (`anchors`), the rule-of-thumb `heuristic` and
the demo flat search. Evaluation game children build these without loading
Torch. Search and checkpoint players are built by `guidance`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from manabot.env.env import Env
from manabot.sim.anchors import RandomMatchupPlayer, ScriptedGreedyPlayer
from manabot.sim.heuristic import HeuristicPlayer
from manabot.sim.registration import PlayerRegistration
from manabot.sim.search.flat import FlatSearchPolicy

CODE_PLAYER_KINDS = frozenset({"demo_search", "heuristic", "random", "scripted_greedy"})


class DemoSearchPlayer:
    """Run the exact demo policy, including its forced-offer seed behavior."""

    def __init__(self, spec: dict[str, Any], seed: int) -> None:
        self.policy = FlatSearchPolicy(
            sims=spec["sims"],
            rollouts_per_world=spec["rollouts_per_world"],
            max_steps=spec["max_steps"],
            seed=seed,
        )

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        return self.policy(env._engine, env.last_raw_obs)


CodePlayer = (
    DemoSearchPlayer | HeuristicPlayer | RandomMatchupPlayer | ScriptedGreedyPlayer
)


def build_code_player(registration: PlayerRegistration, *, seed: int) -> CodePlayer:
    """Build a code player; `seed` drives the random anchor and demo search."""
    spec = registration.player_spec
    kind = spec["kind"]
    if kind == "demo_search":
        return DemoSearchPlayer(dict(spec), seed)
    if kind == "heuristic":
        return HeuristicPlayer()
    if kind == "scripted_greedy":
        return ScriptedGreedyPlayer()
    if kind == "random":
        return RandomMatchupPlayer(seed=seed)
    raise ValueError(f"not a code player kind: {kind}")
