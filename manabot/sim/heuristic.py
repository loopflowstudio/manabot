"""Scalar arena adapter over the viewer-safe `managym.Heuristic` selector.

This lives outside frozen `anchors.py`, whose bytes identify v1 registrations.
`heuristic_source_paths` binds new registrations to this adapter, the native
selector and knowledge modules, and the Python bindings. The rules, tie-breaks
and approximations are documented in `managym/src/agent/heuristic/mod.rs` and
`docs/engine.md`.
"""

from __future__ import annotations

from pathlib import Path

from manabot.env.env import Env
from manabot.env.observation import ObservationBuffers
import managym


def heuristic_source_paths() -> list[Path]:
    """Native rules and adapter bytes identifying newly registered opponents."""
    root = Path(managym.__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        root / "src/agent/heuristic/mod.rs",
        root / "src/agent/heuristic/knowledge.rs",
        root / "src/python/heuristic_bindings.rs",
    ]


class HeuristicPlayer:
    """Stateful adapter to the engine's viewer-safe heuristic selector."""

    def __init__(self, *, max_actions: int | None = None) -> None:
        if max_actions is not None and (
            type(max_actions) is not int or max_actions < 1
        ):
            raise ValueError("heuristic max_actions must be a positive integer")
        self._native = managym.Heuristic(max_actions=max_actions)

    def act(self, env: Env, obs: ObservationBuffers) -> int:
        del obs
        return self._native.select_action(env.last_raw_obs)
