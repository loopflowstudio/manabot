"""Etude experience-boundary consumer of managym's shared semantic contract.

Etude presents viewer-safe :class:`~managym.decision.Observation` values and
lowers interaction drafts into atomic revision-bound
:class:`~managym.decision.Command` values; it never reconstructs rules
meaning, viewer filtering, or replay. This adapter is the Etude-side vertical
slice of ``docs/ARCHITECTURE.md`` step 1 / Rules R1: it projects a composite
viewer-safe Observation for a historical viewer and applies a Command through
the same authority manabot uses. Match identity remains Etude-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from managym.decision import (
    Command,
    DecisionFrame,
    Observation,
    SemanticTransition,
    apply_semantic_command,
)


@dataclass
class SemanticExperienceBoundary:
    """Etude adapter over managym's shared semantic authority."""

    @staticmethod
    def decision_frame(env: Any) -> DecisionFrame:
        return DecisionFrame.from_json(env.semantic_decision_frame_json())

    @staticmethod
    def observe(env: Any, viewer: int) -> Observation:
        return Observation.from_json(env.semantic_observation_json(viewer))

    @staticmethod
    def apply(env: Any, command: Command | Mapping[str, Any]) -> SemanticTransition:
        return apply_semantic_command(env, command)
