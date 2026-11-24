"""Typed, read-only previews of a whole combat declaration.

The engine declares attackers and blockers one creature at a time. A preview
replays its own per-creature answers to a drafted plan on a native clone and
returns the exact offers that declare it, or the first creature it refuses and
why. The semantic address authorizes the query; nothing is committed.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

import managym
from managym.experience import ObjectRenderId


class PlannedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocker: ObjectRenderId
    attacker: ObjectRenderId


class CombatPlan(BaseModel):
    """Attackers for an attack, blocks for a block; a creature left out declines."""

    model_config = ConfigDict(extra="forbid")
    attackers: list[ObjectRenderId] = []
    blocks: list[PlannedBlock] = []


class DeclarableCreature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creature: ObjectRenderId
    label: str
    can_block: list[ObjectRenderId]
    must_block: bool
    must_attack: bool


class DeclarationStep(BaseModel):
    """One per-creature answer, at the native decision it will be asked at."""

    model_config = ConfigDict(extra="forbid")
    revision: int
    fingerprint: str
    creature: ObjectRenderId
    action_index: int
    attack: bool
    blocks: ObjectRenderId | None


class DeclarationRefusal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creature: ObjectRenderId
    label: str
    reason: str


class CombatPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["attackers", "blockers"]
    declarable: list[DeclarableCreature]
    steps: list[DeclarationStep]
    refused: DeclarationRefusal | None
    declared_attackers: list[ObjectRenderId]
    declared_blocks: list[PlannedBlock]


def preview_combat(
    env: managym.Env,
    *,
    viewer: int,
    revision: int,
    fingerprint: str,
    plan: CombatPlan,
) -> CombatPreview:
    """Reject stale/non-actor requests without advancing the native game."""
    return CombatPreview.model_validate_json(
        env.combat_preview_json(viewer, revision, fingerprint, plan.model_dump_json())
    )
