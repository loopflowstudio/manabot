"""Typed, read-only previews of a whole combat declaration.

Deprecated wrapper over :meth:`managym.Env.preview` (scheduled for removal in
0.3.0). New callers should use ``env.decision()``, ``env.preview(choice)``, and
``env.submit(choice)``.
"""

from typing import Literal
import warnings

from pydantic import BaseModel, ConfigDict

import managym
from managym.experience import ObjectRenderId
from managym.decision_group import PlannedAttack, PlannedBlock


class CombatPlan(BaseModel):
    """Attackers for an attack, blocks for a block; a creature left out declines."""

    model_config = ConfigDict(extra="forbid")
    attackers: list[PlannedAttack] = []
    blocks: list[PlannedBlock] = []


class DeclarableCreature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creature: ObjectRenderId
    label: str
    can_attack: list[ObjectRenderId] = []
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
    defender: ObjectRenderId | None = None
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
    declared_attackers: list[PlannedAttack]
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
    warnings.warn(
        "managym.combat.preview_combat is deprecated and removed in 0.3.0; "
        "use env.preview(choice, at=...)",
        DeprecationWarning,
        stacklevel=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        raw = env.combat_preview_json(
            viewer, revision, fingerprint, plan.model_dump_json()
        )
    return CombatPreview.model_validate_json(raw)
