"""Typed grouped-input descriptors, read-only previews and atomic application.

Multi-part decisions (combat declarations, Scry, Surveil and OrderLibrary) keep
their flat sequential native ``Action``s. These models wrap ``Env.group_json``,
``Env.preview_group_json`` and ``Env.apply_group_json`` so callers describe the
active group, preview a whole drafted plan on a native clone, and apply a valid
plan atomically without touching the action space or observation contract.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import managym
from managym.experience import ObjectRenderId

LibraryEnd = Literal["top", "bottom"]


class PlannedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocker: ObjectRenderId
    attacker: ObjectRenderId


class GroupCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: ObjectRenderId
    label: str


class AttackerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: ObjectRenderId
    label: str
    must_attack: bool


class BlockerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: ObjectRenderId
    label: str
    can_block: list[ObjectRenderId]
    must_block: bool


class _GroupAddress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int
    fingerprint: str


class DeclareAttackersGroup(_GroupAddress):
    kind: Literal["declare_attackers"] = "declare_attackers"
    declarable: list[AttackerItem]
    committed: list[ObjectRenderId]


class DeclareBlockersGroup(_GroupAddress):
    kind: Literal["declare_blockers"] = "declare_blockers"
    declarable: list[BlockerItem]
    committed: list[PlannedBlock]


class ScryGroup(_GroupAddress):
    kind: Literal["scry"] = "scry"
    items: list[GroupCard]
    committed_top: list[ObjectRenderId]
    committed_bottom: list[ObjectRenderId]


class SurveilGroup(_GroupAddress):
    kind: Literal["surveil"] = "surveil"
    items: list[GroupCard]
    committed_graveyard: list[ObjectRenderId]


class OrderLibraryGroup(_GroupAddress):
    kind: Literal["order_library"] = "order_library"
    end: LibraryEnd
    items: list[GroupCard]
    committed: list[ObjectRenderId]


type Group = Annotated[
    DeclareAttackersGroup
    | DeclareBlockersGroup
    | ScryGroup
    | SurveilGroup
    | OrderLibraryGroup,
    Field(discriminator="kind"),
]


class DeclareAttackersPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["declare_attackers"] = "declare_attackers"
    attackers: list[ObjectRenderId] = Field(default_factory=list)


class DeclareBlockersPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["declare_blockers"] = "declare_blockers"
    blocks: list[PlannedBlock] = Field(default_factory=list)


class ScryPlan(BaseModel):
    """`top` is drawn-first; `bottom` is top-to-bottom of the bottom group."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["scry"] = "scry"
    top: list[ObjectRenderId] = Field(default_factory=list)
    bottom: list[ObjectRenderId] = Field(default_factory=list)


class SurveilPlan(BaseModel):
    """Cards to put into the graveyard; all remaining looked-at cards stay on top."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["surveil"] = "surveil"
    graveyard: list[ObjectRenderId] = Field(default_factory=list)


class OrderLibraryPlan(BaseModel):
    """Remaining cards in desired order, drawn-first at index 0."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["order_library"] = "order_library"
    order: list[ObjectRenderId] = Field(default_factory=list)


type GroupPlan = Annotated[
    DeclareAttackersPlan
    | DeclareBlockersPlan
    | ScryPlan
    | SurveilPlan
    | OrderLibraryPlan,
    Field(discriminator="kind"),
]


class GroupStep(BaseModel):
    """One sequential native step compiled from a grouped plan."""

    model_config = ConfigDict(extra="forbid")
    revision: int
    fingerprint: str
    action_index: int
    item: ObjectRenderId | None


class GroupRefusal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: ObjectRenderId | None
    label: str
    reason: str


class DeclareAttackersOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["declare_attackers"] = "declare_attackers"
    attackers: list[ObjectRenderId]


class DeclareBlockersOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["declare_blockers"] = "declare_blockers"
    blocks: list[PlannedBlock]


class ScryOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["scry"] = "scry"
    top: list[ObjectRenderId]
    bottom: list[ObjectRenderId]


class SurveilOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["surveil"] = "surveil"
    graveyard: list[ObjectRenderId]
    kept: list[ObjectRenderId]


class OrderLibraryOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["order_library"] = "order_library"
    end: LibraryEnd
    order: list[ObjectRenderId]


type GroupOutcome = Annotated[
    DeclareAttackersOutcome
    | DeclareBlockersOutcome
    | ScryOutcome
    | SurveilOutcome
    | OrderLibraryOutcome,
    Field(discriminator="kind"),
]


class GroupPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: Group
    steps: list[GroupStep]
    refused: GroupRefusal | None
    complete: bool
    outcome: GroupOutcome | None


_GROUP_ADAPTER: TypeAdapter[Group] = TypeAdapter(Group)
_PLAN_ADAPTER: TypeAdapter[GroupPlan] = TypeAdapter(GroupPlan)


def group(env: managym.Env, *, viewer: int) -> Group | None:
    """Return the active grouped decision for `viewer`, or `None`."""
    raw = env.group_json(viewer)
    if raw is None:
        return None
    return _GROUP_ADAPTER.validate_json(raw)


def preview_group(
    env: managym.Env,
    *,
    viewer: int,
    revision: int,
    fingerprint: str,
    plan: GroupPlan,
) -> GroupPreview:
    """Read-only preview bound to `(viewer, revision, fingerprint)`."""
    plan_json = _PLAN_ADAPTER.dump_json(plan).decode()
    return GroupPreview.model_validate_json(
        env.preview_group_json(viewer, revision, fingerprint, plan_json)
    )


def apply_group(
    env: managym.Env,
    *,
    viewer: int,
    revision: int,
    fingerprint: str,
    plan: GroupPlan,
) -> GroupPreview:
    """Validate on a clone and execute all compiled steps atomically."""
    plan_json = _PLAN_ADAPTER.dump_json(plan).decode()
    return GroupPreview.model_validate_json(
        env.apply_group_json(viewer, revision, fingerprint, plan_json)
    )
