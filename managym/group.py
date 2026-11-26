"""Unified interactive decision, preview, and atomic submission types.

Multi-part decisions (combat declarations, Scry, Surveil, OrderLibrary and
PartitionPiles) keep their flat sequential native ``Action``s. Callers inspect
the active decision through ``env.decision()``, preview a single offer id or
grouped ``Choice`` on a native clone via ``env.preview(choice, at=...)``, and
commit atomically via ``env.submit(choice, at=...)`` without touching
``env.step(action)`` or the observation contract.
"""

import json
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from managym.experience import ObjectRenderId, ProtocolModel, SubjectRef

LibraryEnd = Literal["top", "bottom"]


class PlannedBlock(ProtocolModel):
    blocker: ObjectRenderId
    attacker: ObjectRenderId


class GroupCard(ProtocolModel):
    item: ObjectRenderId
    label: str


class AttackerItem(ProtocolModel):
    item: ObjectRenderId
    label: str
    must_attack: bool


class BlockerItem(ProtocolModel):
    item: ObjectRenderId
    label: str
    can_block: list[ObjectRenderId]
    must_block: bool


class DecisionAddress(ProtocolModel):
    """Optimistic-concurrency address binding a decision to `(viewer, revision, fingerprint)`."""

    viewer: int
    revision: int
    fingerprint: str


class Offer(ProtocolModel):
    """One raw single-step legal offer available at the current decision."""

    id: int
    verb: str
    label: str
    source: SubjectRef | None = None


class DecisionBase(ProtocolModel):
    address: DecisionAddress
    offers: list[Offer]


class SingleDecision(DecisionBase):
    kind: Literal["single"] = "single"


class DeclareAttackersDecision(DecisionBase):
    kind: Literal["declare_attackers"] = "declare_attackers"
    declarable: list[AttackerItem]
    committed: list[ObjectRenderId]


class DeclareBlockersDecision(DecisionBase):
    kind: Literal["declare_blockers"] = "declare_blockers"
    declarable: list[BlockerItem]
    committed: list[PlannedBlock]


class ScryDecision(DecisionBase):
    kind: Literal["scry"] = "scry"
    items: list[GroupCard]
    committed_top: list[GroupCard]
    committed_bottom: list[GroupCard]


class SurveilDecision(DecisionBase):
    kind: Literal["surveil"] = "surveil"
    items: list[GroupCard]
    committed_graveyard: list[GroupCard]


class OrderLibraryDecision(DecisionBase):
    kind: Literal["order_library"] = "order_library"
    end: LibraryEnd
    items: list[GroupCard]
    committed: list[GroupCard]


class PartitionPilesDecision(DecisionBase):
    kind: Literal["partition_piles"] = "partition_piles"
    items: list[GroupCard]
    committed_face_up: list[GroupCard]
    committed_face_down: list[GroupCard]


type Decision = Annotated[
    SingleDecision
    | DeclareAttackersDecision
    | DeclareBlockersDecision
    | ScryDecision
    | SurveilDecision
    | OrderLibraryDecision
    | PartitionPilesDecision,
    Field(discriminator="kind"),
]


class DeclareAttackersChoice(ProtocolModel):
    kind: Literal["declare_attackers"] = "declare_attackers"
    attackers: list[ObjectRenderId] = Field(default_factory=list)


class DeclareBlockersChoice(ProtocolModel):
    kind: Literal["declare_blockers"] = "declare_blockers"
    blocks: list[PlannedBlock] = Field(default_factory=list)


class ScryChoice(ProtocolModel):
    """`top` is drawn-first; `bottom` is top-to-bottom of the bottom group."""

    kind: Literal["scry"] = "scry"
    top: list[ObjectRenderId] = Field(default_factory=list)
    bottom: list[ObjectRenderId] = Field(default_factory=list)


class SurveilChoice(ProtocolModel):
    """Cards to put into the graveyard; all remaining looked-at cards stay on top."""

    kind: Literal["surveil"] = "surveil"
    graveyard: list[ObjectRenderId] = Field(default_factory=list)


class OrderLibraryChoice(ProtocolModel):
    """Remaining cards in desired order, drawn-first at index 0."""

    kind: Literal["order_library"] = "order_library"
    order: list[ObjectRenderId] = Field(default_factory=list)


class PartitionPilesChoice(ProtocolModel):
    """Cards to put in the face-up pile; all remaining looked-at cards go face-down."""

    kind: Literal["partition_piles"] = "partition_piles"
    face_up: list[ObjectRenderId] = Field(default_factory=list)


type GroupChoice = Annotated[
    DeclareAttackersChoice
    | DeclareBlockersChoice
    | ScryChoice
    | SurveilChoice
    | OrderLibraryChoice
    | PartitionPilesChoice,
    Field(discriminator="kind"),
]

type Choice = int | GroupChoice


class DecisionStep(ProtocolModel):
    """One sequential native step compiled from a decision choice."""

    revision: int
    fingerprint: str
    action_index: int
    item: ObjectRenderId | None


class DecisionRefusal(ProtocolModel):
    item: ObjectRenderId | None
    label: str
    reason: str


class SingleOutcome(ProtocolModel):
    kind: Literal["single"] = "single"
    offer_id: int


class DeclareAttackersOutcome(ProtocolModel):
    kind: Literal["declare_attackers"] = "declare_attackers"
    attackers: list[ObjectRenderId]


class DeclareBlockersOutcome(ProtocolModel):
    kind: Literal["declare_blockers"] = "declare_blockers"
    blocks: list[PlannedBlock]


class ScryOutcome(ProtocolModel):
    kind: Literal["scry"] = "scry"
    top: list[ObjectRenderId]
    bottom: list[ObjectRenderId]


class SurveilOutcome(ProtocolModel):
    kind: Literal["surveil"] = "surveil"
    graveyard: list[ObjectRenderId]
    kept: list[ObjectRenderId]


class OrderLibraryOutcome(ProtocolModel):
    kind: Literal["order_library"] = "order_library"
    end: LibraryEnd
    order: list[ObjectRenderId]


class PartitionPilesOutcome(ProtocolModel):
    kind: Literal["partition_piles"] = "partition_piles"
    face_up: list[ObjectRenderId]
    face_down: list[ObjectRenderId]


type DecisionOutcome = Annotated[
    SingleOutcome
    | DeclareAttackersOutcome
    | DeclareBlockersOutcome
    | ScryOutcome
    | SurveilOutcome
    | OrderLibraryOutcome
    | PartitionPilesOutcome,
    Field(discriminator="kind"),
]


class DecisionPreview(ProtocolModel):
    decision: Decision
    steps: list[DecisionStep]
    refused: DecisionRefusal | None
    complete: bool
    outcome: DecisionOutcome | None


DECISION_ADAPTER: TypeAdapter[Decision] = TypeAdapter(Decision)
GROUP_CHOICE_ADAPTER: TypeAdapter[GroupChoice] = TypeAdapter(GroupChoice)


def parse_decision(raw: str) -> Decision:
    return DECISION_ADAPTER.validate_json(raw)


def parse_preview(raw: str) -> DecisionPreview:
    return DecisionPreview.model_validate_json(raw)


def dump_choice(choice: Choice) -> str:
    if isinstance(choice, bool):
        raise TypeError("choice must be an int offer id or a grouped Choice, not bool")
    if isinstance(choice, int):
        if choice < 0:
            raise ValueError(f"offer id must be non-negative, got {choice}")
        return json.dumps({"kind": "single", "offer_id": choice})
    validated = GROUP_CHOICE_ADAPTER.validate_python(choice)
    return GROUP_CHOICE_ADAPTER.dump_json(validated).decode()


def dump_address(at: DecisionAddress) -> str:
    return DecisionAddress.model_validate(at).model_dump_json()
