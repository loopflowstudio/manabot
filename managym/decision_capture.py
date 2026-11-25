"""Portable text envelope around native-validated observations and continuation facts.

The native handle owns origin and submission. These strict read-only models admit
serialized evidence for pure rendering; they cannot reconstruct an origin handle.
Schema-8 Observation and Command retain their existing owners in decision.py.
Native deserialization validates viewer fields and continuation frames; Python
keeps only the portable presentation/content envelope.
"""

import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    ValidationInfo,
    field_validator,
)

import managym
from managym.decision import Observation


class CaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ManaCost(CaptureModel):
    cost: list[int]
    mana_value: int
    hybrid: list[list[int | str]] = []
    spend_on_lesson: bool = False
    # Number of {X} symbols, omitted when zero; mana_value counts none of them.
    x: int = 0


class Keywords(CaptureModel):
    flying: bool
    reach: bool
    haste: bool
    flash: bool
    vigilance: bool
    trample: bool
    first_strike: bool
    double_strike: bool
    deathtouch: bool
    lifelink: bool
    defender: bool
    menace: bool
    hexproof: bool
    indestructible: bool


class DefinitionTypes(CaptureModel):
    types: list[str]


class CastOption(CaptureModel):
    """The words for a card's optional cast cost that is not kicker."""

    # A spell cast with the cost taken: "bargained".
    taken: str
    # The cost itself: "bargain".
    cost: str


class Definition(CaptureModel):
    name: str
    text_box: str
    mana_cost: ManaCost | None
    types: DefinitionTypes
    supertypes: list[str]
    subtypes: list[str]
    keywords: Keywords
    power: int | None
    toughness: int | None
    printed_loyalty: int | None = Field(default=None, exclude_if=lambda value: value is None)
    kicker: ManaCost | None
    ward: ManaCost | None
    # The life a ward costs ("Ward—Pay 7 life."); absent for any other ward.
    ward_life: int | None = Field(default=None, exclude_if=lambda value: value is None)
    # The cost a `kicked` flag on this card's spell stands for when it is not
    # kicker (bargain, an alternative cost); absent otherwise.
    cast_option: CastOption | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    is_token: bool


class ObjectRenderId(CaptureModel):
    entity: int
    incarnation: int


class Subject(CaptureModel):
    kind: Literal["object", "player"]
    id: ObjectRenderId | int


class DecisionObject(CaptureModel):
    entity_id: int
    subject: Subject
    display_id: int
    label: str


class DecisionTarget(CaptureModel):
    clause: int
    object: DecisionObject


class Paid(CaptureModel):
    """One object's payment toward the pending cost, and what remains of it."""

    phrase: str
    object: DecisionObject | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class Continuation(CaptureModel):
    recent_facts: list[Literal["unsupported"]]
    actor: int
    stage: str
    instruction: str
    source: DecisionObject | None
    targets: list[DecisionTarget]
    reserved_sacrifice: DecisionObject | None
    mana_cost: str | None
    chosen_mode: int | None
    pending: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    # `source` is a spell or activated ability being announced: on the stack by
    # the rules (CR 601.2a, 602.2a) while its costs are paid. Omitted otherwise.
    announcing: Literal["spell", "activated_ability"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # What objects have already paid toward the cost being paid, in paid order.
    paid: list[Paid] = Field(default_factory=list, exclude_if=lambda value: not value)


class ObjectLink(CaptureModel):
    permanent_id: int | None
    incarnation: int


class UnavailableReference(CaptureModel):
    kind: Literal["unavailable"]


class PlayerReference(CaptureModel):
    kind: Literal["player"]
    seat: int


class CardReference(CaptureModel):
    kind: Literal["card"]
    definition: int
    owner: int
    zone: int | None
    context_id: int | None


HistoryReference = UnavailableReference | PlayerReference | CardReference


class HistoryRow(CaptureModel):
    kind: int
    amount: int
    source: HistoryReference
    target: HistoryReference


class PublicHistory(CaptureModel):
    version: Literal[1]
    capacity: int
    rows: list[HistoryRow]


class DecisionInput(CaptureModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    capture_version: Literal[3, 4, 5, 6]
    # Captures through v5 carry a world label inside their digest; nothing
    # reads it, and v6 has none.
    world_version: str | None = None
    content_digest: str
    observation: SkipValidation[Observation]
    frames: SkipValidation[list[managym.ContinuationFrame]]
    continuation: Continuation
    definitions: dict[str, Definition]
    objects: dict[str, ObjectLink]
    history: PublicHistory
    digest: str

    @field_validator("observation", mode="before")
    @classmethod
    def parse_observation(cls, value: object, info: ValidationInfo) -> Observation:
        text = json.dumps(value)
        # Capture v3/v4 retain schema 7 as historical evidence. This read-only
        # envelope never creates a native submission handle or current binding.
        version = 7 if info.data.get("capture_version") in (3, 4) else 8
        managym.validate_observation_json(text, schema_version=version)
        return Observation.from_json(text, _expected_version=version)

    @field_validator("frames", mode="before")
    @classmethod
    def parse_frames(
        cls, value: object, info: ValidationInfo
    ) -> list[managym.ContinuationFrame]:
        frames = managym.parse_continuation_frames_json(json.dumps(value))
        if info.data.get("capture_version") == 3 and any(
            target.legal_at_resolution_start is not None
            for frame in frames
            for target in frame.targets
        ):
            raise ValueError("capture v3 has no retained eligibility contract")
        return frames
