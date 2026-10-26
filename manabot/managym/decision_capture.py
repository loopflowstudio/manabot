"""Portable text envelope around native-validated observations and continuation facts.

The native handle owns origin and submission. These strict read-only models admit
serialized evidence for pure rendering; they cannot reconstruct an origin handle.
Schema-6 Observation and Command retain their existing owners in decision.py.
Native deserialization validates viewer fields and continuation frames; Python
keeps only the portable presentation/content envelope.
"""

import json
from typing import Literal

from managym.decision import Observation
from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator

import managym


class CaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ManaCost(CaptureModel):
    cost: list[int]
    mana_value: int
    hybrid: list[list[int | str]] = []
    spend_on_lesson: bool = False


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
    kicker: ManaCost | None
    ward: ManaCost | None
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

    capture_version: Literal[2]
    world_version: str
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
    def parse_observation(cls, value: object) -> Observation:
        text = json.dumps(value)
        managym.validate_observation_json(text)
        return Observation.from_json(text)

    @field_validator("frames", mode="before")
    @classmethod
    def parse_frames(cls, value: object) -> list[managym.ContinuationFrame]:
        return managym.parse_continuation_frames_json(json.dumps(value))
