"""Explicit Python representation of the experience protocol-v1 wire boundary.

Rust owns the canonical JSON Schema in ``protocol/experience-v1.schema.json``.
These Pydantic models give the Python adapter an equally explicit runtime
representation and are checked against that generated schema in tests.
"""

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION: Literal[1] = 1

UInt8: TypeAlias = Annotated[int, Field(ge=0, le=2**8 - 1)]
UInt16: TypeAlias = Annotated[int, Field(ge=0, le=2**16 - 1)]
UInt32: TypeAlias = Annotated[int, Field(ge=0, le=2**32 - 1)]
UInt64: TypeAlias = Annotated[int, Field(ge=0, le=2**64 - 1)]
Int32: TypeAlias = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]


class ProtocolModel(BaseModel):
    """Closed wire object: unknown fields are protocol drift, not sidecars."""

    model_config = ConfigDict(extra="forbid")


class ObjectRenderId(ProtocolModel):
    entity: UInt32
    incarnation: UInt32


class ObjectSubject(ProtocolModel):
    kind: Literal["object"]
    id: ObjectRenderId


class StackSubject(ProtocolModel):
    kind: Literal["stack"]
    id: UInt64


class PlayerSubject(ProtocolModel):
    kind: Literal["player"]
    id: UInt8


SubjectRef: TypeAlias = Annotated[
    ObjectSubject | StackSubject | PlayerSubject,
    Field(discriminator="kind"),
]


class OfferVerb(str, Enum):
    CAST = "cast"
    PLAY_LAND = "play_land"
    ACTIVATE = "activate"
    PASS_PRIORITY = "pass_priority"
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    CHOOSE = "choose"
    PAY = "pay"
    SPECIAL = "special"


class SubjectCandidateValue(ProtocolModel):
    kind: Literal["subject"]
    subject: SubjectRef


class ModeCandidateValue(ProtocolModel):
    kind: Literal["mode"]
    key: str


class PaymentPlanCandidateValue(ProtocolModel):
    kind: Literal["payment_plan"]
    id: str


class BooleanCandidateValue(ProtocolModel):
    kind: Literal["boolean"]
    value: bool


CandidateValue: TypeAlias = Annotated[
    SubjectCandidateValue
    | ModeCandidateValue
    | PaymentPlanCandidateValue
    | BooleanCandidateValue,
    Field(discriminator="kind"),
]


class Candidate(ProtocolModel):
    id: UInt32
    value: CandidateValue
    label: str
    help: str | None
    preview: str | None


class CandidateSource(ProtocolModel):
    id: UInt32
    depends_on: list[UInt16]
    initial: list[Candidate] | None


class SelectChoiceStep(ProtocolModel):
    kind: Literal["select"]
    role: UInt16
    label: str
    candidates: CandidateSource
    min: UInt16
    max: UInt16
    ordered: bool
    distinct: bool


class NumberChoiceStep(ProtocolModel):
    kind: Literal["number"]
    role: UInt16
    label: str
    min: Int32
    max: Int32


class AssignChoiceStep(ProtocolModel):
    kind: Literal["assign"]
    role: UInt16
    label: str
    sources: CandidateSource
    destinations: CandidateSource
    min_per_source: UInt16
    max_per_source: UInt16


class OrderChoiceStep(ProtocolModel):
    kind: Literal["order"]
    role: UInt16
    label: str
    candidates: CandidateSource


class PaymentChoiceStep(ProtocolModel):
    kind: Literal["payment"]
    role: UInt16
    label: str
    plans: CandidateSource
    allow_auto: bool


ChoiceStep: TypeAlias = Annotated[
    SelectChoiceStep
    | NumberChoiceStep
    | AssignChoiceStep
    | OrderChoiceStep
    | PaymentChoiceStep,
    Field(discriminator="kind"),
]


class InteractionOffer(ProtocolModel):
    id: UInt32
    actor: UInt8
    verb: OfferVerb
    source: SubjectRef | None
    label: str
    help: str | None
    choices: list[ChoiceStep]
    confirm_label: str
    action_type: str
    focus: list[UInt32]


class CandidatesChoiceAnswer(ProtocolModel):
    kind: Literal["candidates"]
    role: UInt16
    candidates: list[UInt32]


class NumberChoiceAnswer(ProtocolModel):
    kind: Literal["number"]
    role: UInt16
    value: Int32


class AssignmentsChoiceAnswer(ProtocolModel):
    kind: Literal["assignments"]
    role: UInt16
    pairs: list[tuple[UInt32, UInt32]]


class OrderChoiceAnswer(ProtocolModel):
    kind: Literal["order"]
    role: UInt16
    candidates: list[UInt32]


class PaymentChoiceAnswer(ProtocolModel):
    kind: Literal["payment"]
    role: UInt16
    plan: str


ChoiceAnswer: TypeAlias = Annotated[
    CandidatesChoiceAnswer
    | NumberChoiceAnswer
    | AssignmentsChoiceAnswer
    | OrderChoiceAnswer
    | PaymentChoiceAnswer,
    Field(discriminator="kind"),
]


class Command(ProtocolModel):
    command_id: str
    match_id: str
    expected_revision: UInt64
    prompt_id: UInt64
    offer_id: UInt32
    answers: list[ChoiceAnswer]


class PromptView(ProtocolModel):
    id: UInt64
    actor: UInt8
    kind: str
    title: str
    instruction: str


class AuthorityStatus(str, Enum):
    READY = "ready"
    THINKING = "thinking"
    RESOLVING = "resolving"
    RECONNECTING = "reconnecting"
    GAME_OVER = "game_over"


class LegacyCardTypesView(ProtocolModel):
    is_creature: bool
    is_land: bool
    is_spell: bool
    is_artifact: bool
    is_enchantment: bool
    is_planeswalker: bool
    is_battle: bool


class LegacyCardView(ProtocolModel):
    id: UInt32
    registry_key: UInt32
    name: str
    zone: str
    owner_id: UInt32
    power: Int32
    toughness: Int32
    mana_value: Int32
    types: LegacyCardTypesView


class LegacyPermanentView(ProtocolModel):
    id: UInt32
    name: str | None
    controller_id: UInt32
    tapped: bool
    damage: Int32
    summoning_sick: bool
    power: Int32 | None
    toughness: Int32 | None
    base_power: Int32 | None
    base_toughness: Int32 | None
    plus1_counters: UInt32


class LegacyPlayerView(ProtocolModel):
    player_index: UInt8
    id: UInt32
    is_active: bool
    is_agent: bool
    life: Int32
    zone_counts: dict[str, UInt32]
    library_count: UInt32
    hand_hidden_count: UInt32 | None = None
    hand: list[LegacyCardView]
    graveyard: list[LegacyCardView]
    exile: list[LegacyCardView]
    stack: list[LegacyCardView]
    battlefield: list[LegacyPermanentView]


class LegacyTurnView(ProtocolModel):
    turn_number: UInt32
    phase: str
    step: str
    active_player_id: UInt32
    agent_player_id: UInt32


class LegacyHeroObservation(ProtocolModel):
    game_over: bool
    won: bool
    turn: LegacyTurnView
    agent: LegacyPlayerView
    opponent: LegacyPlayerView


class StopsConfig(ProtocolModel):
    my: list[str]
    opponent: list[str]
    stop_on_stack: bool
    auto_pass: bool


class DeckNames(ProtocolModel):
    hero: str
    villain: str


class AssetPackReference(ProtocolModel):
    id: str
    version: str
    manifest_sha256: str


class ExperienceFrame(ProtocolModel):
    protocol: Literal[1]
    match_id: str
    revision: UInt64
    frame_hash: str
    content_hash: str
    asset_manifest_hash: str
    status: AuthorityStatus
    prompt: PromptView | None
    projection: LegacyHeroObservation
    offers: list[InteractionOffer]
    winner: UInt8 | None
    action_space: str
    stops: StopsConfig
    deck_names: DeckNames | None = None
    asset_pack: AssetPackReference | None = None
    log: list[str] | None = None
    auto_passed: UInt32 | None = None


class PresentationImportance(str, Enum):
    AMBIENT = "ambient"
    NORMAL = "normal"
    EMPHASIZED = "emphasized"
    CRITICAL = "critical"


class CastPresentation(ProtocolModel):
    kind: Literal["cast"]
    object: ObjectRenderId
    controller: UInt8
    stack: UInt64


class TargetedPresentation(ProtocolModel):
    kind: Literal["targeted"]
    source: SubjectRef
    target: SubjectRef


class ResolvedPresentation(ProtocolModel):
    kind: Literal["resolved"]
    stack: UInt64


class DamagePresentation(ProtocolModel):
    kind: Literal["damage"]
    source: SubjectRef | None
    target: SubjectRef
    amount: Int32


class DestroyedPresentation(ProtocolModel):
    kind: Literal["destroyed"]
    objects: list[ObjectRenderId]


class DiedPresentation(ProtocolModel):
    kind: Literal["died"]
    objects: list[ObjectRenderId]


PresentationKind: TypeAlias = Annotated[
    CastPresentation
    | TargetedPresentation
    | ResolvedPresentation
    | DamagePresentation
    | DestroyedPresentation
    | DiedPresentation,
    Field(discriminator="kind"),
]


class PresentationEvent(ProtocolModel):
    seq: UInt64
    from_revision: UInt64
    to_revision: UInt64
    caused_by: str | None
    group: UInt64
    importance: PresentationImportance
    suggested_ms: UInt32
    sound: str | None
    kind: PresentationKind


class CommandReceipt(ProtocolModel):
    command_id: str
    actor: UInt8
    accepted_at: UInt64
    resulting_revision: UInt64
    resulting_frame_hash: str


class RecoveryReason(str, Enum):
    INITIAL_CONNECT = "initial_connect"
    EXPLICIT_RESYNC = "explicit_resync"
    REVISION_GAP = "revision_gap"
    RECONNECT = "reconnect"
    DUPLICATE_COMMAND = "duplicate_command"
    STALE_COMMAND = "stale_command"
    AUTHORITY_RESTART = "authority_restart"


class RecoveryEnvelope(ProtocolModel):
    protocol: Literal[1]
    engine_version: str
    content_hash: str
    asset_manifest_hash: str
    reason: RecoveryReason
    frame: ExperienceFrame
    presentation_tail: list[PresentationEvent]
    accepted_commands: list[CommandReceipt]
    replay_cursor: UInt64
    checkpoint: str | None


class ProtocolV1ConformanceBundle(ProtocolModel):
    recovery: RecoveryEnvelope
    command: Command
