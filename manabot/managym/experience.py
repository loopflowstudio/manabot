"""Explicit Python representation of the experience protocol-v1 wire boundary.

Rust owns the protocol types (``managym/src/experience.rs``) and generates the
canonical JSON Schema in ``protocol/experience-v1.schema.json``. These Pydantic
models give Python an equally explicit runtime representation and are checked
against that generated schema in tests. Academy speaks them over its transport;
manabot records protocol-v1 frames in its replay and teacher evidence, so the
models live here, below both packages.
"""

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION: Literal[1] = 1
# Frames whose content predates versioned content identity carry this content
# hash. Live play and recorded evidence both depend on the exact string.
UNVERSIONED_CONTENT_HASH = "legacy-content-unversioned"

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


class DecisionObjectView(ProtocolModel):
    entity_id: UInt32
    subject: SubjectRef
    display_id: UInt32
    label: str


class DecisionTargetView(ProtocolModel):
    clause: UInt32
    object: DecisionObjectView


class DecisionContextView(ProtocolModel):
    recent_facts: list[str]
    actor: UInt8
    stage: str
    instruction: str
    source: DecisionObjectView | None
    targets: list[DecisionTargetView]
    reserved_sacrifice: DecisionObjectView | None
    mana_cost: str | None
    chosen_mode: UInt32 | None


class OfferSubjectView(ProtocolModel):
    role: str
    object: DecisionObjectView


class InteractionOffer(ProtocolModel):
    subjects: list[OfferSubjectView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
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


class LegacyCastPermissionView(ProtocolModel):
    player_id: UInt32
    cost: str | None = Field(default=None, exclude_if=lambda value: value is None)
    this_turn: bool = Field(default=False, exclude_if=lambda value: not value)
    own_turn_only: bool = Field(default=False, exclude_if=lambda value: not value)
    after_turn: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_permanent_id: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    flash: bool = Field(default=False, exclude_if=lambda value: not value)
    any_mana: bool = Field(default=False, exclude_if=lambda value: not value)
    waterbend: bool = Field(default=False, exclude_if=lambda value: not value)
    foretold: bool = Field(default=False, exclude_if=lambda value: not value)


class LegacyRestrictedManaView(ProtocolModel):
    spells: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    abilities: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    mana: tuple[Int32, Int32, Int32, Int32, Int32, Int32]


class LegacyCardView(ProtocolModel):
    text_box: str = Field(default="", exclude_if=lambda value: not value)
    id: UInt32
    registry_key: UInt32
    name: str
    zone: str
    owner_id: UInt32
    power: Int32
    toughness: Int32
    mana_value: Int32
    types: LegacyCardTypesView
    face_down: bool = Field(default=False, exclude_if=lambda value: not value)
    cast_permission: LegacyCastPermissionView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    x: Int32 | None = Field(default=None, exclude_if=lambda value: value is None)


class ContinuousEffectView(ProtocolModel):
    """One continuous-effect record on a permanent or player (CR 611, 613)."""

    modification: str
    power: Int32 | None = Field(default=None, exclude_if=lambda value: value is None)
    toughness: Int32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    amount: Int32 | None = Field(default=None, exclude_if=lambda value: value is None)
    keywords: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    subtypes: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    color: str | None = Field(default=None, exclude_if=lambda value: value is None)
    duration: str
    source: str | None = Field(default=None, exclude_if=lambda value: value is None)
    timestamp: UInt32


class PlayerTurnHistoryView(ProtocolModel):
    """What one player has done this turn."""

    permanents_sacrificed: UInt32 = Field(
        default=0, exclude_if=lambda value: not value
    )


class CopyView(ProtocolModel):
    """A copy effect: the printed card it copies and its exceptions (CR 707.9b)."""

    name: str
    exceptions: list[ContinuousEffectView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )


class LegacyPermanentView(ProtocolModel):
    types: LegacyCardTypesView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    is_token: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    temp_subtypes: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    exile_if_dies_this_turn: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    exhausted_abilities: UInt32 = Field(default=0, exclude_if=lambda value: not value)
    once_each_turn_used: UInt32 = Field(default=0, exclude_if=lambda value: not value)
    granted_firebending: UInt32 = Field(default=0, exclude_if=lambda value: not value)
    text_box: str = Field(default="", exclude_if=lambda value: not value)
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
    counters: dict[str, UInt32] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    attached_to: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    creature_types: list[str] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    is_granted_land: bool = Field(default=False, exclude_if=lambda value: not value)
    attacking: bool = Field(default=False, exclude_if=lambda value: not value)
    blocking: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    abilities_lost: bool = Field(default=False, exclude_if=lambda value: not value)
    used_abilities: list[UInt32] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    effects: list[ContinuousEffectView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    copy_of: CopyView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class SideboardCardView(ProtocolModel):
    text_box: str = Field(default="", exclude_if=lambda value: not value)
    candidate_id: UInt32
    owner_id: UInt32
    registry_key: UInt32
    name: str


class LegacyPlayerView(ProtocolModel):
    mana: tuple[Int32, Int32, Int32, Int32, Int32, Int32] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    lesson_mana: tuple[Int32, Int32, Int32, Int32, Int32, Int32] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    restricted_mana: list[LegacyRestrictedManaView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    hand_only_until: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    no_maximum_hand_size: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    combat_mana: Int32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    inspected: list[LegacyCardView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    player_index: UInt8
    id: UInt32
    is_active: bool
    is_agent: bool
    life: Int32
    zone_counts: dict[str, UInt32]
    library_count: UInt32
    known_hand: dict[str, UInt32] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    sideboard_counts: dict[str, UInt32] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    remaining_sideboard_counts: dict[str, UInt32] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    sideboard: list[SideboardCardView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    hand_hidden_count: UInt32 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    keeps_red_mana: bool = Field(default=False, exclude_if=lambda value: not value)
    turn_history: PlayerTurnHistoryView = Field(
        default_factory=PlayerTurnHistoryView,
        exclude_if=lambda value: value == PlayerTurnHistoryView(),
    )
    no_maximum_hand_size: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    effects: list[ContinuousEffectView] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
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


class StackEntryView(ProtocolModel):
    id: UInt64
    kind: str
    controller: UInt8
    label: str
    source_permanent: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    ability_index: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    chosen_mode: UInt32 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    x: UInt32 | None = Field(default=None, exclude_if=lambda value: value is None)


class LegacyHeroObservation(ProtocolModel):
    stack_objects: list[StackEntryView] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    decision_context: DecisionContextView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    definition_names: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    game_over: bool
    won: bool
    turn: LegacyTurnView
    agent: LegacyPlayerView
    opponent: LegacyPlayerView

    @model_validator(mode="after")
    def validate_context_owner(self) -> "LegacyHeroObservation":
        if (
            self.decision_context is not None
            and self.decision_context.actor != self.agent.player_index
            and (
                self.decision_context.source is not None
                or self.decision_context.targets
                or self.decision_context.reserved_sacrifice is not None
                or self.decision_context.mana_cost is not None
                or self.decision_context.chosen_mode is not None
            )
        ):
            raise ValueError("decision context is not oriented to its acting viewer")
        return self


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
    deck_names: DeckNames | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    asset_pack: AssetPackReference | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    log: list[str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    auto_passed: UInt32 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


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


class AttackGroupPresentation(ProtocolModel):
    kind: Literal["attack_group"]
    attackers: list[ObjectRenderId]
    defender: SubjectRef


class BlockedPresentation(ProtocolModel):
    kind: Literal["blocked"]
    attacker: ObjectRenderId
    blockers: list[ObjectRenderId]


class TurnStartedPresentation(ProtocolModel):
    kind: Literal["turn_started"]
    player: UInt8
    turn_number: UInt32


PresentationKind: TypeAlias = Annotated[
    CastPresentation
    | TargetedPresentation
    | ResolvedPresentation
    | DamagePresentation
    | DestroyedPresentation
    | DiedPresentation
    | AttackGroupPresentation
    | BlockedPresentation
    | TurnStartedPresentation,
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
    presentation_cursor: UInt64
    presentation_tail: list[PresentationEvent]
    accepted_commands: list[CommandReceipt]
    replay_cursor: UInt64
    checkpoint: str | None


class ProtocolV1ConformanceBundle(ProtocolModel):
    recovery: RecoveryEnvelope
    command: Command
