"""Validated Python mirror of atomic native decision capture v1.

The native handle owns origin and submission. These strict read-only models admit
serialized evidence for pure rendering; they cannot reconstruct an origin handle.
Schema-6 Observation and Command retain their existing owners in decision.py.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


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


class CardTypes(CaptureModel):
    is_castable: bool
    is_permanent: bool
    is_non_land_permanent: bool
    is_non_creature_permanent: bool
    is_spell: bool
    is_creature: bool
    is_land: bool
    is_planeswalker: bool
    is_enchantment: bool
    is_artifact: bool
    is_kindred: bool
    is_battle: bool


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


class Card(CaptureModel):
    id: int
    registry_key: int
    name: str
    text_box: str
    zone: int
    owner_id: int
    power: int
    toughness: int
    is_token: bool
    is_legendary: bool
    is_ally: bool
    is_lesson: bool
    ward_cost: int
    kicker_cost: int
    card_types: CardTypes
    keywords: Keywords
    mana_cost: ManaCost


class SideboardCard(CaptureModel):
    candidate_id: int
    owner_id: int
    registry_key: int
    name: str
    text_box: str


class Permanent(CaptureModel):
    id: int
    registry_key: int
    controller_id: int
    tapped: bool
    damage: int
    is_summoning_sick: bool
    plus1_counters: int
    cant_be_blocked_this_turn: bool
    power: int
    toughness: int
    is_animated: bool
    has_exile_link: bool
    exile_if_dies_this_turn: bool
    temp_subtypes: list[str]
    keywords: Keywords


class Player(CaptureModel):
    player_index: int
    id: int
    is_agent: bool
    is_active: bool
    life: int
    zone_counts: list[int]
    graveyard_lessons: int
    combat_mana: int
    mana: list[int]
    lesson_mana: list[int]
    known_hand: dict[str, int]
    sideboard_counts: dict[str, int]
    remaining_sideboard_counts: dict[str, int]


class Turn(CaptureModel):
    turn_number: int
    phase: int
    step: int
    active_player_id: int
    agent_player_id: int


class Action(CaptureModel):
    type: int
    focus: list[int]
    declared: bool | None
    mode_index: int | None
    ability_index: int | None
    description: str | None = None


class ActionSpace(CaptureModel):
    type: int
    actions: list[Action]


class StackTarget(CaptureModel):
    kind: int
    player_id: int | None
    permanent_id: int | None
    stack_object_id: int | None
    card_id: int | None


class StackObject(CaptureModel):
    stack_object_id: int
    kind: int
    controller_id: int
    source_card_registry_key: int
    source_permanent_id: int | None
    ability_index: int | None
    chosen_mode: int | None
    targets: list[StackTarget]


class ViewerState(CaptureModel):
    game_over: bool
    won: bool
    turn: Turn
    action_space: ActionSpace
    agent: Player
    opponent: Player
    agent_cards: list[Card]
    opponent_cards: list[Card]
    agent_sideboard: list[SideboardCard]
    agent_permanents: list[Permanent]
    opponent_permanents: list[Permanent]
    stack_objects: list[StackObject]
    # Schema-6 composite observations have no transition increments. A future
    # nonempty increment needs a versioned renderer, not silent omission.
    recent_events: list[Literal["unsupported"]]


class Identity(CaptureModel):
    schema_version: Literal[6]
    revision: int
    viewer: int
    viewer_state_hash: str


class ObjectCandidate(CaptureModel):
    decision_fingerprint: str
    offer_id: int
    role: int
    candidate_id: int


class Commitment(CaptureModel):
    kind: str
    card: str | None = None


class Offer(CaptureModel):
    id: int
    actor: int
    verb: str
    label: str
    source: None
    help: None
    confirm_label: str
    choices: list[Literal["unsupported"]]
    public_commitment: Commitment | None = None


class Frame(CaptureModel):
    schema_version: Literal[6]
    revision: int
    actor: int
    fingerprint: str
    offers: list[Offer]
    object_candidates: list[ObjectCandidate]


class Observation(CaptureModel):
    identity: Identity
    viewer_state: ViewerState
    decision: Frame
    events: list[Literal["unsupported"]]


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
    capture_version: Literal[1]
    world_version: str
    content_digest: str
    observation: Observation
    continuation: Continuation
    definitions: dict[str, Definition]
    objects: dict[str, ObjectLink]
    history: PublicHistory
    digest: str
