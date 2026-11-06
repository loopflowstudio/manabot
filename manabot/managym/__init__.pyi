from __future__ import annotations

from enum import IntEnum
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

class PoolDefinition(TypedDict):
    card_def_id: int
    key: str | None
    scryfall_id: str | None
    registry_name: str
    eligibility: Literal["eligible", "unsupported"]
    reason: str | None

class ContentPool(TypedDict):
    world_version: str
    content_digest: str
    definitions: list[PoolDefinition]

def content_pool(name: str) -> ContentPool:
    """The world pool (``"world"``, the only pool): every catalog card and its support."""

class WorldIdentity(TypedDict):
    world_version: str
    card_digest: str

def world_identity() -> WorldIdentity:
    """The installed world's version and the digest of its cards."""

def set_cards(sets: list[str]) -> list[PoolDefinition]:
    """World pool rows for cards whose key prefix is in ``sets``."""

def unsupported_cards(names: list[str]) -> list[str]:
    """Distinct names the current world does not support for deck entry."""

class LearningCatalog(TypedDict):
    content_digest: str
    registry_names: list[str]
    token_kind: list[int]
    token_value: list[int]
    definition_offsets: list[int]
    program_offsets: list[int]
    definition_program_offsets: list[int]
    definition_program_rows: list[int]
    definition_ref_source_tokens: list[int]
    definition_ref_target_rows: list[int]

def learning_catalog() -> LearningCatalog:
    """Learning tokens for every catalog definition, rows by CardDefId."""

def card_definitions_json() -> str:
    """Every catalog definition as the JSON its learning tokens serialize."""

def learning_token_value(name: str) -> int:
    """The learning catalog's stable token value for a Rust or printed name."""

def learning_record_names() -> list[tuple[str, str, int, int | None, int | None]]:
    """Every name fact/shared-effect encoders hash, as `(namespace, name,
    value, token_kind, token_value)`; the token is card text's for the same
    concept. Namespaces: `token`, `name`, `restriction`."""

class PreparedPossibleWorldMaterializer:
    @property
    def viewer(self) -> int: ...
    @property
    def space_identity(self) -> str: ...
    @property
    def support_size(self) -> int: ...
    @property
    def max_batch_size(self) -> int: ...
    @property
    def construction_count(self) -> int: ...
    def diagnostics_json(self) -> str: ...
    def materialize_indexes(
        self,
        world_indexes: List[int],
        seeds: List[int],
        refresh_opponent_commitment: bool = False,
    ) -> List[Env]: ...

class DecisionCapture:
    def to_json(self) -> str: ...

class Env:
    def __init__(
        self,
        seed: int = 0,
        skip_trivial: bool = True,
        enable_profiler: bool = False,
        enable_behavior_tracking: bool = False,
    ) -> None: ...
    def reset(
        self, player_configs: List[PlayerConfig]
    ) -> Tuple[Observation, Dict[str, Any]]: ...
    def enable_card_capture(self) -> None: ...
    def card_capture_json(self) -> str: ...
    def set_seed(self, seed: int) -> None: ...
    def step(
        self, action: int
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]: ...
    def compound_offers(self) -> StructuredOfferSet: ...
    def compound_commands_json(
        self, offers: StructuredOfferSet, submission_json: str
    ) -> str: ...
    def structured_offers(self) -> StructuredOfferSet: ...
    def structured_search_offers(self) -> StructuredOfferSet: ...
    def search_context_json(self, include_witness: bool = True) -> str: ...
    def search_witness_json(self) -> str: ...
    def step_structured(
        self, offers: StructuredOfferSet, submission_json: str
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], int]: ...
    def step_legacy_submission(
        self, offers: StructuredOfferSet, submission_json: str
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], int]: ...
    # Shared semantic decision contract (managym.decision). JSON mirrors of
    # DecisionFrame / composite Observation / {receipt, observation}.
    def semantic_decision_frame_json(self) -> str: ...
    def player_controls_json(self) -> str: ...
    def public_schedule_json(self) -> str: ...
    def decision_context_json(self, viewer: int) -> str: ...
    def decision_offer_subjects_json(self, viewer: int) -> str: ...
    def presentation_subjects_json(self) -> str: ...
    def semantic_observation_json(self, viewer: int) -> str: ...
    def capture_decision(
        self, viewer: int, history_capacity: int = 0
    ) -> DecisionCapture: ...
    def execute_captured_command_json(
        self, capture: DecisionCapture, command_json: str
    ) -> str: ...
    def observation_for_player(self, player_index: int) -> Observation: ...
    def execute_semantic_command_json(self, command_json: str) -> str: ...
    def hidden_hand_constraints_json(self, viewer: int) -> str: ...
    def possible_world_space_json(self, viewer: int) -> str: ...
    def prepare_possible_world_materializer(
        self,
        viewer: int,
        expected_space_identity: str,
        max_batch_size: int,
    ) -> PreparedPossibleWorldMaterializer: ...
    def possible_world_support_json(
        self, viewer: int, space_identity: str, query_json: str
    ) -> str: ...
    def possible_world_space_construction_count(self) -> int: ...
    def materialize_sampled_hand(
        self,
        viewer: int,
        constraints_json: str,
        hand: dict[str, int],
        seed: int,
    ) -> Env: ...
    def materialize_possible_world(
        self,
        viewer: int,
        space_identity: str,
        world_index: int,
        seed: int,
        refresh_opponent_commitment: bool = False,
    ) -> "Env": ...
    def state_digest(self) -> str: ...
    def info(self) -> Dict[str, Any]: ...
    def content_pack_manifest(self) -> Dict[str, Any]: ...
    def skip_trivial_count(self) -> int: ...
    def export_profile_baseline(self) -> str: ...
    def compare_profile(self, baseline: str) -> str: ...
    def clone_env(self) -> "Env": ...
    def current_agent_index(self) -> Optional[int]: ...
    def is_game_over(self) -> bool: ...
    def winner_index(self) -> Optional[int]: ...
    def action_count(self) -> int: ...
    # Scenario / state-injection helpers — test & measurement harnesses only
    # (manabot/verify/competency.py). Bypass the rules engine: no events,
    # no triggers, no costs. Inject at a priority decision, then call
    # scenario_refresh() once.
    def scenario_naming_vocabulary(self) -> list[NameCandidate]:
        """Republish a pending name decision with the checked-in test vocabulary."""
        ...
    def scenario_set_life(self, player: int, life: int) -> None: ...
    def scenario_clear_hand(self, player: int) -> None: ...
    def scenario_force_card_in_hand(self, player: int, name: str) -> None: ...
    def scenario_force_battlefield(
        self, player: int, name: str, ready: bool = True
    ) -> int: ...
    def scenario_refresh(self) -> Observation: ...
    def determinize(self, seed: int, perspective: Optional[int] = None) -> None: ...
    def random_playout(self, seed: int, max_steps: int = 2000) -> Optional[int]: ...
    def reseed_rollout(self, seed: int) -> None: ...
    def random_action_index(self) -> int: ...
    def flat_mc_scores(
        self, worlds: int, rollouts: int, seed: int, max_steps: int = 2000
    ) -> Tuple[List[float], int, int]: ...
    def flat_mc_scores_for_worlds(
        self,
        viewer: int,
        space_identity: str,
        world_indexes: List[int],
        world_seeds: List[int],
        rollouts: int,
        max_steps: int = 2000,
    ) -> Tuple[List[float], int, int]: ...
    def encode_observation(
        self,
        obs: Observation,
        policy_input_version: int = 6,
        max_facts: int | None = None,
    ) -> Dict[str, np.ndarray]: ...
    def encode_observation_into(
        self,
        obs: Observation,
        out: Dict[str, np.ndarray],
        action_overflow: str = "error",
    ) -> None: ...

class StructuredOfferSet:
    def projection_json(self) -> str: ...

class SelectedBranchRuntime:
    def __init__(self, match_id: str, audit: bool = False) -> None: ...
    @property
    def driver_id(self) -> str: ...
    def fork_exact(self, source: Env, site: str) -> Env: ...
    def determinize(self, branch: Env, seed: int, perspective: int) -> None: ...
    def reseed_rollout(self, branch: Env, seed: int) -> None: ...
    def sample_policy_index(self, branch: Env) -> int: ...
    def apply_policy_choice(
        self,
        branch: Env,
        site: str,
        policy_index: int,
        preconditions_json: Optional[str] = None,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any], str]: ...
    def record_leaf_playout(self, hit_cap: bool = False) -> None: ...
    def snapshot_json(self) -> str: ...

class VectorEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        skip_trivial: bool = True,
        opponent_policy: str = "none",
        seed_stride: int | None = None,
    ) -> None: ...
    def reset_all(
        self, player_configs: List[PlayerConfig]
    ) -> List[Tuple[Observation, Dict[str, Any]]]: ...
    def step(
        self, actions: List[int]
    ) -> List[Tuple[Observation, float, bool, bool, Dict[str, Any]]]: ...
    def set_buffers(
        self, buffers: Dict[str, np.ndarray], action_overflow: str = "error"
    ) -> None: ...
    def scenario_naming_vocabulary_into_buffers(self) -> None: ...
    def reset_all_into_buffers(self, player_configs: List[PlayerConfig]) -> None: ...
    def reset_each_into_buffers(
        self, player_configs: list[list[PlayerConfig]]
    ) -> None: ...
    def step_into_buffers(
        self, actions: list[int], active: list[bool] | None = None
    ) -> None: ...
    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool], recover_capacity: bool = False
    ) -> None: ...
    def replace_failed_into_buffers(self, failed: list[bool]) -> None: ...
    def semantic_decision_frames_json(
        self, active: list[bool] | None = None
    ) -> list[str | None]: ...
    def current_game_seeds(self) -> list[int]: ...
    def current_action_summaries(self) -> list[tuple[int, str]]: ...
    def set_training_decks(
        self, decks: List[PlayerConfig], distinct: bool = True
    ) -> None: ...
    def current_player_configs(self) -> list[list[PlayerConfig]]: ...
    def current_agent_indices(self) -> list[int]: ...
    def reset_seeds_into_buffers(self, seeds: list[int]) -> None: ...
    def skip_trivial_counts(self) -> List[int]: ...
    def get_last_info(self) -> List[Dict[str, Any]]: ...

class PlayerConfig:
    """A seat's setup; ``decklist``/``sideboard`` read ``deck`` under receipt names."""

    def __init__(
        self,
        name: str,
        decklist: Dict[str, int],
        sideboard: Optional[Dict[str, int]] = None,
        world: str | None = None,
        disclosure: Literal["open", "closed"] = "open",
    ) -> None: ...
    name: str
    world: str | None
    """``WORLD_SETUP`` submits the deck to the world (reset admits it); ``None``
    runs it as given. Any other value raises: content packs are retired."""
    # Whether the opponent knows this player's main deck and sideboard.
    disclosure: Literal["open", "closed"]
    @property
    def deck(self) -> Deck: ...
    @property
    def decklist(self) -> Dict[str, int]: ...
    @property
    def sideboard(self) -> Dict[str, int]: ...

class Deck:
    """Main-deck and sideboard counts; identity derives from this content alone.

    Validates as a Pydantic field and serializes as ``{"decklist", "sideboard"}``.
    """

    def __init__(
        self, main: dict[str, int], sideboard: dict[str, int] | None = None
    ) -> None: ...
    @property
    def main(self) -> dict[str, int]: ...
    @property
    def sideboard(self) -> dict[str, int]: ...
    @property
    def identity(self) -> str: ...
    def player(self) -> PlayerConfig:
        """A seat named by the deck's identity, submitted to the world (``WORLD_SETUP``)."""
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: Any) -> Deck: ...
    def __hash__(self) -> int: ...

class DeckIssue:
    @property
    def line(self) -> int | None: ...
    @property
    def section(self) -> Literal["main", "sideboard"] | None: ...
    @property
    def card_name(self) -> str | None: ...
    @property
    def code(self) -> str: ...
    @property
    def message(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class DeckEntry:
    @property
    def line(self) -> int: ...
    @property
    def section(self) -> Literal["main", "sideboard"]: ...
    @property
    def count(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def card_def_id(self) -> int | None: ...
    @property
    def key(self) -> str | None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Decklist:
    """One counted-lines-v1 document: exact source, filename, entries and issues.

    Construction parses; names resolve when the decklist is admitted.
    """

    def __init__(self, source: str, filename: str = "deck.txt") -> None: ...
    @staticmethod
    def read(path: str | os.PathLike[str], filename: str | None = None) -> Decklist:
        """Bounded UTF-8 read keeping BOM/newline bytes; ``ValueError`` on bad files."""
    @property
    def dialect(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def filename(self) -> str: ...
    @property
    def entries(self) -> list[DeckEntry]: ...
    @property
    def issues(self) -> list[DeckIssue]: ...
    def validate_deck(self, deck: Deck) -> None:
        """Raise ``ValueError`` unless this source re-admits to exactly ``deck``
        in the installed world."""
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: Any) -> Decklist: ...

class DeckImport:
    @property
    def decklist(self) -> Decklist: ...
    @property
    def deck(self) -> Deck: ...
    def to_dict(self) -> dict[str, Any]: ...

class DeckImportError(ValueError):
    decklists: dict[str, Decklist]

MAX_DECKLIST_BYTES: int
DEVELOPMENT_RULESET: str
# `PlayerConfig.world` value that submits a deck to the installed world.
WORLD_SETUP: str

def admit_decklists(decklists: list[tuple[str, Decklist]]) -> list[DeckImport]:
    """Admit every decklist against the installed world or raise
    ``DeckImportError`` keyed by label."""

def check_decks(decks: list[Deck]) -> list[list[DeckIssue]]:
    """Each deck's construction and eligibility issues in the installed world, in
    order; an empty list admits. Allocates no game; game creation admits again."""

def authored_deck_setup(setup: str, deck_key: str) -> PlayerConfig:
    """One authored deck as a fixed setup; ``setup`` is a preset or its ``pack_key``."""

def authored_deck_keys(setup: str) -> list[str]:
    """The deck keys of an authored setup, named by preset or ``pack_key``."""

class ZoneEnum(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6

class PhaseEnum(IntEnum):
    BEGINNING = 0
    PRECOMBAT_MAIN = 1
    COMBAT = 2
    POSTCOMBAT_MAIN = 3
    ENDING = 4

class StepEnum(IntEnum):
    BEGINNING_UNTAP = 0
    BEGINNING_UPKEEP = 1
    BEGINNING_DRAW = 2
    PRECOMBAT_MAIN_STEP = 3
    COMBAT_BEGIN = 4
    COMBAT_DECLARE_ATTACKERS = 5
    COMBAT_DECLARE_BLOCKERS = 6
    COMBAT_DAMAGE = 7
    COMBAT_END = 8
    POSTCOMBAT_MAIN_STEP = 9
    ENDING_END = 10
    ENDING_CLEANUP = 11

class ActionEnum(IntEnum):
    PRIORITY_PLAY_LAND = 0
    PRIORITY_CAST_SPELL = 1
    PRIORITY_PASS_PRIORITY = 2
    DECLARE_ATTACKER = 3
    DECLARE_BLOCKER = 4
    CHOOSE_TARGET = 5
    PRIORITY_ACTIVATE_ABILITY = 6
    SCRY_KEEP = 7
    SCRY_BOTTOM = 8
    SELECT_CARD = 9
    DECLINE_CHOICE = 10
    PAY_COST = 11
    CHOOSE_MODE = 12
    TAP_FOR_COST = 13
    LEARN_TAKE_LESSON = 14
    LEARN_DISCARD = 15
    CHOOSE_TRIGGER = 16

class ActionSpaceEnum(IntEnum):
    GAME_OVER = 0
    PRIORITY = 1
    DECLARE_ATTACKER = 2
    DECLARE_BLOCKER = 3
    CHOOSE_TARGET = 4
    SCRY = 5
    LOOK_AND_SELECT = 6
    PAY_OR_NOT = 7
    MODAL = 8
    LEARN = 9
    WATERBEND = 10
    DISCARD = 11
    LEGEND_RULE = 12
    ORDER_TRIGGERS = 13

class StackObjectKindEnum(IntEnum):
    SPELL = 0
    ACTIVATED_ABILITY = 1
    TRIGGERED_ABILITY = 2

class StackTargetKindEnum(IntEnum):
    PLAYER = 0
    PERMANENT = 1
    STACK_OBJECT = 2
    CARD = 3

class EventTypeEnum(IntEnum):
    CARD_MOVED = 0
    DAMAGE_DEALT = 1
    LIFE_CHANGED = 2
    SPELL_CAST = 3
    SPELL_RESOLVED = 4
    SPELL_COUNTERED = 5
    ABILITY_TRIGGERED = 6
    COMBAT_ATTACKERS_DECLARED = 7
    BLOCKERS_DECLARED = 8
    COMBAT_DAMAGE_DEALT = 9
    PERMANENTS_DIED = 10
    TURN_STARTED = 11
    CARD_REVEALED = 12
    ABILITY_COUNTERED = 13

class EventEntityKindEnum(IntEnum):
    NONE = 0
    CARD = 1
    PERMANENT = 2
    PLAYER = 3
    OBJECT = 4
    DEFINITION = 5

class ManaCost:
    hybrid: List[tuple[int, int]]
    cost: List[int]
    mana_value: int

class CardTypes:
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

class Keywords:
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

class Player:
    player_index: int
    id: int
    is_active: bool
    is_agent: bool
    life: int
    zone_counts: List[int]
    graveyard_lessons: int
    known_hand: Dict[int, int]
    # Decklist counts by definition id; None when the viewer's opponent has a
    # closed decklist. Unseen counts the library, plus an opponent's hand and
    # face-down foretold cards; remaining counts sideboard copies outside the game.
    main_counts: Optional[Dict[int, int]]
    unseen_counts: Optional[Dict[int, int]]
    sideboard_counts: Optional[Dict[int, int]]
    remaining_sideboard_counts: Optional[Dict[int, int]]
    keeps_red_mana: bool
    no_maximum_hand_size: bool
    # What this player has done this turn.
    turn_history: PlayerTurnHistory
    # Extra turns this player will take after this one (CR 500.7).
    extra_turns: int
    # Additional combat phases still to come this turn (active player only).
    additional_combats: int
    combat_mana: int
    mana: tuple[int, int, int, int, int, int]
    lesson_mana: tuple[int, int, int, int, int, int]
    restricted_mana: List[RestrictedMana]
    hand_only_until: Optional[int]
    # Continuous effects on this player, in timestamp order.
    effects: list[ContinuousEffect]

class PlayerTurnHistory:
    """What one player has done this turn."""

    attacked: bool
    permanents_sacrificed: int
    land_entered: bool
    cards_drawn: int
    # Bending kinds done this turn: bits water 1, earth 2, fire 4, air 8.
    bent: int

class RestrictedMana:
    spells: List[str]
    abilities: List[str]
    mana: tuple[int, int, int, int, int, int]

class CastPermission:
    player_id: int
    cost: Optional[ManaCost]
    this_turn: bool
    through_next_turn: bool
    own_turn_only: bool
    after_turn: Optional[int]
    source_permanent_id: Optional[int]
    flash: bool
    any_mana: bool
    waterbend: bool
    foretold: bool

class Card:
    """A card in any zone. On the battlefield every characteristic field
    (name, power/toughness, card types, legendary, Ally, Lesson, ward,
    keywords) reads the engine's layered characteristics; elsewhere the
    printed card. Identity, token status and casting costs are printed."""

    zone: ZoneEnum
    owner_id: int
    id: int
    registry_key: int
    name: str
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
    face_down: bool
    cast_permission: Optional[CastPermission]
    x: Optional[int]

class SideboardCard:
    candidate_id: int
    owner_id: int
    registry_key: int
    name: str

class ContinuousEffect:
    """One continuous-effect record on a player or permanent (CR 611, 613)."""

    program: list[tuple[int, int]]
    description: str | None

    # Modification variant: ModifyPt, SetBasePt, AddKeywords, AddSubtypes,
    # BecomeCreature, Nonlegendary, LoseAllAbilities, Firebending,
    # CantBeBlocked, CantBlock, ExileIfWouldDie, KeepUnspentMana or
    # NoMaximumHandSize.
    modification: str
    power: int | None
    toughness: int | None
    amount: int | None
    keywords: list[str]
    subtypes: list[str]
    color: str | None
    # EndOfTurn or Indefinite.
    duration: str
    source_registry_key: int | None
    # The source's printed name (public: the spell or ability resolved).
    source_name: str | None
    timestamp: int

class EffectObject:
    object_id: int | None
    incarnation: int | None

class EffectMembership:
    target: EffectObject
    order: int

class SharedEffect:
    effect: ContinuousEffect
    controller_id: int
    source: EffectObject | None
    members: list[EffectMembership]

class CopyEffect:
    """The copy effect that made a permanent a copy (CR 707.2)."""

    # The printed definition at the root of the copy chain, and its name.
    registry_key: int
    name: str
    # Exceptions in its copiable values (CR 707.9b), in application order.
    exceptions: list[ContinuousEffect]

class Permanent:
    colors: list[str] | None
    keyword_counter_timestamps: dict[str, int]
    transforms: int
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
    attacking: bool
    exhausted_abilities: int
    once_each_turn_used: int
    granted_firebending: int
    temp_subtypes: List[str]
    # Effective keywords (printed + until-EOT grants).
    keywords: Keywords
    # Counters other than +1/+1 by kind (stun, fire, double_strike, quest, finality, conqueror, lore).
    counters: dict[str, int]
    # Permanent id this Aura or Equipment is attached to.
    attached_to: int | None
    # Name and creature types set by an effect; None while printed ones apply.
    name: str | None
    creature_types: List[str] | None
    is_granted_land: bool
    blocking: int | None
    abilities_lost: bool
    used_abilities: list[int]
    dealt_damage_this_turn: bool
    # Continuous effects on this permanent, in timestamp order.
    effects: list[ContinuousEffect]
    # Its timestamp (CR 613.7d/e) when an attachment renewed it; None while
    # it is `id`, its entering timestamp.
    timestamp: int | None
    # The copy effect that made it a copy; None for an original.
    copy: CopyEffect | None

class Turn:
    turn_number: int
    phase: PhaseEnum
    step: StepEnum
    active_player_id: int
    agent_player_id: int

class Action:
    action_type: ActionEnum
    focus: List[int]
    declared: bool | None
    description: str | None
    mode_index: int | None
    name_definition: int | None
    name_index: int | None
    # First and last vocabulary rows of a naming group choice.
    name_group: tuple[int, int] | None
    ability_index: int | None
    continuation_index: int | None

class ActionSpace:
    action_space_type: ActionSpaceEnum
    actions: List[Action]
    focus: List[int]

class StackTarget:
    kind: StackTargetKindEnum
    player_id: int | None
    permanent_id: int | None
    stack_object_id: int | None
    card_id: int | None

class StackObject:
    description: str | None
    stack_object_id: int
    kind: StackObjectKindEnum
    controller_id: int
    source_card_registry_key: int
    source_permanent_id: int | None
    chosen_mode: int | None
    ability_index: int | None
    targets: List[StackTarget]
    # The announced value of X (0 when none).
    x: int
    # Keywords a static grants this spell on the stack.
    granted_keywords: Keywords | None

class EventData:
    event_type: EventTypeEnum
    source_kind: EventEntityKindEnum
    source_id: int
    target_kind: EventEntityKindEnum
    target_id: int
    amount: int
    controller_id: int

class Observation:
    shared_effects: list[SharedEffect] | None
    continuation: list[ContinuationFrame]

    @staticmethod
    def from_json(text: str) -> Observation: ...
    def continuation_json(self) -> str: ...
    game_over: bool
    won: bool
    turn: Turn
    action_space: ActionSpace
    agent: Player
    agent_cards: List[Card]
    agent_sideboard: List[SideboardCard]
    agent_permanents: List[Permanent]
    opponent: Player
    opponent_cards: List[Card]
    opponent_permanents: List[Permanent]
    stack_objects: List[StackObject]
    recent_events: List[EventData]

    def validate(self) -> bool: ...
    def toJSON(self) -> str: ...

class AgentError(RuntimeError): ...

WORLD_VERSION: str
MAX_DECISION_CODE: int
CARD_DEFINITION_COUNT: int
# Most options one naming prompt offers; larger scopes publish groups first.
NAME_BRANCHING: int

class VisibleReference:
    object: int | None
    definition: int | None
    player: int | None

class ClauseTarget:
    legal_at_resolution_start: bool | None
    reference: VisibleReference
    clause: int

class ContinuationFrame:
    program: list[tuple[int, int]]
    program_description: str | None
    remembered: list[VisibleReference]
    inherited_from: VisibleReference | None
    attackers: str | None
    source_transforms: int
    kind: int
    decision: int
    parent: int | None
    controller: int
    source: VisibleReference
    mode: int | None
    ability: int | None
    kicked: bool
    active_clause: int | None
    targets: list[ClauseTarget]
    context: VisibleReference | None
    sacrifice: VisibleReference | None
    cost: list[int] | None
    hybrid: list[list[int]]
    lesson: bool
    waterbend: bool
    can_decline: bool
    mana_outputs: list[ManaOutput]
    resolutions_this_turn: int
    x: int
    granted_keywords: list[bool]
    delayed_frequency: int | None
    this_turn: bool

def validate_observation_json(text: str, *, schema_version: int = 8) -> None: ...
def parse_continuation_frames_json(text: str) -> list[ContinuationFrame]: ...

class ManaOutput:
    mode: int
    mana: list[int]
    lesson: list[int]

class NameCandidate:
    def __init__(self, name: str, definition: int | None = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def definition(self) -> int | None: ...

def naming_catalog() -> list[NameCandidate]: ...
def validate_policy_input(obs: Observation, version: int) -> None: ...
def policy_action_count(
    count: int, capacity: int, action_overflow: str = "error"
) -> int: ...
