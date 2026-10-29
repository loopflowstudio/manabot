from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

class CompiledContentIdentity(TypedDict):
    pack_key: str
    ir_hash: str
    source_hash: str

class PoolDefinition(TypedDict):
    card_def_id: int | None
    semantic_key: str | None
    registry_name: str
    eligibility: Literal["eligible", "unsupported"]
    reason: str | None

class ContentPool(TypedDict):
    pack_key: str
    world_version: str
    content_digest: str
    compiled_semantics: CompiledContentIdentity
    definitions: list[PoolDefinition]

def content_pool(pack_key: str) -> ContentPool: ...
def set_cards(sets: list[str]) -> list[PoolDefinition]:
    """World pool rows for typed cards whose key prefix is in ``sets``."""

def unsupported_cards(names: list[str]) -> list[str]:
    """Distinct names the current world does not support for deck entry."""

def semantic_ir_json(pack_key: str) -> str:
    """Canonical IR compiled natively from the typed set definitions."""

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
        self, obs: Observation, policy_input_version: int = 4
    ) -> Dict[str, np.ndarray]: ...
    def encode_observation_into(
        self, obs: Observation, out: Dict[str, np.ndarray]
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
    def set_buffers(self, buffers: Dict[str, np.ndarray]) -> None: ...
    def reset_all_into_buffers(self, player_configs: List[PlayerConfig]) -> None: ...
    def reset_each_into_buffers(
        self, player_configs: list[list[PlayerConfig]]
    ) -> None: ...
    def step_into_buffers(
        self, actions: list[int], active: list[bool] | None = None
    ) -> None: ...
    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool]
    ) -> None: ...
    def replace_failed_into_buffers(self, failed: list[bool]) -> None: ...
    def semantic_decision_frames_json(
        self, active: list[bool] | None = None
    ) -> list[str | None]: ...
    def current_game_seeds(self) -> list[int]: ...
    def current_agent_indices(self) -> list[int]: ...
    def reset_seeds_into_buffers(self, seeds: list[int]) -> None: ...
    def skip_trivial_counts(self) -> List[int]: ...
    def get_last_info(self) -> List[Dict[str, Any]]: ...

class PlayerConfig:
    def __init__(
        self,
        name: str,
        decklist: Dict[str, int],
        sideboard: Optional[Dict[str, int]] = None,
        content_pack: str | None = None,
    ) -> None: ...
    name: str
    decklist: Dict[str, int]
    sideboard: Dict[str, int]
    content_pack: str | None

def authored_deck_setup(pack_key: str, deck_key: str) -> PlayerConfig: ...

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
    sideboard_counts: Dict[int, int]
    remaining_sideboard_counts: Dict[int, int]
    keeps_red_mana: bool
    no_maximum_hand_size: bool
    combat_mana: int
    mana: tuple[int, int, int, int, int, int]
    lesson_mana: tuple[int, int, int, int, int, int]

class Card:
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

class SideboardCard:
    candidate_id: int
    owner_id: int
    registry_key: int
    name: str

class Permanent:
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
    temp_subtypes: List[str]
    # Effective keywords (printed + until-EOT grants).
    keywords: Keywords
    stun_counters: int
    fire_counters: int
    double_strike_counters: int
    # Permanent id this Aura or Equipment is attached to.
    attached_to: int | None
    # Name and creature types set by an effect; None while printed ones apply.
    name: str | None
    creature_types: List[str] | None
    is_granted_land: bool
    # Firebending gained until end of turn.
    firebending: int
    attacking: bool
    blocking: int | None
    abilities_lost: bool
    used_abilities: list[int]

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
CARD_DEFINITION_COUNT: int

class VisibleReference:
    object: int | None
    definition: int | None
    player: int | None

class ClauseTarget:
    legal_at_resolution_start: bool | None
    reference: VisibleReference
    clause: int

class ContinuationFrame:
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

def validate_observation_json(text: str) -> None: ...
def parse_continuation_frames_json(text: str) -> list[ContinuationFrame]: ...

class ManaOutput:
    mode: int
    mana: list[int]
    lesson: list[int]
