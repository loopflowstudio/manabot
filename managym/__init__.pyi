from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Tuple

import numpy as np

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
    def step(
        self, action: int
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]: ...
    def info(self) -> Dict[str, Any]: ...
    def export_profile_baseline(self) -> str: ...
    def compare_profile(self, baseline: str) -> str: ...
    def encode_observation(self, obs: Observation) -> Dict[str, np.ndarray]: ...
    def encode_observation_into(
        self, obs: Observation, out: Dict[str, np.ndarray]
    ) -> None: ...

class VectorEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        skip_trivial: bool = True,
        opponent_policy: str = "none",
        num_threads: int | None = None,
    ) -> None: ...
    def reset_all(
        self, player_configs: List[PlayerConfig]
    ) -> List[Tuple[Observation, Dict[str, Any]]]: ...
    def step(
        self, actions: List[int]
    ) -> List[Tuple[Observation, float, bool, bool, Dict[str, Any]]]: ...
    def set_buffers(self, buffers: Dict[str, np.ndarray]) -> None: ...
    def reset_all_into_buffers(self, player_configs: List[PlayerConfig]) -> None: ...
    def step_into_buffers(self, actions: List[int]) -> None: ...
    def get_last_info(self) -> List[Dict[str, Any]]: ...

class PlayerConfig:
    def __init__(self, name: str, decklist: Dict[str, int]) -> None: ...
    name: str
    decklist: Dict[str, int]

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

class ActionSpaceEnum(IntEnum):
    GAME_OVER = 0
    PRIORITY = 1
    DECLARE_ATTACKER = 2
    DECLARE_BLOCKER = 3
    CHOOSE_TARGET = 4

class StackObjectKindEnum(IntEnum):
    SPELL = 0
    ACTIVATED_ABILITY = 1
    TRIGGERED_ABILITY = 2

class StackTargetKindEnum(IntEnum):
    PLAYER = 0
    PERMANENT = 1
    STACK_OBJECT = 2

class ManaCost:
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
    vigilance: bool
    trample: bool
    first_strike: bool
    double_strike: bool
    deathtouch: bool
    lifelink: bool
    defender: bool
    menace: bool

class Player:
    player_index: int
    id: int
    is_active: bool
    is_agent: bool
    life: int
    zone_counts: List[int]

class Card:
    zone: ZoneEnum
    owner_id: int
    id: int
    registry_key: int
    name: str
    power: int
    toughness: int
    card_types: CardTypes
    keywords: Keywords
    mana_cost: ManaCost

class Permanent:
    id: int
    controller_id: int
    tapped: bool
    damage: int
    is_summoning_sick: bool

class Turn:
    turn_number: int
    phase: PhaseEnum
    step: StepEnum
    active_player_id: int
    agent_player_id: int

class Action:
    action_type: ActionEnum
    focus: List[int]

class ActionSpace:
    action_space_type: ActionSpaceEnum
    actions: List[Action]
    focus: List[int]

class StackTarget:
    kind: StackTargetKindEnum
    player_id: int | None
    permanent_id: int | None
    stack_object_id: int | None

class StackObject:
    stack_object_id: int
    kind: StackObjectKindEnum
    controller_id: int
    source_card_registry_key: int
    source_permanent_id: int | None
    ability_index: int | None
    targets: List[StackTarget]

class Observation:
    game_over: bool
    won: bool
    turn: Turn
    action_space: ActionSpace
    agent: Player
    agent_cards: List[Card]
    agent_permanents: List[Permanent]
    opponent: Player
    opponent_cards: List[Card]
    opponent_permanents: List[Permanent]
    stack_objects: List[StackObject]

    def validate(self) -> bool: ...
    def toJSON(self) -> str: ...

class AgentError(RuntimeError): ...
