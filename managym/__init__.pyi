from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Tuple


class Env:
    """Main environment class for Magic gameplay."""

    def __init__(self, seed: int = 0, skip_trivial: bool = True, enable_profiler: bool = False, enable_behavior_tracking: bool = False) -> None:
        """Initialize environment.

        Args:
            seed: Random seed for game
            skip_trivial: Skip states with only one action
        """
        ...

    def reset(self, player_configs: List[PlayerConfig]) -> Tuple[Observation, Dict[str, Any]]:
        """Reset environment with new players.

        Args:
            player_configs: List of player configurations

        Returns:
            Tuple of (initial observation, info dict)
        """
        ...

    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """Take an action in the game.

        Args:
            action: Index of action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        ...

    def info(self) -> Dict[str, Any]:
        """Get information about the environment.

        Returns:
            Dictionary of environment information -- profiler and behavior stats
        """
        ...

    def export_profile_baseline(self) -> str:
        """Export profiler stats to a baseline string."""
        ...

    def compare_profile(self, baseline: str) -> str:
        """Compare profiler stats against a baseline."""
        ...

# ---------------------------------------------------------------------
# Player Config -- Used to define the match
# ---------------------------------------------------------------------

class PlayerConfig:
    def __init__(self, name: str, decklist: Dict[str, int]) -> None: ...
    name: str
    decklist: Dict[str, int]


class ZoneEnum(IntEnum):
    """Zones in Magic: The Gathering where objects can exist."""
    LIBRARY = 0      # Player's deck
    HAND = 1         # Cards in hand
    BATTLEFIELD = 2  # Cards in play
    GRAVEYARD = 3    # Discard pile
    STACK = 4        # Spells and abilities waiting to resolve
    EXILE = 5        # Removed from game
    COMMAND = 6      # Command zone for special cards

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


class ActionSpaceEnum(IntEnum):
    GAME_OVER = 0
    PRIORITY = 1
    DECLARE_ATTACKER = 2
    DECLARE_BLOCKER = 3


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
    power: int
    toughness: int
    card_types: CardTypes
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

    def validate(self) -> bool: ...
    def toJSON(self) -> str: ...


class AgentError(RuntimeError): ...
