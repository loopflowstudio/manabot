"""Lightweight Python enum mirrors for GUI wire serialization.

The native PyO3 enums expose constants but are not constructible from integers.
The training observation module has equivalent mirrors, but importing it pulls
the full NumPy/Gymnasium/training stack into the exact Search-64 play runtime.
"""

from enum import IntEnum


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
    DISCARD_THEN_DRAW = 9
    WATERBEND = 10


class ZoneEnum(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6


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
