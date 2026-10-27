"""Constructed competency positions and their scripted villains (exp-09).

Each `Scenario` injects an exact mid-game position through the engine's
scenario_* surface and scores the hero's line with a `Tracker`. This module
is Torch-free so replay children can rebuild scenario games; the agent
runners live in `manabot.verify.competency`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
import managym

HERO_SEAT = 0
VILLAIN_SEAT = 1

# Action / space / zone constants (managym enums).
CAST = int(managym.ActionEnum.PRIORITY_CAST_SPELL)
PLAY_LAND = int(managym.ActionEnum.PRIORITY_PLAY_LAND)
PASS = int(managym.ActionEnum.PRIORITY_PASS_PRIORITY)
PAY_COST = int(managym.ActionEnum.PAY_COST)
SPACE_PRIORITY = int(managym.ActionSpaceEnum.PRIORITY)
SPACE_ATTACKER = int(managym.ActionSpaceEnum.DECLARE_ATTACKER)
SPACE_BLOCKER = int(managym.ActionSpaceEnum.DECLARE_BLOCKER)
SPACE_TARGET = int(managym.ActionSpaceEnum.CHOOSE_TARGET)
SPACE_SCRY = int(managym.ActionSpaceEnum.SCRY)
SPACE_PAY_OR_NOT = int(managym.ActionSpaceEnum.PAY_OR_NOT)
ZONE_HAND = int(managym.ZoneEnum.HAND)
ZONE_BATTLEFIELD = int(managym.ZoneEnum.BATTLEFIELD)
ZONE_GRAVEYARD = int(managym.ZoneEnum.GRAVEYARD)
ZONE_STACK = int(managym.ZoneEnum.STACK)
PHASE_PRECOMBAT = int(managym.PhaseEnum.PRECOMBAT_MAIN)
PHASE_POSTCOMBAT = int(managym.PhaseEnum.POSTCOMBAT_MAIN)


# -----------------------------------------------------------------------------
# Raw-observation helpers
# -----------------------------------------------------------------------------


def _card_name(raw: Any, object_id: int) -> str | None:
    """Name of the card with observation ObjectId `object_id`, if visible."""

    for card in raw.agent_cards:
        if int(card.id) == object_id:
            return str(card.name)
    for card in raw.opponent_cards:
        if int(card.id) == object_id:
            return str(card.name)
    return None


def _battlefield_pairs(raw: Any, side: str) -> list[tuple[Any, Any]]:
    """(CardData, PermanentData) pairs for one side's battlefield.

    The engine pushes each permanent and its battlefield card in lockstep
    (observation.rs add_permanent), so the k-th battlefield-zone card on a
    side matches the k-th permanent on that side.
    """

    cards = raw.agent_cards if side == "agent" else raw.opponent_cards
    perms = raw.agent_permanents if side == "agent" else raw.opponent_permanents
    battlefield_cards = [c for c in cards if int(c.zone) == ZONE_BATTLEFIELD]
    return list(zip(battlefield_cards, perms))


def _zone_names(raw: Any, side: str, zone: int) -> list[str]:
    cards = raw.agent_cards if side == "agent" else raw.opponent_cards
    return [str(c.name) for c in cards if int(c.zone) == zone]


def _stack_names(raw: Any) -> list[str]:
    """Names of spells on the stack, top first (both players' spells)."""

    names = [
        (int(c.id), str(c.name))
        for c in list(raw.agent_cards) + list(raw.opponent_cards)
        if int(c.zone) == ZONE_STACK
    ]
    # populate_cards pushes stack spells top-first per side; merge is
    # approximate when both players have spells on the stack. Scenario and
    # micro games essentially always have <= 2, with the opponent's below.
    return [name for _, name in names]


def _hero_view(raw: Any) -> tuple[str, str]:
    """(hero_side, villain_side) keys for _battlefield_pairs/_zone_names.

    The observation is from the decision-holder's perspective; the hero is
    always seat 0 in scenarios.
    """

    if int(raw.agent.player_index) == HERO_SEAT:
        return "agent", "opponent"
    return "opponent", "agent"


def _action_casts(raw: Any, index: int) -> str | None:
    """If action `index` casts a spell, the spell's name (else None)."""

    action = raw.action_space.actions[index]
    if int(action.action_type) != CAST or not action.focus:
        return None
    return _card_name(raw, int(action.focus[0]))


def _find_cast_action(raw: Any, name: str) -> int | None:
    for index in range(len(raw.action_space.actions)):
        if _action_casts(raw, index) == name:
            return index
    return None


def _pass_index(raw: Any) -> int:
    actions = raw.action_space.actions
    for index, action in enumerate(actions):
        if int(action.action_type) == PASS:
            return index
    return len(actions) - 1


def _count_creatures(pairs: list[tuple[Any, Any]]) -> int:
    return sum(1 for card, _ in pairs if bool(card.card_types.is_creature))


# -----------------------------------------------------------------------------
# Scripted villain
# -----------------------------------------------------------------------------


class ScriptedVillain:
    """Deterministic villain: casts a fixed queue of cards in strict order
    (whenever the queue head is castable), plays a land every turn, attacks
    with everything, never blocks, pays demanded costs when able, declines
    optional mid-resolution choices."""

    def __init__(self, cast_queue: tuple[str, ...]):
        self.queue: list[str] = list(cast_queue)

    def act(self, raw: Any) -> int:
        space = raw.action_space
        kind = int(space.action_space_type)
        actions = space.actions

        if kind == SPACE_PRIORITY:
            if self.queue:
                index = _find_cast_action(raw, self.queue[0])
                if index is not None:
                    self.queue.pop(0)
                    return index
            for index, action in enumerate(actions):
                if int(action.action_type) == PLAY_LAND:
                    return index
            return _pass_index(raw)
        if kind == SPACE_ATTACKER:
            return 0  # [attack, decline] — always attack
        if kind == SPACE_BLOCKER:
            return len(actions) - 1  # last action is "no block"
        if kind == SPACE_TARGET:
            return 0
        if kind == SPACE_PAY_OR_NOT:
            for index, action in enumerate(actions):
                if int(action.action_type) == PAY_COST:
                    return index
            return len(actions) - 1
        if kind == SPACE_SCRY:
            return 0  # keep
        # look_and_select / learn / modal / waterbend: decline/done
        return len(actions) - 1


# -----------------------------------------------------------------------------
# Scenario definitions (instrument 1)
# -----------------------------------------------------------------------------


class Tracker(Protocol):
    finished: bool

    def observe_hero(self, raw: Any, action: int) -> None: ...
    def observe_state(self, raw: Any) -> None: ...
    def result(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Scenario:
    name: str
    hero_deck: dict[str, int]
    villain_deck: dict[str, int]
    hero_hand: tuple[str, ...]
    villain_hand: tuple[str, ...]
    hero_battlefield: tuple[str, ...]
    villain_battlefield: tuple[str, ...]
    hero_life: int
    villain_life: int
    villain_casts: tuple[str, ...]
    max_turns: int
    tracker: Callable[[], "Tracker"]
    correct_line: str


class CounterBombTracker:
    """S1: decline to counter the survivable spell, counter the bomb."""

    BOMB = "Shivan Dragon"
    BAIT = "Gray Ogre"

    def __init__(self) -> None:
        self.finished = False
        self.counter_cast_on: str | None = None
        self.bomb_countered = False
        self.bomb_resolved = False

    def observe_hero(self, raw: Any, action: int) -> None:
        if (
            _action_casts(raw, action) == "Counterspell"
            and self.counter_cast_on is None
        ):
            stack = _stack_names(raw)
            self.counter_cast_on = stack[0] if stack else None

    def observe_state(self, raw: Any) -> None:
        hero_side, villain_side = _hero_view(raw)
        battlefield = [c for c, _ in _battlefield_pairs(raw, villain_side)]
        if any(c.name == self.BOMB for c in battlefield):
            self.bomb_resolved = True
            self.finished = True
            return
        if self.counter_cast_on == self.BOMB and self.BOMB in _zone_names(
            raw, villain_side, ZONE_GRAVEYARD
        ):
            self.bomb_countered = True
            self.finished = True

    def result(self) -> dict[str, Any]:
        return {
            "correct": self.bomb_countered,
            "counter_cast_on": self.counter_cast_on,
            "countered_bait": self.counter_cast_on == self.BAIT,
            "bomb_resolved": self.bomb_resolved,
            "never_countered": self.counter_cast_on is None,
        }


class HoldWipeTracker:
    """S2: wait one villain deploy step so Pyroclasm wipes 4 instead of 2."""

    def __init__(self) -> None:
        self.finished = False
        self.cast_turn: int | None = None
        self.wiped: int | None = None
        self.hero_life_at_cast: int | None = None

    def observe_hero(self, raw: Any, action: int) -> None:
        if _action_casts(raw, action) == "Pyroclasm":
            hero_side, villain_side = _hero_view(raw)
            self.cast_turn = int(raw.turn.turn_number)
            self.wiped = _count_creatures(_battlefield_pairs(raw, villain_side))
            hero = raw.agent if hero_side == "agent" else raw.opponent
            self.hero_life_at_cast = int(hero.life)
            self.finished = True

    def observe_state(self, raw: Any) -> None:
        pass

    def result(self) -> dict[str, Any]:
        return {
            "correct": self.wiped is not None and self.wiped >= 4,
            "premature": self.wiped is not None and self.wiped <= 2,
            "wiped": self.wiped,
            "cast_turn": self.cast_turn,
            "hero_life_at_cast": self.hero_life_at_cast,
            "never_cast": self.wiped is None,
        }


class BoltThreatTracker:
    """S3: bolt the counter-engine creature, not the decoy or a face."""

    KEY = "Earth King's Lieutenant"

    def __init__(self) -> None:
        self.finished = False
        self.bolt_cast = False
        self._pending_target = False
        self._verify_kill = False
        self.target: str | None = None
        self.key_died = False

    def observe_hero(self, raw: Any, action: int) -> None:
        if _action_casts(raw, action) == "Lightning Bolt":
            self.bolt_cast = True
            self._pending_target = True
            return
        kind = int(raw.action_space.action_space_type)
        if self._pending_target and kind == SPACE_TARGET:
            self._pending_target = False
            chosen = raw.action_space.actions[action]
            focus = int(chosen.focus[0]) if chosen.focus else -1
            self.target = self._classify(raw, focus)
            self._verify_kill = self.target == self.KEY

    def _classify(self, raw: Any, focus: int) -> str:
        hero_side, villain_side = _hero_view(raw)
        hero = raw.agent if hero_side == "agent" else raw.opponent
        villain = raw.opponent if hero_side == "agent" else raw.agent
        if focus == int(hero.id):
            return "own_face"
        if focus == int(villain.id):
            return "villain_face"
        for card, perm in _battlefield_pairs(raw, villain_side):
            if int(perm.id) == focus:
                return str(card.name)
        for card, perm in _battlefield_pairs(raw, hero_side):
            if int(perm.id) == focus:
                return f"own:{card.name}"
        return "unknown"

    def observe_state(self, raw: Any) -> None:
        if not self._verify_kill:
            return
        _, villain_side = _hero_view(raw)
        on_battlefield = any(
            card.name == self.KEY for card, _ in _battlefield_pairs(raw, villain_side)
        )
        if not on_battlefield:
            self.key_died = True
            self.finished = True
        # If it survived the bolt (out of range), the run ends at max_turns.

    def result(self) -> dict[str, Any]:
        return {
            "correct": self.key_died,
            "bolt_cast": self.bolt_cast,
            "target": self.target,
            "never_cast": not self.bolt_cast,
        }


class RaceBlockTracker:
    """S4: attack with the evasive creatures, keep the ground as blockers."""

    def __init__(self) -> None:
        self.finished = False
        self._first_batch_done = False
        self.attacked: list[str] = []
        self.declined: list[str] = []
        self.blocks = 0
        self.winner: int | None = None

    def observe_hero(self, raw: Any, action: int) -> None:
        kind = int(raw.action_space.action_space_type)
        turn = int(raw.turn.turn_number)
        if kind == SPACE_ATTACKER and turn == 1 and not self._first_batch_done:
            chosen = raw.action_space.actions[action]
            focus = int(chosen.focus[0]) if chosen.focus else -1
            hero_side, _ = _hero_view(raw)
            name = "unknown"
            for card, perm in _battlefield_pairs(raw, hero_side):
                if int(perm.id) == focus:
                    name = str(card.name)
                    break
            # Attacker spaces are [attack, decline] in that order.
            (self.attacked if action == 0 else self.declined).append(name)
        if kind == SPACE_BLOCKER and action < len(raw.action_space.actions) - 1:
            self.blocks += 1

    def observe_state(self, raw: Any) -> None:
        if int(raw.turn.turn_number) > 1:
            self._first_batch_done = True
        if raw.game_over:
            self.finished = True

    def set_winner(self, winner: int | None) -> None:
        self.winner = winner

    def result(self) -> dict[str, Any]:
        ogres = sum(1 for name in self.attacked if name == "Gray Ogre")
        drakes = sum(1 for name in self.attacked if name == "Wind Drake")
        return {
            "correct": ogres == 0 and drakes == 2,
            "raced": ogres > 0,
            "full_hold": not self.attacked,
            "ogres_attacked": ogres,
            "drakes_attacked": drakes,
            "blocks": self.blocks,
            "hero_won": self.winner == HERO_SEAT if self.winner is not None else None,
        }


class HoldQuenchTracker:
    """S5: hold the two open mana for the counter instead of a main-phase 2-drop."""

    THREAT = "Craw Wurm"

    def __init__(self) -> None:
        self.finished = False
        self.main_choice: str | None = None
        self.quench_cast = False
        self.quenched = False
        self.threat_resolved = False

    def observe_hero(self, raw: Any, action: int) -> None:
        kind = int(raw.action_space.action_space_type)
        if (
            self.main_choice is None
            and kind == SPACE_PRIORITY
            and _find_cast_action(raw, "Otter-Penguin") is not None
        ):
            cast = _action_casts(raw, action)
            self.main_choice = cast if cast is not None else "pass"
        if _action_casts(raw, action) == "It'll Quench Ya!":
            self.quench_cast = True

    def observe_state(self, raw: Any) -> None:
        _, villain_side = _hero_view(raw)
        battlefield = [c.name for c, _ in _battlefield_pairs(raw, villain_side)]
        if self.THREAT in battlefield:
            self.threat_resolved = True
            self.finished = True
            return
        if self.quench_cast and self.THREAT in _zone_names(
            raw, villain_side, ZONE_GRAVEYARD
        ):
            self.quenched = True
            self.finished = True

    def result(self) -> dict[str, Any]:
        held = self.main_choice == "pass"
        return {
            "correct": held and self.quenched,
            "held_main": held,
            "main_choice": self.main_choice,
            "quenched": self.quenched,
            "threat_resolved": self.threat_resolved,
        }


SCENARIOS: dict[str, Scenario] = {
    "s1_counter_the_bomb": Scenario(
        name="s1_counter_the_bomb",
        hero_deck={"Island": 37, "Counterspell": 1, "Wind Drake": 2},
        villain_deck={"Mountain": 30, "Gray Ogre": 5, "Shivan Dragon": 5},
        hero_hand=("Counterspell",),
        villain_hand=("Gray Ogre", "Shivan Dragon"),
        hero_battlefield=("Wind Drake", "Wind Drake", "Island", "Island"),
        villain_battlefield=("Mountain",) * 6,
        hero_life=12,
        villain_life=20,
        villain_casts=("Gray Ogre", "Shivan Dragon", "Shivan Dragon", "Gray Ogre"),
        max_turns=6,
        tracker=CounterBombTracker,
        correct_line="decline the Gray Ogre, counter the Shivan Dragon",
    ),
    "s2_hold_the_wipe": Scenario(
        name="s2_hold_the_wipe",
        hero_deck={"Mountain": 39, "Pyroclasm": 1},
        villain_deck={"Mountain": 26, "Gray Ogre": 14},
        hero_hand=("Pyroclasm",),
        villain_hand=("Gray Ogre", "Gray Ogre"),
        hero_battlefield=("Mountain", "Mountain"),
        villain_battlefield=("Gray Ogre", "Gray Ogre") + ("Mountain",) * 6,
        hero_life=20,
        villain_life=20,
        villain_casts=("Gray Ogre",) * 6,
        max_turns=6,
        tracker=HoldWipeTracker,
        correct_line="pass turn 1, wipe 4 creatures on turn 2 instead of 2",
    ),
    "s3_bolt_the_threat": Scenario(
        name="s3_bolt_the_threat",
        hero_deck={"Mountain": 19, "Island": 18, "Lightning Bolt": 1, "Wind Drake": 2},
        villain_deck={
            "Forest": 13,
            "Plains": 13,
            "Invasion Reinforcements": 5,
            "Earth King's Lieutenant": 5,
            "White Lotus Reinforcements": 4,
        },
        hero_hand=("Lightning Bolt",),
        villain_hand=(
            "Earth King's Lieutenant",
            "White Lotus Reinforcements",
            "Invasion Reinforcements",
        ),
        hero_battlefield=("Wind Drake", "Wind Drake", "Mountain", "Island", "Island"),
        villain_battlefield=(
            "Invasion Reinforcements",
            "Forest",
            "Forest",
            "Plains",
            "Plains",
        ),
        hero_life=14,
        villain_life=20,
        villain_casts=(
            "Earth King's Lieutenant",
            "White Lotus Reinforcements",
            "Invasion Reinforcements",
        ),
        max_turns=4,
        tracker=BoltThreatTracker,
        correct_line="hold the bolt for the Lieutenant; kill it before it leaves 3-damage range",
    ),
    "s4_race_vs_block": Scenario(
        name="s4_race_vs_block",
        hero_deck={"Island": 24, "Mountain": 12, "Wind Drake": 2, "Gray Ogre": 2},
        villain_deck={"Mountain": 37, "Gray Ogre": 3},
        hero_hand=(),
        villain_hand=(),
        hero_battlefield=("Wind Drake", "Wind Drake", "Gray Ogre", "Gray Ogre"),
        villain_battlefield=(
            "Gray Ogre",
            "Gray Ogre",
            "Gray Ogre",
            "Mountain",
            "Mountain",
            "Mountain",
        ),
        hero_life=10,
        villain_life=17,
        villain_casts=(),
        max_turns=10,
        tracker=RaceBlockTracker,
        correct_line="attack with the two fliers only; hold both ground creatures as blockers",
    ),
    "s5_hold_up_quench": Scenario(
        name="s5_hold_up_quench",
        hero_deck={"Island": 38, "It'll Quench Ya!": 1, "Otter-Penguin": 1},
        villain_deck={"Forest": 33, "Craw Wurm": 7},
        hero_hand=("It'll Quench Ya!", "Otter-Penguin"),
        villain_hand=("Craw Wurm",),
        hero_battlefield=("Island", "Island"),
        villain_battlefield=("Forest",) * 6,
        hero_life=20,
        villain_life=20,
        villain_casts=("Craw Wurm",) * 3,
        max_turns=3,
        tracker=HoldQuenchTracker,
        correct_line="pass the main phase holding 1U; counter the tapped-out Craw Wurm",
    ),
}




def build_scenario_env(
    scenario: Scenario, obs_space: ObservationSpace | None, seed: int
) -> tuple[Env, dict[str, np.ndarray], Any]:
    """Reset an Env, inject the scenario position, and refresh.

    Returns (env, encoded_obs, raw_obs) at the first hero decision of the
    constructed position.
    """

    match = Match(
        MatchConfig(
            hero="hero",
            villain="villain",
            hero_deck=dict(scenario.hero_deck),
            villain_deck=dict(scenario.villain_deck),
        )
    )
    env = Env(
        match,
        obs_space or ObservationSpace(),
        Reward(RewardConfig()),
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    _, _ = env.reset(seed=seed)
    raw = env.last_raw_obs
    if int(raw.action_space.action_space_type) != SPACE_PRIORITY:
        raise RuntimeError(
            f"scenario {scenario.name}: first decision is not priority "
            f"(kind={int(raw.action_space.action_space_type)})"
        )
    engine = env._engine
    engine.scenario_clear_hand(HERO_SEAT)
    engine.scenario_clear_hand(VILLAIN_SEAT)
    for name in scenario.hero_hand:
        engine.scenario_force_card_in_hand(HERO_SEAT, name)
    for name in scenario.villain_hand:
        engine.scenario_force_card_in_hand(VILLAIN_SEAT, name)
    for name in scenario.hero_battlefield:
        engine.scenario_force_battlefield(HERO_SEAT, name, True)
    for name in scenario.villain_battlefield:
        engine.scenario_force_battlefield(VILLAIN_SEAT, name, True)
    engine.scenario_set_life(HERO_SEAT, scenario.hero_life)
    engine.scenario_set_life(VILLAIN_SEAT, scenario.villain_life)
    obs, raw = env.scenario_refresh()
    return env, obs, raw
