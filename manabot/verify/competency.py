"""Control-competency measurement suite (exp-09, wave/search C9).

Can an agent play reactive/control Magic, or does it just look like it can?
Two instruments, built to discriminate H1 ("GW/aggro is structurally
favored") from H2 ("the flat-MC pilot cannot play control: strategy fusion
never holds interaction for value, random rollouts burn inherited
counterspells on the first target"):

1. **Competency scenarios** — construct an exact mid-game position via the
   engine's scenario_* state-injection surface (managym/src/flow/scenario.rs),
   run an agent from it N times against a scripted villain, and score each
   run against a documented known-correct line. Every scenario puts the hero
   in seat 0; the villain is a deterministic script, so run-to-run variance
   comes only from the agent's own randomness (search/determinization seeds,
   library shuffles).

2. **Micro-format mirrors** — two minimal decks (MICRO_AGGRO, MICRO_CONTROL)
   played seat-balanced at several search strengths, with per-decision
   behavioral probes on the control side: what counterspells countered and
   when, what bolts targeted, and whether instants were held.

Usage:
    uv run python -m manabot.verify.competency scenarios \
        --agents random,search-16,search-64,search-256 --runs 100 \
        --workers 4 --out reports/data/exp-09-scenarios.json
    uv run python -m manabot.verify.competency micro \
        --games 300 --workers 4 --out reports/data/exp-09-micro.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.flat_mc import (
    make_player,
    spec_name,
    wilson_interval,
)
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
# Micro-format decks (instrument 2)
# -----------------------------------------------------------------------------

# 40 cards, 5 distinct: creatures that curve out and turn sideways.
MICRO_AGGRO = {
    "Mountain": 9,
    "Island": 9,
    "Gray Ogre": 8,
    "Raging Goblin": 7,
    "Wind Drake": 7,
}

# 40 cards, 6 distinct: fewer creatures, permission, removal, card draw.
MICRO_CONTROL = {
    "Island": 10,
    "Mountain": 8,
    "Wind Drake": 6,
    "Counterspell": 6,
    "Lightning Bolt": 7,
    "Ancestral Recall": 3,
}


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

    names = [(int(c.id), str(c.name)) for c in list(raw.agent_cards) + list(raw.opponent_cards)
             if int(c.zone) == ZONE_STACK]
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
        # look_and_select / discard_then_draw / modal / waterbend: decline/done
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
        if _action_casts(raw, action) == "Counterspell" and self.counter_cast_on is None:
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
        villain_battlefield=("Invasion Reinforcements", "Forest", "Forest", "Plains", "Plains"),
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
        villain_battlefield=("Gray Ogre", "Gray Ogre", "Gray Ogre", "Mountain", "Mountain", "Mountain"),
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


# -----------------------------------------------------------------------------
# Scenario runner
# -----------------------------------------------------------------------------


def build_scenario_env(
    scenario: Scenario, obs_space: ObservationSpace | None, seed: int
) -> tuple[Env, dict[str, np.ndarray], Any]:
    """Reset an Env, inject the scenario position, and refresh.

    Returns (env, encoded_obs, raw_obs) at the first hero decision of the
    constructed position.
    """

    match = Match(
        MatchHypers(
            hero="hero",
            villain="villain",
            hero_deck=dict(scenario.hero_deck),
            villain_deck=dict(scenario.villain_deck),
        )
    )
    env = Env(
        match,
        obs_space or ObservationSpace(),
        Reward(RewardHypers()),
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


def run_scenario_once(
    scenario: Scenario,
    player: Any,
    obs_space: ObservationSpace | None,
    seed: int,
    max_steps: int = 4000,
) -> dict[str, Any]:
    """One scored run: hero = `player`, villain = the scenario script."""

    env, obs, raw = build_scenario_env(scenario, obs_space, seed)
    villain = ScriptedVillain(scenario.villain_casts)
    tracker = scenario.tracker()

    done = False
    steps = 0
    winner: int | None = None
    while not done and not tracker.finished and steps < max_steps:
        raw = env.last_raw_obs
        tracker.observe_state(raw)
        if tracker.finished:
            break
        if int(raw.turn.turn_number) > scenario.max_turns:
            break
        actor = int(raw.agent.player_index)
        if actor == VILLAIN_SEAT:
            action = villain.act(raw)
        else:
            action = player.act(env, obs)
            tracker.observe_hero(raw, action)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = bool(terminated or truncated)
    if done:
        from manabot.verify.util import winner_from_info_or_obs

        winner = winner_from_info_or_obs(info, env.last_raw_obs)
    tracker.observe_state(env.last_raw_obs)
    if hasattr(tracker, "set_winner"):
        tracker.set_winner(winner)

    result = tracker.result()
    result["steps"] = steps
    result["end_turn"] = int(env.last_raw_obs.turn.turn_number)
    result["game_over"] = done
    return result


def _scenario_worker(args: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    scenario = SCENARIOS[args["scenario"]]
    results = []
    for run_index in range(args["num_runs"]):
        seed = args["base_seed"] + args["run_offset"] + run_index
        player, obs_space = make_player(args["spec"], seed=seed * 2 + 1)
        results.append(
            run_scenario_once(scenario, player, obs_space, seed=seed)
        )
    return results


def aggregate_scenario_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    lo, hi = wilson_interval(correct, total)
    aggregate: dict[str, Any] = {
        "runs": total,
        "correct": correct,
        "correct_rate": correct / total if total else 0.0,
        "ci_lower": lo,
        "ci_upper": hi,
    }
    # Sum/aggregate the scenario-specific detail fields.
    detail: dict[str, Any] = {}
    for key in results[0].keys() if results else []:
        if key in ("correct", "steps", "end_turn", "game_over"):
            continue
        values = [r.get(key) for r in results]
        if all(isinstance(v, bool) or v is None for v in values):
            detail[key] = sum(1 for v in values if v)
        elif all(isinstance(v, (int, float)) or v is None for v in values):
            numeric = [v for v in values if v is not None]
            detail[key] = float(np.mean(numeric)) if numeric else None
        else:
            counts: dict[str, int] = {}
            for v in values:
                label = str(v)
                counts[label] = counts.get(label, 0) + 1
            detail[key] = counts
    aggregate["detail"] = detail
    return aggregate


def run_scenario_suite(
    scenario_names: list[str],
    agent_specs: list[dict[str, Any]],
    *,
    runs: int,
    workers: int,
    base_seed: int,
    out_path: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    ctx = mp.get_context("spawn")
    for scenario_name in scenario_names:
        scenario_block = results.setdefault(scenario_name, {})
        scenario_block.setdefault(
            "correct_line", SCENARIOS[scenario_name].correct_line
        )
        for spec in agent_specs:
            agent_name = spec_name(spec)
            if agent_name in scenario_block:
                print(f"[skip] {scenario_name} / {agent_name}")
                continue
            start = time.perf_counter()
            chunks = []
            per_worker = runs // workers
            remainder = runs % workers
            offset = 0
            for w in range(workers):
                chunk = per_worker + (1 if w < remainder else 0)
                if chunk == 0:
                    continue
                chunks.append(
                    {
                        "scenario": scenario_name,
                        "spec": spec,
                        "num_runs": chunk,
                        "base_seed": base_seed,
                        "run_offset": offset,
                    }
                )
                offset += chunk
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    outputs = list(pool.map(_scenario_worker, chunks))
            except Exception as error:  # checkpoint dim mismatch etc.
                print(f"[fail] {scenario_name} / {agent_name}: {error}")
                scenario_block[agent_name] = {"error": str(error)}
                out_path.write_text(json.dumps(results, indent=2))
                continue
            run_results = [r for out in outputs for r in out]
            aggregate = aggregate_scenario_results(run_results)
            aggregate["wall_seconds"] = time.perf_counter() - start
            scenario_block[agent_name] = aggregate
            out_path.write_text(json.dumps(results, indent=2))
            print(
                f"[done] {scenario_name} / {agent_name}: "
                f"correct {aggregate['correct_rate']:.3f} "
                f"[{aggregate['ci_lower']:.3f},{aggregate['ci_upper']:.3f}] "
                f"({aggregate['wall_seconds']:.0f}s)"
            )
    return results


# -----------------------------------------------------------------------------
# Micro-format matchups with behavioral probes (instrument 2)
# -----------------------------------------------------------------------------


@dataclass
class BehaviorProbe:
    """Per-game behavioral stats for the control-deck hero."""

    counter_windows: int = 0
    counter_casts: int = 0
    counter_casts_first_window: int = 0
    countered_names: dict[str, int] = field(default_factory=dict)
    countered_mana_values: list[int] = field(default_factory=list)
    bolt_casts: int = 0
    bolt_biggest: int = 0
    bolt_face: int = 0
    bolt_own: int = 0
    bolt_multi_choice: int = 0
    hold_windows: int = 0
    hold_breaks: int = 0

    _seen_counter_window: bool = False
    _pending_bolt: bool = False
    _pending_counter: bool = False

    def observe(self, raw: Any, action: int) -> None:
        kind = int(raw.action_space.action_space_type)
        if kind == SPACE_TARGET:
            if self._pending_bolt:
                self._pending_bolt = False
                self._classify_bolt_target(raw, action)
            if self._pending_counter:
                self._pending_counter = False
                self._classify_counter_target(raw, action)
            return
        self._pending_bolt = False
        self._pending_counter = False
        if kind != SPACE_PRIORITY:
            return

        counter_index = _find_cast_action(raw, "Counterspell")
        if counter_index is not None:
            self.counter_windows += 1
            first_window = not self._seen_counter_window
            self._seen_counter_window = True
            if _action_casts(raw, action) == "Counterspell":
                self.counter_casts += 1
                if first_window:
                    self.counter_casts_first_window += 1
                stack = self._stack_cards(raw)
                if len(stack) == 1:
                    self._record_countered(stack[0])
                else:
                    self._pending_counter = True

        if _action_casts(raw, action) == "Lightning Bolt":
            self.bolt_casts += 1
            self._pending_bolt = True

        # Instant-holding: own main phase, empty stack, a creature is
        # castable, a Counterspell is in hand, and casting the creature
        # would drop open mana below UU.
        if (
            bool(raw.agent.is_active)
            and int(raw.turn.phase) in (PHASE_PRECOMBAT, PHASE_POSTCOMBAT)
            and not list(raw.stack_objects)
            and any(c.name == "Counterspell" and int(c.zone) == ZONE_HAND for c in raw.agent_cards)
        ):
            untapped_lands = sum(
                1
                for card, perm in _battlefield_pairs(raw, "agent")
                if bool(card.card_types.is_land) and not bool(perm.tapped)
            )
            creature_costs = [
                int(c.mana_cost.mana_value)
                for index in range(len(raw.action_space.actions))
                for c in raw.agent_cards
                if (name := _action_casts(raw, index)) is not None
                and c.name == name
                and int(c.zone) == ZONE_HAND
                and bool(c.card_types.is_creature)
            ]
            if creature_costs and untapped_lands - min(creature_costs) < 2:
                self.hold_windows += 1
                cast = _action_casts(raw, action)
                if cast is not None:
                    for c in raw.agent_cards:
                        if c.name == cast and bool(c.card_types.is_creature):
                            self.hold_breaks += 1
                            break

    def _stack_cards(self, raw: Any) -> list[Any]:
        return [
            c
            for c in list(raw.agent_cards) + list(raw.opponent_cards)
            if int(c.zone) == ZONE_STACK
        ]

    def _record_countered(self, card: Any) -> None:
        name = str(card.name)
        self.countered_names[name] = self.countered_names.get(name, 0) + 1
        self.countered_mana_values.append(int(card.mana_cost.mana_value))

    def _classify_counter_target(self, raw: Any, action: int) -> None:
        chosen = raw.action_space.actions[action]
        focus = int(chosen.focus[0]) if chosen.focus else -1
        for card in self._stack_cards(raw):
            if int(card.id) == focus:
                self._record_countered(card)
                return

    def _classify_bolt_target(self, raw: Any, action: int) -> None:
        chosen = raw.action_space.actions[action]
        focus = int(chosen.focus[0]) if chosen.focus else -1
        villain_creatures = [
            (card, perm) for card, perm in _battlefield_pairs(raw, "opponent")
            if bool(card.card_types.is_creature)
        ]
        # Target-quality is only meaningful when there was a real choice
        # among enemy creatures: bolt_biggest counts max-power picks among
        # multi-creature boards only, so bolt_biggest / bolt_multi_choice
        # is a proper rate in [0, 1].
        multi = len(villain_creatures) >= 2
        if multi:
            self.bolt_multi_choice += 1
        if focus == int(raw.opponent.id):
            self.bolt_face += 1
            return
        for card, perm in villain_creatures:
            if int(perm.id) == focus:
                max_power = max(int(p.power) for _, p in villain_creatures)
                if multi and int(perm.power) >= max_power:
                    self.bolt_biggest += 1
                return
        for card, perm in _battlefield_pairs(raw, "agent"):
            if int(perm.id) == focus:
                self.bolt_own += 1
                return

    def to_dict(self) -> dict[str, Any]:
        return {
            "counter_windows": self.counter_windows,
            "counter_casts": self.counter_casts,
            "counter_casts_first_window": self.counter_casts_first_window,
            "countered_names": self.countered_names,
            "countered_mana_values": self.countered_mana_values,
            "bolt_casts": self.bolt_casts,
            "bolt_biggest": self.bolt_biggest,
            "bolt_face": self.bolt_face,
            "bolt_own": self.bolt_own,
            "bolt_multi_choice": self.bolt_multi_choice,
            "hold_windows": self.hold_windows,
            "hold_breaks": self.hold_breaks,
        }


def play_probed_games(
    hero_spec: dict[str, Any],
    villain_spec: dict[str, Any],
    *,
    hero_deck: dict[str, int],
    villain_deck: dict[str, int],
    num_games: int,
    seed: int,
    game_offset: int = 0,
) -> dict[str, Any]:
    """Seat-balanced matchup loop with per-decision hero behavior probes.

    Mirrors manabot.sim.flat_mc.play_games (game i puts the hero in seat
    i % 2) and adds a BehaviorProbe on every hero decision.
    """

    from manabot.verify.util import winner_from_info_or_obs

    hero_player, hero_obs_space = make_player(hero_spec, seed=seed * 2 + 1)
    villain_player, villain_obs_space = make_player(villain_spec, seed=seed * 2 + 2)
    obs_space = hero_obs_space or villain_obs_space or ObservationSpace()

    match = Match(
        MatchHypers(
            hero=spec_name(hero_spec)[:32],
            villain=spec_name(villain_spec)[:32],
            hero_deck=dict(hero_deck),
            villain_deck=dict(villain_deck),
        )
    )
    match_swapped = match.swapped()
    env = Env(
        match,
        obs_space,
        Reward(RewardHypers()),
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )

    records: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    start = time.perf_counter()
    for i in range(num_games):
        game_index = game_offset + i
        hero_seat = game_index % 2
        obs, _ = env.reset(
            seed=seed + game_index,
            options={"match": match_swapped} if hero_seat == 1 else None,
        )
        probe = BehaviorProbe()
        done = False
        steps = 0
        info: dict[str, Any] = {}
        while not done:
            raw = env.last_raw_obs
            acting = int(raw.agent.player_index)
            if acting == hero_seat:
                action = hero_player.act(env, obs)
                probe.observe(raw, action)
            else:
                action = villain_player.act(env, obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
        winner = winner_from_info_or_obs(info, env.last_raw_obs)
        records.append(
            {
                "game_index": game_index,
                "hero_seat": hero_seat,
                "hero_won": winner == hero_seat,
                "winner": winner,
                "steps": steps,
                "turns": int(env.last_raw_obs.turn.turn_number),
            }
        )
        probes.append(probe.to_dict())
    return {
        "records": records,
        "probes": probes,
        "wall_seconds": time.perf_counter() - start,
    }


def _micro_worker(args: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    return play_probed_games(
        args["hero_spec"],
        args["villain_spec"],
        hero_deck=args["hero_deck"],
        villain_deck=args["villain_deck"],
        num_games=args["num_games"],
        seed=args["seed"],
        game_offset=args["game_offset"],
    )


def aggregate_micro_cell(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    records = [r for out in outputs for r in out["records"]]
    probes = [p for out in outputs for p in out["probes"]]
    total = len(records)
    wins = sum(1 for r in records if r["hero_won"])
    lo, hi = wilson_interval(wins, total)

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    counter_windows = sum(p["counter_windows"] for p in probes)
    counter_casts = sum(p["counter_casts"] for p in probes)
    first_window = sum(p["counter_casts_first_window"] for p in probes)
    mana_values = [v for p in probes for v in p["countered_mana_values"]]
    countered_names: dict[str, int] = {}
    for p in probes:
        for name, count in p["countered_names"].items():
            countered_names[name] = countered_names.get(name, 0) + count
    bolt_casts = sum(p["bolt_casts"] for p in probes)
    bolt_biggest = sum(p["bolt_biggest"] for p in probes)
    bolt_face = sum(p["bolt_face"] for p in probes)
    bolt_own = sum(p["bolt_own"] for p in probes)
    bolt_multi = sum(p["bolt_multi_choice"] for p in probes)
    hold_windows = sum(p["hold_windows"] for p in probes)
    hold_breaks = sum(p["hold_breaks"] for p in probes)

    per_seat = {}
    for seat, label in ((0, "play"), (1, "draw")):
        seat_records = [r for r in records if r["hero_seat"] == seat]
        seat_wins = sum(1 for r in seat_records if r["hero_won"])
        per_seat[f"win_rate_on_{label}"] = rate(seat_wins, len(seat_records))

    return {
        "games": total,
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "ci_lower": lo,
        "ci_upper": hi,
        **per_seat,
        "mean_turns": float(np.mean([r["turns"] for r in records])) if records else 0.0,
        "draws_or_caps": sum(1 for r in records if r["winner"] is None),
        "behavior": {
            "counter_windows": counter_windows,
            "counter_casts": counter_casts,
            "counter_cast_rate_per_window": rate(counter_casts, counter_windows),
            "counter_first_window_rate": rate(first_window, counter_casts),
            "countered_mv_mean": float(np.mean(mana_values)) if mana_values else None,
            "countered_mv_le2_share": (
                rate(sum(1 for v in mana_values if v <= 2), len(mana_values))
            ),
            "countered_names": dict(sorted(countered_names.items())),
            "bolt_casts": bolt_casts,
            "bolt_biggest_rate_multi": rate(bolt_biggest, bolt_multi),
            "bolt_face_share": rate(bolt_face, bolt_casts),
            "bolt_own_share": rate(bolt_own, bolt_casts),
            "hold_windows": hold_windows,
            "hold_breaks": hold_breaks,
            "instant_holding_rate": (
                1.0 - hold_breaks / hold_windows if hold_windows else None
            ),
        },
    }


def micro_cells(games: int) -> list[dict[str, Any]]:
    search = lambda n: {"kind": "search", "sims": n}  # noqa: E731
    random_spec = {"kind": "random"}
    return [
        {
            "name": "aggro_mirror__search-64",
            "hero_spec": search(64),
            "villain_spec": search(64),
            "hero_deck": MICRO_AGGRO,
            "villain_deck": MICRO_AGGRO,
            "games": games,
        },
        {
            "name": "control_mirror__search-64",
            "hero_spec": search(64),
            "villain_spec": search(64),
            "hero_deck": MICRO_CONTROL,
            "villain_deck": MICRO_CONTROL,
            "games": games,
        },
        {
            "name": "control_vs_aggro__random",
            "hero_spec": random_spec,
            "villain_spec": random_spec,
            "hero_deck": MICRO_CONTROL,
            "villain_deck": MICRO_AGGRO,
            "games": games,
        },
        {
            "name": "control_vs_aggro__search-16",
            "hero_spec": search(16),
            "villain_spec": search(16),
            "hero_deck": MICRO_CONTROL,
            "villain_deck": MICRO_AGGRO,
            "games": games,
        },
        {
            "name": "control_vs_aggro__search-64",
            "hero_spec": search(64),
            "villain_spec": search(64),
            "hero_deck": MICRO_CONTROL,
            "villain_deck": MICRO_AGGRO,
            "games": games,
        },
        {
            "name": "control_vs_aggro__search-256",
            "hero_spec": search(256),
            "villain_spec": search(256),
            "hero_deck": MICRO_CONTROL,
            "villain_deck": MICRO_AGGRO,
            "games": games,
        },
    ]


def run_micro_suite(
    *,
    games: int,
    workers: int,
    base_seed: int,
    out_path: Path,
    only: set[str] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    ctx = mp.get_context("spawn")
    for cell_index, cell in enumerate(micro_cells(games)):
        name = cell["name"]
        if only is not None and name not in only:
            continue
        if name in results:
            print(f"[skip] {name}")
            continue
        start = time.perf_counter()
        chunks = []
        per_worker = cell["games"] // workers
        remainder = cell["games"] % workers
        offset = 0
        for w in range(workers):
            chunk = per_worker + (1 if w < remainder else 0)
            if chunk == 0:
                continue
            chunks.append(
                {
                    "hero_spec": cell["hero_spec"],
                    "villain_spec": cell["villain_spec"],
                    "hero_deck": cell["hero_deck"],
                    "villain_deck": cell["villain_deck"],
                    "num_games": chunk,
                    "seed": base_seed + cell_index * 7_777_777 + w * 1_000_000,
                    "game_offset": offset,
                }
            )
            offset += chunk
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            outputs = list(pool.map(_micro_worker, chunks))
        aggregate = aggregate_micro_cell(outputs)
        aggregate["wall_seconds"] = time.perf_counter() - start
        results[name] = aggregate
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"[done] {name}: hero win {aggregate['win_rate']:.3f} "
            f"[{aggregate['ci_lower']:.3f},{aggregate['ci_upper']:.3f}] "
            f"({aggregate['wall_seconds']:.0f}s)"
        )
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_agent_specs(text: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for token in text.split(","):
        token = token.strip()
        if token == "random":
            specs.append({"kind": "random"})
        elif token.startswith("search-"):
            specs.append({"kind": "search", "sims": int(token.split("-", 1)[1])})
        elif token.startswith("checkpoint:"):
            path = token.split(":", 1)[1]
            specs.append({"kind": "checkpoint", "path": path, "name": Path(path).stem})
        else:
            raise ValueError(f"unknown agent spec: {token}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scen = sub.add_parser("scenarios", help="run the competency scenarios")
    scen.add_argument(
        "--agents", type=str, default="random,search-16,search-64,search-256"
    )
    scen.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    scen.add_argument("--runs", type=int, default=100)
    scen.add_argument("--workers", type=int, default=4)
    scen.add_argument("--seed", type=int, default=0)
    scen.add_argument(
        "--out", type=str, default="reports/data/exp-09-scenarios.json"
    )

    micro = sub.add_parser("micro", help="run the micro-format matchups")
    micro.add_argument("--games", type=int, default=300)
    micro.add_argument("--workers", type=int, default=4)
    micro.add_argument("--seed", type=int, default=0)
    micro.add_argument("--only", type=str, default=None)
    micro.add_argument("--out", type=str, default="reports/data/exp-09-micro.json")

    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")

    if args.command == "scenarios":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_scenario_suite(
            [s.strip() for s in args.scenarios.split(",")],
            parse_agent_specs(args.agents),
            runs=args.runs,
            workers=args.workers,
            base_seed=args.seed,
            out_path=out_path,
        )
    elif args.command == "micro":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_micro_suite(
            games=args.games,
            workers=args.workers,
            base_seed=args.seed,
            out_path=out_path,
            only=set(args.only.split(",")) if args.only else None,
        )


if __name__ == "__main__":
    main()
