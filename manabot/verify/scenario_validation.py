"""Revalidate historical control fixtures on the installed rules runtime.

``validate_scenarios`` executes contrasting scripted lines and verifies resolved
outcomes, not just historical tracker labels. Fixtures retain their original
cards and custom setups: they are not interchangeable with a selected-match
checkpoint. These injected, bounded trajectories are mechanism diagnostics,
not replayable full-game arena evidence or estimates of optimal strategy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from manabot.infra.hypers import MatchHypers
from manabot.verify.competency import (
    SCENARIOS,
    SPACE_ATTACKER,
    SPACE_BLOCKER,
    SPACE_PRIORITY,
    SPACE_TARGET,
    ZONE_GRAVEYARD,
    Scenario,
    ScriptedVillain,
    _action_casts,
    _battlefield_pairs,
    _find_cast_action,
    _hero_view,
    _pass_index,
    _stack_names,
    _zone_names,
    build_scenario_env,
)
import managym


@dataclass(frozen=True)
class ScenarioLineEvidence:
    """One bounded, legal script attempt; failure is retained rather than dropped."""

    seed: int
    line: Literal["reference", "contrast"]
    steps: int
    expected_outcome: bool
    outcome: str
    error: str | None


@dataclass(frozen=True)
class ScenarioValidation:
    scenario: str
    world: str
    content_sha256: str | None
    fixture_source_sha256: str
    validator_source_sha256: str
    selected_setup_supported: bool
    status: Literal["validated_fixture", "failed_fixture", "unsupported_selected_setup"]
    premise: str
    limitation: str
    attempts: tuple[ScenarioLineEvidence, ...]


class _Hero:
    """Fixed diagnostic scripts, using only the current actor's visible fields."""

    def __init__(self, name: str, reference: bool) -> None:
        self.name = name
        self.reference = reference
        self.blocks: dict[int, int] = {}

    def act(self, raw: managym.Observation) -> int:
        kind = int(raw.action_space.action_space_type)
        actions = raw.action_space.actions
        if kind == SPACE_PRIORITY:
            card: str | None = None
            stack = _stack_names(raw)
            _, villain = _hero_view(raw)
            enemies = _battlefield_pairs(raw, villain)
            if self.name.startswith("s1"):
                if not self.reference or "Shivan Dragon" in stack:
                    card = "Counterspell"
            elif self.name.startswith("s2"):
                if (
                    not self.reference
                    or sum(bool(c.card_types.is_creature) for c, _ in enemies) >= 4
                ):
                    card = "Pyroclasm"
            elif self.name.startswith("s3"):
                if not self.reference or any(
                    c.name == "Earth King's Lieutenant" for c, _ in enemies
                ):
                    card = "Lightning Bolt"
            elif self.name.startswith("s5"):
                card = "It'll Quench Ya!" if self.reference else "Otter-Penguin"
            if card is not None:
                index = _find_cast_action(raw, card)
                if index is not None:
                    return index
            return _pass_index(raw)
        if kind == SPACE_TARGET and self.name.startswith("s3"):
            _, villain = _hero_view(raw)
            target = (
                "Earth King's Lieutenant"
                if self.reference
                else "Invasion Reinforcements"
            )
            ids = {
                int(p.id)
                for c, p in _battlefield_pairs(raw, villain)
                if c.name == target
            }
            for index, action in enumerate(actions):
                if action.focus and int(action.focus[0]) in ids:
                    return index
        if kind == SPACE_ATTACKER:
            if self.name.startswith("s4"):
                if not self.reference:
                    return 0
                focus = int(actions[0].focus[0]) if actions[0].focus else -1
                hero, _ = _hero_view(raw)
                flying = any(
                    c.name == "Wind Drake" and int(p.id) == focus
                    for c, p in _battlefield_pairs(raw, hero)
                )
                return 0 if flying else len(actions) - 1
            return len(actions) - 1
        if kind == SPACE_BLOCKER and self.name.startswith("s4") and self.reference:
            turn = int(raw.turn.turn_number)
            index = self.blocks.get(turn, 0)
            self.blocks[turn] = index + 1
            return min(index, len(actions) - 1)
        if kind == SPACE_TARGET:
            return 0
        return len(actions) - 1


def _outcome(
    scenario: Scenario,
    raw: managym.Observation,
    reference: bool,
    cast: str | None,
    creatures_at_cast: int,
) -> tuple[bool, str] | None:
    """Check actual resolution; cast intent alone never establishes removal."""
    _, villain = _hero_view(raw)
    board = Counter(str(c.name) for c, _ in _battlefield_pairs(raw, villain))
    grave = Counter(_zone_names(raw, villain, ZONE_GRAVEYARD))
    if scenario.name.startswith("s1"):
        if grave["Shivan Dragon"]:
            return reference, "bomb countered"
        if board["Shivan Dragon"]:
            return not reference, "bomb resolved after bait"
    elif (
        scenario.name.startswith("s2") and cast == "Pyroclasm" and not _stack_names(raw)
    ):
        if grave["Gray Ogre"] >= creatures_at_cast:
            expected = creatures_at_cast >= 4 if reference else creatures_at_cast == 2
            return expected, f"wipe resolved: {creatures_at_cast} creatures removed"
    elif (
        scenario.name.startswith("s3")
        and cast == "Lightning Bolt"
        and not _stack_names(raw)
    ):
        if reference and grave["Earth King's Lieutenant"]:
            return True, "Lieutenant killed by held removal"
        if (
            not reference
            and grave["Invasion Reinforcements"]
            and board["Earth King's Lieutenant"]
        ):
            return True, "removal spent on decoy; Lieutenant survived"
    elif scenario.name.startswith("s4") and raw.game_over:
        hero_won = bool(raw.won) == (int(raw.agent.player_index) == 0)
        return hero_won == reference, "hero won" if hero_won else "hero lost"
    elif scenario.name.startswith("s5"):
        if grave["Craw Wurm"]:
            return reference, "held mana countered Wurm"
        if board["Craw Wurm"]:
            return not reference, "tap-out allowed Wurm to resolve"
    return None


def _run_line(
    scenario: Scenario, seed: int, reference: bool, max_steps: int
) -> tuple[ScenarioLineEvidence, str | None]:
    line: Literal["reference", "contrast"] = "reference" if reference else "contrast"
    steps = 0
    content: str | None = None
    env = None
    try:
        env, _, raw = build_scenario_env(scenario, None, seed)
        content = hashlib.sha256(
            json.dumps(env._engine.content_pack_manifest(), sort_keys=True).encode()
        ).hexdigest()
        # State injection must still establish the original strategic premise.
        if (
            int(raw.agent.player_index) != 0
            or int(raw.agent.life) != scenario.hero_life
            or int(raw.opponent.life) != scenario.villain_life
            or Counter(_zone_names(raw, "agent", int(managym.ZoneEnum.HAND)))
            != Counter(scenario.hero_hand)
            or Counter(str(c.name) for c, _ in _battlefield_pairs(raw, "agent"))
            != Counter(scenario.hero_battlefield)
            or Counter(str(c.name) for c, _ in _battlefield_pairs(raw, "opponent"))
            != Counter(scenario.villain_battlefield)
        ):
            raise ValueError(
                "injected position no longer matches the historical premise"
            )
        hero = _Hero(scenario.name, reference)
        villain = ScriptedVillain(scenario.villain_casts)
        cast: str | None = None
        creatures_at_cast = 0
        while steps < max_steps:
            raw = env.last_raw_obs
            resolved = _outcome(scenario, raw, reference, cast, creatures_at_cast)
            if resolved is not None:
                expected, outcome = resolved
                return ScenarioLineEvidence(
                    seed, line, steps, expected, outcome, None
                ), content
            if raw.game_over or int(raw.turn.turn_number) > scenario.max_turns:
                break
            is_hero = int(raw.agent.player_index) == 0
            action = hero.act(raw) if is_hero else villain.act(raw)
            if not 0 <= action < len(raw.action_space.actions):
                raise ValueError("script selected a nonlegal action index")
            if is_hero:
                spell = _action_casts(raw, action)
                if spell is not None:
                    cast = spell
                    _, side = _hero_view(raw)
                    creatures_at_cast = sum(
                        bool(c.card_types.is_creature)
                        for c, _ in _battlefield_pairs(raw, side)
                    )
            env.step(action)
            steps += 1
        return ScenarioLineEvidence(
            seed,
            line,
            steps,
            False,
            "expected resolution unavailable within fixture bound",
            None,
        ), content
    except Exception as error:
        return ScenarioLineEvidence(
            seed,
            line,
            steps,
            False,
            "attempt failed",
            f"{type(error).__name__}: {error}",
        ), content
    finally:
        if env is not None:
            env.close()


def validate_scenarios(
    *,
    seeds: tuple[int, ...] = (2,),
    selected_match: MatchHypers | None = None,
    max_steps: int = 1000,
) -> tuple[ScenarioValidation, ...]:
    """Run both lines per seed, retaining errors and exact source/content identities.

    Omitting ``selected_match`` validates only the custom historical fixtures.
    A selected setup must equal the fixture decks (including empty sideboards);
    successful custom-deck scripts cannot admit an Allies/Lessons checkpoint.
    This does not prove global strategic optimality or historical score parity.
    """
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("distinct nonnegative fixture seeds are required")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    fixture_digest = hashlib.sha256(
        Path(__file__).with_name("competency.py").read_bytes()
    ).hexdigest()
    validator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records: list[ScenarioValidation] = []
    for scenario in SCENARIOS.values():
        attempts: list[ScenarioLineEvidence] = []
        content: str | None = None
        for seed in seeds:
            for reference in (True, False):
                attempt, digest = _run_line(scenario, seed, reference, max_steps)
                attempts.append(attempt)
                if digest is not None:
                    if content is not None and content != digest:
                        raise RuntimeError("scenario content changed during validation")
                    content = digest
        supported = selected_match is None or (
            selected_match.hero_deck == scenario.hero_deck
            and selected_match.villain_deck == scenario.villain_deck
            and not selected_match.hero_sideboard
            and not selected_match.villain_sideboard
        )
        status: Literal[
            "validated_fixture", "failed_fixture", "unsupported_selected_setup"
        ]
        status = (
            "validated_fixture"
            if all(a.expected_outcome and a.error is None for a in attempts)
            else "failed_fixture"
        )
        if not supported:
            status = "unsupported_selected_setup"
        records.append(
            ScenarioValidation(
                scenario.name,
                str(managym.WORLD_VERSION),
                content,
                fixture_digest,
                validator_digest,
                supported,
                status,
                scenario.correct_line,
                "Injected custom-deck mechanism check, not full-game replay or an optimal-policy proof. "
                "Selected-match checkpoint scoring is unsupported when its setup differs.",
                tuple(attempts),
            )
        )
    return tuple(records)
