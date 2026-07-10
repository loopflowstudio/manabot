"""
server.py
FastAPI server for interactive managym play over WebSocket.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

# Local imports
from manabot.env.observation import (
    ActionEnum,
    ActionSpaceEnum,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
)
import managym

from . import trace as trace_store, villain as villain_module
from .trace import GameConfig, Trace, TraceEvent
from .villain import VillainPolicy, build_villain_policy

# Mirrors of manabot.verify.util deck constants (copied so the server does
# not import torch at startup; tests assert they stay in sync).
INTERACTIVE_DECK = {
    "Island": 12,
    "Mountain": 12,
    "Grey Ogre": 6,
    "Wind Drake": 6,
    "Man-o'-War": 4,
    "Raging Goblin": 4,
    "Lightning Bolt": 6,
    "Counterspell": 4,
    "Ancestral Recall": 3,
    "Pyroclasm": 3,
}
UR_LESSONS_DECK = {
    "Island": 9,
    "Mountain": 8,
    "Tiger-Seal": 2,
    "Otter-Penguin": 2,
    "Fire Nation Cadets": 2,
    "First-Time Flyer": 2,
    "Forecasting Fortune Teller": 1,
    "Dragonfly Swarm": 1,
    "Firebending Lesson": 4,
    "Igneous Inspiration": 2,
    "Pop Quiz": 2,
    "Divide by Zero": 2,
    "It'll Quench Ya!": 2,
    "Accumulate Wisdom": 2,
}
GW_ALLIES_DECK = {
    "Plains": 9,
    "Forest": 8,
    "Water Tribe Rallier": 2,
    "Invasion Reinforcements": 2,
    "Compassionate Healer": 2,
    "Earth Kingdom Jailer": 2,
    "White Lotus Reinforcements": 2,
    "Earth King's Lieutenant": 2,
    "Kyoshi Warriors": 2,
    "Badgermole Cub": 2,
    "Suki, Kyoshi Warrior": 1,
    "South Pole Voyager": 1,
    "Allies at Last": 2,
    "Yip Yip!": 1,
    "Fancy Footwork": 2,
}

# Decks selectable by name over the wire (new_game.config hero_deck /
# villain_deck may be one of these keys instead of a {card: count} object).
NAMED_DECKS: dict[str, dict[str, int]] = {
    "interactive": INTERACTIVE_DECK,
    "ur_lessons": UR_LESSONS_DECK,
    "gw_allies": GW_ALLIES_DECK,
}
DECK_DISPLAY_NAMES = {
    "interactive": "Interactive",
    "ur_lessons": "UR Lessons",
    "gw_allies": "GW Allies",
    "custom": "Custom",
}
# Default matchup: the Milestone-1 two-deck slice, UR as hero vs GW villain.
DEFAULT_HERO_DECK_NAME = "ur_lessons"
DEFAULT_VILLAIN_DECK_NAME = "gw_allies"
# Backwards-compatible alias (tests and older callers).
DEFAULT_DECK = INTERACTIVE_DECK

MAX_AUTOPLAY_STEPS = 1024
HERO_PLAYER_INDEX = 0
VILLAIN_TYPES = {"passive", "random", "search", "checkpoint"}

# MTGO-style priority stops. Stop keys are the human-facing step names; they
# map onto the engine's StepEnum names (serialize_observation reports the same
# strings in turn.step). Steps without a stop key (untap, cleanup, end of
# combat) can never be stopped on and always auto-pass unless the stack rule
# fires.
STOP_STEP_TO_ENGINE_STEP = {
    "upkeep": "BEGINNING_UPKEEP",
    "draw": "BEGINNING_DRAW",
    "main1": "PRECOMBAT_MAIN_STEP",
    "begin_combat": "COMBAT_BEGIN",
    "declare_attackers": "COMBAT_DECLARE_ATTACKERS",
    "declare_blockers": "COMBAT_DECLARE_BLOCKERS",
    "combat_damage": "COMBAT_DAMAGE",
    "main2": "POSTCOMBAT_MAIN_STEP",
    "end_step": "ENDING_END",
}
ENGINE_STEP_TO_STOP_STEP = {
    engine: stop for stop, engine in STOP_STEP_TO_ENGINE_STEP.items()
}
STOP_SIDES = ("my", "opponent")
# Defaults chosen for the INTERACTIVE_DECK matchup: act in your own main
# phases, hold instants at the opponent's end step, and always stop when the
# stack is non-empty (that is how you get to counter things).
DEFAULT_STOPS: dict[str, list[str]] = {
    "my": ["main1", "main2"],
    "opponent": ["end_step"],
}
ACTION_LABELS = {
    "PRIORITY_PLAY_LAND": "Play land",
    "PRIORITY_CAST_SPELL": "Cast spell",
    "PRIORITY_ACTIVATE_ABILITY": "Activate ability",
    "DECLARE_ATTACKER": "Declare attacker",
    "DECLARE_BLOCKER": "Declare blocker",
    "CHOOSE_TARGET": "Choose target",
}

app = FastAPI(title="manabot-gui")
SESSION_TTL = timedelta(minutes=15)
SESSION_EXPIRED_END_REASON = "session_expired"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionRecord:
    session_id: str
    resume_token: str
    game: "GameSession"
    websocket: WebSocket | None = None
    expires_at: datetime = field(default_factory=lambda: _now_utc() + SESSION_TTL)

    def touch(self) -> None:
        self.expires_at = _now_utc() + SESSION_TTL


SESSION_REGISTRY: dict[str, SessionRecord] = {}


def _enum_name(enum_type, value: Any) -> str:
    try:
        return enum_type(int(value)).name
    except Exception:
        return str(int(value))


def _serialize_card(card: managym.Card) -> dict[str, Any]:
    return {
        "id": int(card.id),
        "registry_key": int(card.registry_key),
        "name": card.name,
        "zone": _enum_name(ZoneEnum, card.zone),
        "owner_id": int(card.owner_id),
        "power": int(card.power),
        "toughness": int(card.toughness),
        "mana_value": int(card.mana_cost.mana_value),
        "types": {
            "is_creature": bool(card.card_types.is_creature),
            "is_land": bool(card.card_types.is_land),
            "is_spell": bool(card.card_types.is_spell),
            "is_artifact": bool(card.card_types.is_artifact),
            "is_enchantment": bool(card.card_types.is_enchantment),
            "is_planeswalker": bool(card.card_types.is_planeswalker),
            "is_battle": bool(card.card_types.is_battle),
        },
    }


def _serialize_permanent(
    permanent: managym.Permanent,
    card: managym.Card | None,
) -> dict[str, Any]:
    return {
        "id": int(permanent.id),
        "name": card.name if card else None,
        "controller_id": int(permanent.controller_id),
        "tapped": bool(permanent.tapped),
        "damage": int(permanent.damage),
        "summoning_sick": bool(permanent.is_summoning_sick),
        "power": int(card.power) if card else None,
        "toughness": int(card.toughness) if card else None,
    }


def _serialize_player(
    player: managym.Player,
    cards: list[managym.Card],
    permanents: list[managym.Permanent],
) -> dict[str, Any]:
    grouped_cards: dict[str, list[dict[str, Any]]] = {
        "HAND": [],
        "GRAVEYARD": [],
        "EXILE": [],
        "STACK": [],
    }
    battlefield_cards: list[managym.Card] = []
    for card in cards:
        zone_name = _enum_name(ZoneEnum, card.zone)
        if zone_name == "BATTLEFIELD":
            battlefield_cards.append(card)
        elif zone_name in grouped_cards:
            grouped_cards[zone_name].append(_serialize_card(card))

    # Permanents and battlefield cards correspond 1:1 in order.
    # Fall back to None if the lists don't align.
    serialized_permanents = []
    for i, permanent in enumerate(permanents):
        card = battlefield_cards[i] if i < len(battlefield_cards) else None
        serialized_permanents.append(_serialize_permanent(permanent, card))

    zone_counts = {
        _enum_name(ZoneEnum, index): int(count)
        for index, count in enumerate(player.zone_counts)
    }

    return {
        "player_index": int(player.player_index),
        "id": int(player.id),
        "is_active": bool(player.is_active),
        "is_agent": bool(player.is_agent),
        "life": int(player.life),
        "zone_counts": zone_counts,
        "library_count": zone_counts.get("LIBRARY", 0),
        "hand": grouped_cards["HAND"],
        "graveyard": grouped_cards["GRAVEYARD"],
        "exile": grouped_cards["EXILE"],
        "stack": grouped_cards["STACK"],
        "battlefield": serialized_permanents,
    }


def serialize_observation(obs: managym.Observation) -> dict[str, Any]:
    return {
        "game_over": bool(obs.game_over),
        "won": bool(obs.won),
        "turn": {
            "turn_number": int(obs.turn.turn_number),
            "phase": _enum_name(PhaseEnum, obs.turn.phase),
            "step": _enum_name(StepEnum, obs.turn.step),
            "active_player_id": int(obs.turn.active_player_id),
            "agent_player_id": int(obs.turn.agent_player_id),
        },
        "agent": _serialize_player(obs.agent, obs.agent_cards, obs.agent_permanents),
        "opponent": _serialize_player(
            obs.opponent,
            obs.opponent_cards,
            obs.opponent_permanents,
        ),
    }


def _player_label(player: managym.Player) -> str:
    return "Hero" if int(player.player_index) == HERO_PLAYER_INDEX else "Villain"


def _permanent_names(
    cards: list[managym.Card],
    permanents: list[managym.Permanent],
) -> dict[int, str]:
    """Map permanent ids to card names.

    Permanents carry their own object ids (distinct from card ids), but the
    observation lists permanents and battlefield cards in the same order —
    the same alignment _serialize_player relies on.
    """
    battlefield_cards = [
        card
        for card in cards
        if _enum_name(ZoneEnum, card.zone) == "BATTLEFIELD"
    ]
    names: dict[int, str] = {}
    for index, permanent in enumerate(permanents):
        if index < len(battlefield_cards):
            names[int(permanent.id)] = battlefield_cards[index].name
    return names


def _build_id_to_name(obs: managym.Observation) -> dict[int, str]:
    """Map object ids (players, cards, permanents) to display names."""
    names: dict[int, str] = {
        int(obs.agent.id): _player_label(obs.agent),
        int(obs.opponent.id): _player_label(obs.opponent),
    }
    names.update(
        {int(card.id): card.name for card in [*obs.agent_cards, *obs.opponent_cards]}
    )
    names.update(_permanent_names(obs.agent_cards, obs.agent_permanents))
    names.update(_permanent_names(obs.opponent_cards, obs.opponent_permanents))
    return names


def _format_action(
    action: managym.Action,
    names: dict[int, str],
    space_kind: str = "",
) -> str:
    """Render a legal action in Magic terms with card names.

    Focus semantics (managym observation.rs action_focus):
      PLAY_LAND / CAST_SPELL: [card_id]
      DECLARE_ATTACKER:       [attacker_permanent_id]
      DECLARE_BLOCKER:        [blocker_permanent_id, attacker_permanent_id?]
                              (one entry means "this creature does not block")
      CHOOSE_TARGET:          [player_id or permanent_id]
      SCRY_* / SELECT_CARD:   [card_id] (library cards revealed by the
                              pending decision are in the observation)
      TAP_FOR_COST:           [permanent_id]
      DECLINE/PAY/MODE:       [resolving or cast card_id]
      PASS_PRIORITY:          []
    """
    action_name = _enum_name(ActionEnum, action.action_type)
    focus_names = [names.get(int(value)) for value in action.focus]
    first = focus_names[0] if focus_names else None

    if action_name == "PRIORITY_PASS_PRIORITY":
        return "Pass priority"
    if action_name == "PRIORITY_PLAY_LAND" and first:
        return f"Play {first}"
    if action_name == "PRIORITY_CAST_SPELL" and first:
        return f"Cast {first}"
    if action_name == "DECLARE_ATTACKER" and first:
        return f"Attack with {first}"
    if action_name == "DECLARE_BLOCKER" and first:
        if len(focus_names) >= 2 and focus_names[1]:
            return f"Block {focus_names[1]} with {first}"
        return f"{first}: do not block"
    if action_name == "CHOOSE_TARGET" and first:
        return f"Target {first}"
    # Mid-resolution decision actions (scry / look-and-select / learn /
    # pay-or-not / modal / waterbend) in Magic terms.
    if action_name == "SCRY_KEEP" and first:
        return f"Keep {first} on top"
    if action_name == "SCRY_BOTTOM" and first:
        return f"Put {first} on the bottom"
    if action_name == "SELECT_CARD" and first:
        if space_kind == "DISCARD_THEN_DRAW":
            return f"Discard {first}, then draw a card"
        return f"Put {first} into your hand"
    if action_name == "DECLINE_CHOICE":
        if space_kind == "DISCARD_THEN_DRAW":
            return "Keep your hand (do not discard)"
        return "Decline"
    if action_name == "PAY_COST":
        return f"Pay the cost ({first})" if first else "Pay the cost"
    if action_name == "CHOOSE_MODE":
        return f"Choose a mode ({first})" if first else "Choose a mode"
    if action_name == "TAP_FOR_COST" and first:
        return f"Tap {first} to help pay"

    label = ACTION_LABELS.get(action_name, action_name)
    if first:
        return f"{label}: {first}"
    return label


def describe_actions(obs: managym.Observation) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    names = _build_id_to_name(obs)
    space_kind = _enum_name(ActionSpaceEnum, int(obs.action_space.action_space_type))
    for index, action in enumerate(obs.action_space.actions):
        focus = [int(value) for value in action.focus]
        results.append(
            {
                "index": index,
                "type": _enum_name(ActionEnum, action.action_type),
                "focus": focus,
                "description": _format_action(action, names, space_kind),
            }
        )
    return results


def hero_view(obs: managym.Observation) -> dict[str, Any]:
    """Serialize an observation for the human player.

    Two guarantees, regardless of whose perspective the engine observation is
    from (at game over it can be the villain's):
      1. ``agent`` is always the hero and ``opponent`` always the villain.
      2. The villain's hand is redacted (libraries are never serialized).
    """
    data = serialize_observation(obs)
    if int(data["agent"]["player_index"]) != HERO_PLAYER_INDEX:
        data["agent"], data["opponent"] = data["opponent"], data["agent"]
        data["won"] = bool(obs.game_over) and not bool(obs.won)

    trace_store.redact_observation(data)
    return data


def _winner_for_hero(obs: managym.Observation) -> int | None:
    if not obs.game_over:
        return None

    agent_is_hero = int(obs.agent.player_index) == HERO_PLAYER_INDEX
    if bool(obs.won):
        return 0 if agent_is_hero else 1
    return 1 if agent_is_hero else 0


def _is_hero_turn(obs: managym.Observation) -> bool:
    return int(obs.agent.player_index) == HERO_PLAYER_INDEX


def _is_priority_space(obs: managym.Observation) -> bool:
    return obs.action_space.action_space_type == managym.ActionSpaceEnum.PRIORITY


def _hero_is_active_player(obs: managym.Observation) -> bool:
    """Whether the hero is the active player (it is 'my' turn).

    Perspective-independent: finds the hero side by player_index rather than
    assuming ``obs.agent`` is the hero.
    """
    hero = (
        obs.agent
        if int(obs.agent.player_index) == HERO_PLAYER_INDEX
        else obs.opponent
    )
    return int(obs.turn.active_player_id) == int(hero.id)


def _pass_priority_index(actions: list[dict[str, Any]]) -> int | None:
    for action in actions:
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return int(action["index"])
    return None


def _parse_stops(value: Any) -> dict[str, list[str]]:
    """Validate a stops object: {"my": [step, ...], "opponent": [step, ...]}."""
    if value is None:
        return {side: list(steps) for side, steps in DEFAULT_STOPS.items()}
    if not isinstance(value, dict):
        raise ValueError('stops must be an object of {"my"/"opponent": [steps]}.')

    unknown_sides = set(value) - set(STOP_SIDES)
    if unknown_sides:
        raise ValueError(
            "stops sides must be 'my' or 'opponent'; got: "
            + ", ".join(sorted(str(side) for side in unknown_sides))
        )

    stops: dict[str, list[str]] = {}
    for side in STOP_SIDES:
        steps = value.get(side, [])
        if not isinstance(steps, list):
            raise ValueError(f"stops.{side} must be a list of step names.")
        normalized: list[str] = []
        for step in steps:
            if step not in STOP_STEP_TO_ENGINE_STEP:
                raise ValueError(
                    f"Unknown stop step: {step!r}. Valid steps: "
                    + ", ".join(STOP_STEP_TO_ENGINE_STEP)
                )
            if step not in normalized:
                normalized.append(step)
        stops[side] = normalized
    return stops


def _parse_flag(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _normalize_deck(value: Any, fallback_name: str) -> tuple[dict[str, int], str]:
    """Resolve a wire deck config to ``(deck, deck_name)``.

    Accepts a named deck (one of NAMED_DECKS), a {card: count} object
    (recorded as "custom"), or None (the named fallback).
    """
    if value is None:
        return dict(NAMED_DECKS[fallback_name]), fallback_name
    if isinstance(value, str):
        if value not in NAMED_DECKS:
            raise ValueError(
                "Unknown deck name: "
                + repr(value)
                + ". Valid decks: "
                + ", ".join(sorted(NAMED_DECKS))
            )
        return dict(NAMED_DECKS[value]), value
    if not isinstance(value, dict):
        raise ValueError(
            "Deck config must be a deck name or an object of {card_name: count}."
        )

    deck: dict[str, int] = {}
    for card_name, count in value.items():
        if not isinstance(card_name, str):
            raise ValueError("Deck card names must be strings.")
        try:
            normalized_count = int(count)
        except Exception as exc:
            raise ValueError(
                f"Deck count for '{card_name}' must be an integer."
            ) from exc
        if normalized_count < 0:
            raise ValueError(f"Deck count for '{card_name}' must be non-negative.")
        deck[card_name] = normalized_count

    return deck, "custom"


def _parse_game_config(config: Any) -> GameConfig:
    data = config or {}
    if not isinstance(data, dict):
        raise ValueError("new_game.config must be an object.")

    villain_type = data.get("villain_type", "search")
    if villain_type not in VILLAIN_TYPES:
        raise ValueError(
            "villain_type must be one of: " + ", ".join(sorted(VILLAIN_TYPES)) + "."
        )

    seed_value = data.get("seed")
    if seed_value is None:
        seed: int | None = None
    else:
        try:
            seed = int(seed_value)
        except Exception as exc:
            raise ValueError("seed must be an integer.") from exc

    villain_sims: int | None = None
    if villain_type == "search":
        sims_value = data.get("villain_sims", villain_module.DEFAULT_SEARCH_SIMS)
        try:
            villain_sims = int(sims_value)
        except Exception as exc:
            raise ValueError("villain_sims must be an integer.") from exc
        if villain_sims < 1 or villain_sims > villain_module.MAX_SEARCH_SIMS:
            raise ValueError(
                f"villain_sims must be between 1 and {villain_module.MAX_SEARCH_SIMS}."
            )

    villain_checkpoint: str | None = None
    villain_deterministic = False
    if villain_type == "checkpoint":
        checkpoint_value = data.get("villain_checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError(
                "checkpoint villain requires a 'villain_checkpoint' path."
            )
        if not Path(checkpoint_value).is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint_value}")
        villain_checkpoint = checkpoint_value
        villain_deterministic = bool(data.get("villain_deterministic", False))

    hero_deck, hero_deck_name = _normalize_deck(
        data.get("hero_deck"), DEFAULT_HERO_DECK_NAME
    )
    villain_deck, villain_deck_name = _normalize_deck(
        data.get("villain_deck"), DEFAULT_VILLAIN_DECK_NAME
    )

    return GameConfig(
        hero_deck=hero_deck,
        villain_deck=villain_deck,
        hero_deck_name=hero_deck_name,
        villain_deck_name=villain_deck_name,
        villain_type=villain_type,
        seed=seed,
        villain_sims=villain_sims,
        villain_checkpoint=villain_checkpoint,
        villain_deterministic=villain_deterministic,
        stops=_parse_stops(data.get("stops")),
        stop_on_stack=_parse_flag(data, "stop_on_stack", True),
        auto_pass=_parse_flag(data, "auto_pass", True),
    )


class GameSession:
    def __init__(self, trace_dir: Path | None = None):
        self.trace_dir = trace_dir or trace_store.TRACES_DIR
        self.env: managym.Env | None = None
        self.obs: managym.Observation | None = None
        self.villain_policy: VillainPolicy | None = None
        self.trace: Trace | None = None
        self.trace_id: str | None = None
        # Display names for the current matchup, echoed on every payload.
        self.deck_names: dict[str, str] | None = None
        self._trace_saved = False
        self._pending_villain_log: list[str] = []
        # Priority-stop state (see STOP_STEP_TO_ENGINE_STEP / DEFAULT_STOPS).
        self.stops: dict[str, list[str]] = {
            side: list(steps) for side, steps in DEFAULT_STOPS.items()
        }
        self.stop_on_stack = True
        self.auto_pass = True
        # F6: while set, every hero priority window auto-passes (even at stops,
        # even through a non-empty stack) until this turn number ends or a
        # non-priority decision surfaces.
        self._f6_turn: int | None = None
        # Hero priority windows auto-passed since the last surfaced message.
        self._auto_passed_since_surface = 0

    def new_game(self, raw_config: Any) -> dict[str, Any]:
        if self.trace is not None:
            self.close(end_reason="new_game")

        config = _parse_game_config(raw_config)
        try:
            self.villain_policy = build_villain_policy(config)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to build villain policy: {exc}") from exc

        seed = config.seed if config.seed is not None else 0
        self.env = managym.Env(seed=seed)

        player_configs = [
            managym.PlayerConfig("Hero", config.hero_deck),
            managym.PlayerConfig("Villain", config.villain_deck),
        ]
        self.obs, _ = self.env.reset(player_configs)

        self.deck_names = {
            "hero": DECK_DISPLAY_NAMES.get(config.hero_deck_name, "Custom"),
            "villain": DECK_DISPLAY_NAMES.get(config.villain_deck_name, "Custom"),
        }
        self.trace = Trace(
            config=config,
            events=[],
            final_observation={},
            winner=None,
            end_reason="disconnect",
            timestamp=trace_store.utc_now_iso(),
        )
        self._trace_saved = False
        self.trace_id = None
        self._pending_villain_log = []
        self.stops = {
            side: list(steps) for side, steps in (config.stops or DEFAULT_STOPS).items()
        }
        self.stop_on_stack = config.stop_on_stack
        self.auto_pass = config.auto_pass
        self._f6_turn = None
        self._auto_passed_since_surface = 0

        self._advance()
        return self._wire_message()

    def hero_action(self, raw_index: Any) -> dict[str, Any]:
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if self.obs.game_over:
            raise ValueError("Game is already over. Start a new game.")
        if not _is_hero_turn(self.obs):
            raise ValueError("Cannot accept hero action: waiting on villain auto-play.")

        try:
            action_index = int(raw_index)
        except Exception as exc:
            raise ValueError("Action index must be an integer.") from exc

        actions = describe_actions(self.obs)
        if action_index < 0 or action_index >= len(actions):
            raise ValueError(f"Action index out of range: {action_index}")

        self._step_and_record(actor="hero", action_index=action_index, actions=actions)
        self._advance()
        return self._wire_message()

    def set_stops(
        self,
        raw_stops: Any,
        raw_stop_on_stack: Any,
        raw_auto_pass: Any,
    ) -> dict[str, Any]:
        """Update the stop configuration mid-game; missing fields keep their
        current value. If the hero is currently parked at a window that no
        longer stops, fast-forward immediately."""
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")

        if raw_stops is not None:
            self.stops = _parse_stops(raw_stops)
        if raw_stop_on_stack is not None:
            if not isinstance(raw_stop_on_stack, bool):
                raise ValueError("stop_on_stack must be a boolean.")
            self.stop_on_stack = raw_stop_on_stack
        if raw_auto_pass is not None:
            if not isinstance(raw_auto_pass, bool):
                raise ValueError("auto_pass must be a boolean.")
            self.auto_pass = raw_auto_pass

        self._advance()
        return self._wire_message()

    def pass_turn(self) -> dict[str, Any]:
        """F6: yield every hero priority window (through stops and stack)
        until the current turn ends or a non-priority decision surfaces."""
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if self.obs.game_over:
            raise ValueError("Game is already over. Start a new game.")

        if _is_hero_turn(self.obs) and _is_priority_space(self.obs):
            self._f6_turn = int(self.obs.turn.turn_number)
            self._advance()
        return self._wire_message()

    def current_message(self) -> dict[str, Any]:
        if self.obs is None:
            raise ValueError("No active game session. Send new_game first.")
        return self._wire_message()

    def _step_and_record(
        self,
        actor: str,
        action_index: int,
        actions: list[dict[str, Any]],
        auto: bool = False,
    ) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            raise RuntimeError("Cannot step without an active game.")

        observation = serialize_observation(self.obs)
        action_description = actions[action_index]["description"]
        next_obs, reward, _, _, _ = self.env.step(action_index)
        if actor == "villain":
            self._pending_villain_log.append(f"Villain: {action_description}")
        self.trace.events.append(
            TraceEvent(
                actor=actor,
                observation=observation,
                actions=actions,
                action=action_index,
                action_description=action_description,
                reward=float(reward),
                auto=auto,
            )
        )
        self.obs = next_obs

    def _auto_play_villain(self) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            return
        if self.villain_policy is None:
            raise RuntimeError("Villain policy not initialized.")

        steps = 0
        while not self.obs.game_over and not _is_hero_turn(self.obs):
            if steps >= MAX_AUTOPLAY_STEPS:
                raise RuntimeError("Villain auto-play exceeded safety step limit.")
            steps += 1

            actions = describe_actions(self.obs)
            action_index = int(self.villain_policy(self.env, self.obs))
            if action_index < 0 or action_index >= len(actions):
                raise RuntimeError(
                    f"Villain policy selected invalid action index: {action_index}"
                )
            self._step_and_record(
                actor="villain", action_index=action_index, actions=actions
            )

    def _advance(self) -> None:
        """Run the game forward to the next point the hero must see.

        Alternates villain auto-play with hero auto-passes until the game is
        over or a hero decision surfaces per the stop rules. Bounded so a
        stuck engine cannot spin the server forever.
        """
        if self.env is None or self.obs is None or self.trace is None:
            return

        hero_passes = 0
        while True:
            self._auto_play_villain()
            if self.obs.game_over:
                self._f6_turn = None
                return

            # Hero is to act from here on.
            self._maybe_clear_f6()
            if self._should_surface():
                return

            if hero_passes >= MAX_AUTOPLAY_STEPS:
                raise RuntimeError("Hero auto-pass exceeded safety step limit.")
            hero_passes += 1

            actions = describe_actions(self.obs)
            pass_index = _pass_priority_index(actions)
            if pass_index is None:
                # Defensive: _should_surface only lets pure priority windows
                # through, which always contain a pass action.
                return
            self._step_and_record(
                actor="hero",
                action_index=pass_index,
                actions=actions,
                auto=True,
            )
            self._auto_passed_since_surface += 1

    def _maybe_clear_f6(self) -> None:
        if self._f6_turn is None or self.obs is None:
            return
        if int(self.obs.turn.turn_number) != self._f6_turn:
            self._f6_turn = None
        elif not _is_priority_space(self.obs):
            # A non-priority decision (attack/block/target) always clears F6.
            self._f6_turn = None

    def _should_surface(self) -> bool:
        """Decide whether the hero's current decision point surfaces.

        Assumes the hero is to act on a non-terminal observation. Stops only
        govern pure priority windows; everything else always surfaces.
        """
        obs = self.obs
        assert obs is not None

        if not _is_priority_space(obs):
            return True
        if not self.auto_pass:
            return True
        if self._f6_turn is not None:
            # F6 yields through stops and stack alike until the turn ends.
            return False
        if self.stop_on_stack and len(obs.stack_objects) > 0:
            return True

        step_name = _enum_name(StepEnum, obs.turn.step)
        stop_step = ENGINE_STEP_TO_STOP_STEP.get(step_name)
        if stop_step is None:
            # Untap/cleanup/end-of-combat: no stop exists for these steps.
            return False
        side = "my" if _hero_is_active_player(obs) else "opponent"
        return stop_step in self.stops.get(side, [])

    def _stops_payload(self) -> dict[str, Any]:
        return {
            "my": list(self.stops.get("my", [])),
            "opponent": list(self.stops.get("opponent", [])),
            "stop_on_stack": self.stop_on_stack,
            "auto_pass": self.auto_pass,
        }

    def _drain_pending_villain_log(self) -> list[str]:
        pending = list(self._pending_villain_log)
        self._pending_villain_log = []
        return pending

    def _wire_message(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("No observation available.")

        data = hero_view(self.obs)
        log = self._drain_pending_villain_log()
        auto_passed = self._auto_passed_since_surface
        self._auto_passed_since_surface = 0
        if self.obs.game_over:
            self._finalize_trace(end_reason="game_over")
            payload = {
                "type": "game_over",
                "data": data,
                "winner": _winner_for_hero(self.obs),
                "stops": self._stops_payload(),
            }
            if self.deck_names:
                payload["deck_names"] = dict(self.deck_names)
            if log:
                payload["log"] = log
            if auto_passed:
                payload["auto_passed"] = auto_passed
            return payload

        payload = {
            "type": "observation",
            "data": data,
            "actions": describe_actions(self.obs),
            "action_space": _enum_name(
                ActionSpaceEnum, int(self.obs.action_space.action_space_type)
            ),
            "stops": self._stops_payload(),
        }
        if self.deck_names:
            payload["deck_names"] = dict(self.deck_names)
        if log:
            payload["log"] = log
        if auto_passed:
            payload["auto_passed"] = auto_passed
        return payload

    def _finalize_trace(self, end_reason: str) -> None:
        if self.trace is None or self.obs is None:
            return
        if self._trace_saved:
            return

        final_trace = replace(
            self.trace,
            final_observation=serialize_observation(self.obs),
            winner=_winner_for_hero(self.obs),
            end_reason=end_reason,
        )
        path = trace_store.save_trace(final_trace, self.trace_dir)
        self.trace = final_trace
        self.trace_id = path.stem
        self._trace_saved = True

    def close(self, end_reason: str) -> None:
        if self.trace is not None and not self._trace_saved:
            self._finalize_trace(end_reason=end_reason)

        self.env = None
        self.obs = None
        self.villain_policy = None


def _error_message(message: str) -> dict[str, str]:
    return {"type": "error", "message": message}


def _create_session_record() -> SessionRecord:
    while True:
        session_id = secrets.token_urlsafe(12)
        if session_id not in SESSION_REGISTRY:
            break

    record = SessionRecord(
        session_id=session_id,
        resume_token=secrets.token_urlsafe(24),
        game=GameSession(),
    )
    SESSION_REGISTRY[session_id] = record
    return record


def _drop_session(session_id: str, end_reason: str) -> None:
    record = SESSION_REGISTRY.pop(session_id, None)
    if record is None:
        return
    record.game.close(end_reason=end_reason)
    record.websocket = None


def _cleanup_expired_sessions() -> None:
    now = _now_utc()
    expired_ids = [
        session_id
        for session_id, record in SESSION_REGISTRY.items()
        if record.expires_at <= now
    ]
    for session_id in expired_ids:
        _drop_session(session_id, end_reason=SESSION_EXPIRED_END_REASON)


def _session_from_resume(
    raw_session_id: Any,
    raw_resume_token: Any,
) -> SessionRecord:
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise ValueError("resume messages require a non-empty 'session_id'.")
    if not isinstance(raw_resume_token, str) or not raw_resume_token:
        raise ValueError("resume messages require a non-empty 'resume_token'.")

    record = SESSION_REGISTRY.get(raw_session_id)
    if record is None:
        raise ValueError("Session not found or expired. Start a new game.")
    if record.resume_token != raw_resume_token:
        raise ValueError("Invalid resume credentials. Start a new game.")
    if record.expires_at <= _now_utc():
        _drop_session(raw_session_id, end_reason=SESSION_EXPIRED_END_REASON)
        raise ValueError("Session has expired. Start a new game.")
    return record


def _response_with_session(
    response: dict[str, Any],
    record: SessionRecord,
) -> dict[str, Any]:
    if response.get("type") == "observation":
        payload = dict(response)
        payload["session_id"] = record.session_id
        payload["resume_token"] = record.resume_token
        return payload
    return response


async def _attach_session_websocket(
    record: SessionRecord, websocket: WebSocket
) -> None:
    previous_websocket = record.websocket
    record.websocket = websocket
    record.touch()
    if previous_websocket is not None and previous_websocket is not websocket:
        with suppress(Exception):
            await previous_websocket.close(code=4000)


async def _get_or_create_attached_session(
    attached_session_id: str | None,
    websocket: WebSocket,
) -> SessionRecord:
    if attached_session_id is not None:
        existing = SESSION_REGISTRY.get(attached_session_id)
        if existing is not None:
            await _attach_session_websocket(existing, websocket)
            return existing

    record = _create_session_record()
    await _attach_session_websocket(record, websocket)
    return record


def _detach_session_websocket(session_id: str, websocket: WebSocket) -> None:
    record = SESSION_REGISTRY.get(session_id)
    if record is None:
        return
    if record.websocket is websocket:
        record.websocket = None
        record.touch()


@app.websocket("/ws/play")
async def play_socket(websocket: WebSocket) -> None:
    _cleanup_expired_sessions()
    await websocket.accept()
    attached_session_id: str | None = None

    try:
        while True:
            try:
                request = await websocket.receive_json()
            except WebSocketDisconnect:
                if attached_session_id is not None:
                    _detach_session_websocket(attached_session_id, websocket)
                raise

            _cleanup_expired_sessions()
            try:
                if not isinstance(request, dict):
                    raise ValueError("WebSocket payload must be a JSON object.")

                message_type = request.get("type")
                if message_type == "new_game":
                    record = await _get_or_create_attached_session(
                        attached_session_id,
                        websocket,
                    )
                    attached_session_id = record.session_id
                    response = record.game.new_game(request.get("config", {}))
                    record.touch()
                    response = _response_with_session(response, record)
                elif message_type == "action":
                    if "index" not in request:
                        raise ValueError("action messages require an 'index' field.")
                    if attached_session_id is None:
                        raise ValueError("No active game session. Send new_game first.")

                    record = SESSION_REGISTRY.get(attached_session_id)
                    if record is None:
                        attached_session_id = None
                        raise ValueError("Session expired. Start a new game.")

                    response = record.game.hero_action(request.get("index"))
                    record.touch()
                    response = _response_with_session(response, record)
                elif message_type in {"set_stops", "pass_turn"}:
                    if attached_session_id is None:
                        raise ValueError("No active game session. Send new_game first.")

                    record = SESSION_REGISTRY.get(attached_session_id)
                    if record is None:
                        attached_session_id = None
                        raise ValueError("Session expired. Start a new game.")

                    if message_type == "set_stops":
                        response = record.game.set_stops(
                            request.get("stops"),
                            request.get("stop_on_stack"),
                            request.get("auto_pass"),
                        )
                    else:
                        response = record.game.pass_turn()
                    record.touch()
                    response = _response_with_session(response, record)
                elif message_type == "resume":
                    record = _session_from_resume(
                        request.get("session_id"),
                        request.get("resume_token"),
                    )
                    await _attach_session_websocket(record, websocket)
                    attached_session_id = record.session_id
                    response = record.game.current_message()
                    response = _response_with_session(response, record)
                else:
                    raise ValueError(f"Unsupported message type: {message_type}")
            except ValueError as exc:
                await websocket.send_json(_error_message(str(exc)))
                continue

            await websocket.send_json(response)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if attached_session_id is not None:
            _drop_session(attached_session_id, end_reason="error")
        with suppress(Exception):
            await websocket.send_json(_error_message(str(exc)))
        with suppress(Exception):
            await websocket.close()


@app.get("/api/traces")
async def list_traces() -> list[dict[str, Any]]:
    return trace_store.list_trace_summaries()


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, reveal_hidden: bool = False) -> dict[str, Any]:
    try:
        payload = trace_store.load_trace(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = trace_store.prepare_trace_payload(payload, reveal_hidden=reveal_hidden)
    payload["id"] = trace_id
    return payload
