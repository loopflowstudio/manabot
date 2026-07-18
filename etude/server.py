"""
server.py
FastAPI server for interactive managym play over WebSocket.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import secrets
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import ValidationError

import managym
from managym.decision import Command as SemanticCommand, SemanticTransition

from . import trace as trace_store, villain as villain_module
from .advice import (
    AdviceRequest,
    advice_meta,
    request_advice,
    request_versioned_fixture_advice,
)
from .curated_pack import (
    CURATED_PACK,
    JEONG_INCREMENT_PACK,
    curated_pack_for_matchup,
)
from .enums import (
    ActionEnum,
    ActionSpaceEnum,
    EventTypeEnum,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
)
from .experience_protocol import PROTOCOL_VERSION, Command, ExperienceFrame
from .presentation import PresentationProjector
from .replay_index import (
    CANONICAL_REPLAY_VERSION,
    CanonicalReplayProjectionV1,
    CanonicalReplayUnavailableError,
    CanonicalReplayV1,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecision,
    RestoredReplayDecision,
    ViewerPresentationTrack,
    canonical_projection_sha256,
    load_canonical_replay,
    project_replay,
    projection_with_addresses,
    restore_decision,
)
from .study_branch import StudyBranch, StudyBranchUnavailableError, StudyForkProvider
from .study_runtime import (
    HistoricalStudyEvidenceProvider,
    HistoricalStudyEvidenceRequest,
    JoinedStudyEvidence,
    StudyEvidenceMismatchError,
    StudyEvidenceUnavailableError,
    StudyPlanKind,
    StudyPlanUnavailableError,
    UnavailableHistoricalStudyEvidenceProvider,
    join_historical_study_evidence,
    select_study_plan,
)
from .testing_house_protocol import (
    REQUEST_ADAPTER,
    REQUEST_TYPES,
    ROLE_CAPABILITIES,
    ActionMessage,
    AuthorBeliefMessage,
    BranchPreviewMessage,
    BranchRevealMessage,
    CommandMessage,
    JoinTableMessage,
    NewGameMessage,
    PassTurnMessage,
    RematchMessage,
    RestoreDecisionMessage,
    ResumeMessage,
    RetryDecisionMessage,
    ReturnFromBranchMessage,
    ReturnToLiveMessage,
    SetStopsMessage,
    ShareBeliefMessage,
    TableCapability,
    TableSnapshot,
    TransferPilotMessage,
    ViewerAccess,
    ViewerIdentity,
    ViewerRole,
)
from .trace import GameConfig, Trace, TraceEvent
from .villain import VillainPolicy, build_villain_policy

# Interactive remains a lightweight mirror of manabot.verify.util. The two
# curated defaults load from the installed manifest so the server does not
# import torch at startup and does not carry a competing deck definition.
INTERACTIVE_DECK = {
    "Island": 12,
    "Mountain": 12,
    "Gray Ogre": 6,
    "Wind Drake": 6,
    "Man-o'-War": 4,
    "Raging Goblin": 4,
    "Lightning Bolt": 6,
    "Counterspell": 4,
    "Ancestral Recall": 3,
    "Pyroclasm": 3,
}
UR_LESSONS_DECK = dict(CURATED_PACK.hero_deck)
GW_ALLIES_DECK = dict(CURATED_PACK.villain_deck)
GW_ALLIES_JEONG_DECK = dict(JEONG_INCREMENT_PACK.hero_deck)

# Decks selectable by name over the wire (new_game.config hero_deck /
# villain_deck may be one of these keys instead of a {card: count} object).
NAMED_DECKS: dict[str, dict[str, int]] = {
    "interactive": INTERACTIVE_DECK,
    CURATED_PACK.hero_deck_id: UR_LESSONS_DECK,
    CURATED_PACK.villain_deck_id: GW_ALLIES_DECK,
    JEONG_INCREMENT_PACK.hero_deck_id: GW_ALLIES_JEONG_DECK,
}
DECK_DISPLAY_NAMES = {
    "interactive": "Interactive",
    CURATED_PACK.hero_deck_id: CURATED_PACK.hero_display_name,
    CURATED_PACK.villain_deck_id: CURATED_PACK.villain_display_name,
    JEONG_INCREMENT_PACK.hero_deck_id: JEONG_INCREMENT_PACK.hero_display_name,
    "custom": "Custom",
}
# Default matchup: the Milestone-1 two-deck slice, UR as hero vs GW villain.
DEFAULT_HERO_DECK_NAME = CURATED_PACK.hero_deck_id
DEFAULT_VILLAIN_DECK_NAME = CURATED_PACK.villain_deck_id
# Backwards-compatible alias (tests and older callers).
DEFAULT_DECK = INTERACTIVE_DECK

MAX_AUTOPLAY_STEPS = 1024
HERO_PLAYER_INDEX = 0
VILLAIN_TYPES = {"passive", "random", "search", "checkpoint"}
MAX_ACCEPTED_COMMANDS = 64
MAX_PRESENTATION_EVENTS = 256
MAX_UINT64 = 2**64 - 1
CONTENT_HASH = "legacy-content-unversioned"
ASSET_MANIFEST_HASH = CURATED_PACK.manifest_sha256
CURATED_SEMANTIC_PACK_KEYS = {
    CURATED_PACK.pack_id: "ur-lessons-vs-gw-allies",
    JEONG_INCREMENT_PACK.pack_id: "tla-jeong-increment-v1",
}

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

app = FastAPI(title="Etude Fantasia")
SESSION_TTL = timedelta(minutes=15)
SESSION_EXPIRED_END_REASON = "session_expired"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ParticipantLease:
    viewer_id: str
    resume_token: str
    role: ViewerRole
    grant_revision: int = 1
    websocket: WebSocket | None = None
    personal_beliefs: dict[str, dict[str, Any]] = field(default_factory=dict)
    study_attempts: set[str] = field(default_factory=set)


@dataclass
class SessionRecord:
    session_id: str
    resume_token: str
    game: "GameSession"
    websocket: WebSocket | None = None
    expires_at: datetime = field(default_factory=lambda: _now_utc() + SESSION_TTL)
    participants: dict[str, ParticipantLease] = field(default_factory=dict)
    watcher_invite_digest: str | None = None
    table_revision: int = 0
    shared_beliefs: dict[str, dict[str, Any]] = field(default_factory=dict)
    accepted_command_submitters: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.participants:
            return
        viewer_id = f"viewer.{secrets.token_urlsafe(12)}"
        self.participants[viewer_id] = ParticipantLease(
            viewer_id=viewer_id,
            resume_token=self.resume_token,
            role=ViewerRole.PILOT,
            websocket=self.websocket,
        )

    def pilot(self) -> ParticipantLease:
        return next(
            participant
            for participant in self.participants.values()
            if participant.role == ViewerRole.PILOT
        )

    def touch(self) -> None:
        self.expires_at = _now_utc() + SESSION_TTL


SESSION_REGISTRY: dict[str, SessionRecord] = {}


@dataclass(frozen=True)
class DecisionContext:
    """One immutable actor-safe frame and its engine lowering map."""

    viewer: int
    revision: int
    prompt_id: int
    action_space: str
    action_by_offer: dict[int, int]
    actions: list[dict[str, Any]]
    offers: list[dict[str, Any]]
    frame: dict[str, Any]


# Transitional name retained for type references in downstream tests.
PublishedPrompt = DecisionContext


@dataclass(frozen=True)
class AuthorityTransition:
    """Authority-private evidence for one exact engine transition."""

    actor: int
    source: str | None
    automatic: bool
    from_revision: int
    to_revision: int
    action_space: str
    action_type: str
    legal_action_count: int
    offer_count: int
    prompt_id: int | None
    offer: dict[str, Any] | None
    command: dict[str, Any] | None
    state_before: str
    state_after: str
    semantic_events: list[dict[str, Any]]
    presentation_events: list[dict[str, Any]]
    encountered_definition_ids: list[int]


class StudyAttemptNotFoundError(ValueError):
    """No live Study attempt exists under this authority session."""


class StudyCommandUnavailableError(ValueError):
    """A canonical replay offer cannot lower through the retained Study fork."""


@dataclass
class StudyAttempt:
    attempt_id: str
    owner_viewer_id: str
    trace_id: str
    address: str
    projection: CanonicalReplayProjectionV1
    restored: RestoredReplayDecision
    branch: StudyBranch
    retry_command: Command
    retry_projection: dict[str, Any]
    retry_presentation: list[dict[str, Any]]
    evidence: JoinedStudyEvidence | None = None


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
        # Effective P/T (statics, until-EOT buffs, +1/+1 counters) — what the
        # SBA actually checks. Printed values ride alongside so the UI can
        # show "4/4 (2/2)" on a buffed permanent.
        "power": int(permanent.power),
        "toughness": int(permanent.toughness),
        "base_power": int(card.power) if card else None,
        "base_toughness": int(card.toughness) if card else None,
        "plus1_counters": int(permanent.plus1_counters),
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


def _definition_ids_by_object(obs: managym.Observation) -> dict[int, int]:
    """Resolve live card, permanent, and stack identities to typed definitions."""

    result: dict[int, int] = {}
    for cards, permanents in (
        (obs.agent_cards, obs.agent_permanents),
        (obs.opponent_cards, obs.opponent_permanents),
    ):
        for card in cards:
            result[int(card.id)] = int(card.registry_key)
        battlefield_cards = [
            card for card in cards if _enum_name(ZoneEnum, card.zone) == "BATTLEFIELD"
        ]
        for index, permanent in enumerate(permanents):
            if index < len(battlefield_cards):
                result[int(permanent.id)] = int(battlefield_cards[index].registry_key)
    for stack_object in obs.stack_objects:
        result[int(stack_object.stack_object_id)] = int(
            stack_object.source_card_registry_key
        )
    return result


def _semantic_event_payload(
    event: managym.EventData,
    definition_ids: dict[int, int],
) -> dict[str, Any]:
    related_definitions = sorted(
        {
            definition_ids[identity]
            for identity in (int(event.source_id), int(event.target_id))
            if identity in definition_ids
        }
    )
    return {
        "event_type": _enum_name(EventTypeEnum, event.event_type),
        "source_kind": int(event.source_kind),
        "source_id": int(event.source_id),
        "source_incarnation": int(event.source_incarnation),
        "target_kind": int(event.target_kind),
        "target_id": int(event.target_id),
        "target_incarnation": int(event.target_incarnation),
        "amount": int(event.amount),
        "controller_id": int(event.controller_id),
        "from_zone": int(event.from_zone),
        "to_zone": int(event.to_zone),
        "definition_ids": related_definitions,
    }


def _encountered_definition_ids(
    before: managym.Observation,
    after: managym.Observation,
    action: dict[str, Any],
    semantic_events: list[dict[str, Any]],
) -> list[int]:
    before_ids = _definition_ids_by_object(before)
    after_ids = _definition_ids_by_object(after)
    identities = {
        int(identity)
        for identity in action.get("focus", [])
        if isinstance(identity, int)
    }
    definitions = {
        mapping[identity]
        for mapping in (before_ids, after_ids)
        for identity in identities
        if identity in mapping
    }
    definitions.update(
        definition_id
        for event in semantic_events
        for definition_id in event["definition_ids"]
    )
    definitions.update(
        int(stack_object.source_card_registry_key)
        for stack_object in [*before.stack_objects, *after.stack_objects]
    )

    before_taps: dict[int, tuple[int, bool]] = {}
    after_taps: dict[int, tuple[int, bool]] = {}
    for observation, target in ((before, before_taps), (after, after_taps)):
        for cards, permanents in (
            (observation.agent_cards, observation.agent_permanents),
            (observation.opponent_cards, observation.opponent_permanents),
        ):
            battlefield_cards = [
                card
                for card in cards
                if _enum_name(ZoneEnum, card.zone) == "BATTLEFIELD"
            ]
            for index, permanent in enumerate(permanents):
                if index < len(battlefield_cards):
                    target[int(permanent.id)] = (
                        int(battlefield_cards[index].registry_key),
                        bool(permanent.tapped),
                    )
    for identity, (definition_id, was_tapped) in before_taps.items():
        current = after_taps.get(identity)
        if current is not None and not was_tapped and current[1]:
            definitions.add(definition_id)
    return sorted(definitions)


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
        card for card in cards if _enum_name(ZoneEnum, card.zone) == "BATTLEFIELD"
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
    declared: bool | None = None,
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
    if declared is None:
        declared = getattr(action, "declared", None)
    focus_names = [names.get(int(value)) for value in action.focus]
    first = focus_names[0] if focus_names else None

    if action_name == "PRIORITY_PASS_PRIORITY":
        return "Pass priority"
    if action_name == "PRIORITY_PLAY_LAND" and first:
        return f"Play {first}"
    if action_name == "PRIORITY_CAST_SPELL" and first:
        return f"Cast {first}"
    if action_name == "DECLARE_ATTACKER" and first:
        return (
            f"Do not attack with {first}"
            if declared is False
            else f"Attack with {first}"
        )
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
                "declared": action.declared,
                "description": _format_action(
                    action, names, space_kind, action.declared
                ),
            }
        )
    return results


def viewer_view(obs: managym.Observation, viewer: int) -> dict[str, Any]:
    """Serialize an observation for one acting viewer.

    Two guarantees, regardless of whose perspective the engine observation is
    from:
      1. ``agent`` is always ``viewer`` and ``opponent`` the other player.
      2. The opponent's hand is redacted (libraries are never serialized).
    """
    data = serialize_observation(obs)
    if int(data["agent"]["player_index"]) != viewer:
        data["agent"], data["opponent"] = data["opponent"], data["agent"]
        data["won"] = bool(obs.game_over) and not bool(obs.won)

    trace_store.redact_observation(data)
    return data


def hero_view(obs: managym.Observation) -> dict[str, Any]:
    """Backwards-compatible player-0 projection for the live table."""
    return viewer_view(obs, HERO_PLAYER_INDEX)


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
        obs.agent if int(obs.agent.player_index) == HERO_PLAYER_INDEX else obs.opponent
    )
    return int(obs.turn.active_player_id) == int(hero.id)


def _pass_priority_index(actions: list[dict[str, Any]]) -> int | None:
    for action in actions:
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return int(action["index"])
    return None


def _offer_verb(action_type: str) -> str:
    return {
        "PRIORITY_CAST_SPELL": "cast",
        "PRIORITY_PLAY_LAND": "play_land",
        "PRIORITY_ACTIVATE_ABILITY": "activate",
        "PRIORITY_PASS_PRIORITY": "pass_priority",
        "DECLARE_ATTACKER": "declare_attackers",
        "DECLARE_BLOCKER": "declare_blockers",
        "PAY_COST": "pay",
    }.get(action_type, "choose")


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
            raise ValueError("checkpoint villain requires a 'villain_checkpoint' path.")
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

    selected_pack = curated_pack_for_matchup(hero_deck_name, villain_deck_name)

    return GameConfig(
        hero_deck=hero_deck,
        villain_deck=villain_deck,
        hero_deck_name=hero_deck_name,
        villain_deck_name=villain_deck_name,
        asset_pack=selected_pack.reference if selected_pack is not None else None,
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
    def __init__(
        self,
        trace_dir: Path | None = None,
        *,
        id_factory: Callable[[str], str] | None = None,
        clock: Callable[[], str] | None = None,
        villain_offer_policy: Callable[[DecisionContext], int] | None = None,
        capture_authority_evidence: bool = False,
        historical_evidence_provider: HistoricalStudyEvidenceProvider | None = None,
        allow_fixture_study_evidence: bool = False,
    ):
        self.trace_dir = trace_dir or trace_store.TRACES_DIR
        self._id_factory = id_factory or (lambda _kind: secrets.token_urlsafe(16))
        self._clock = clock or trace_store.utc_now_iso
        self._villain_offer_policy = villain_offer_policy
        self._capture_authority_evidence = capture_authority_evidence
        self._historical_evidence_provider = (
            historical_evidence_provider or UnavailableHistoricalStudyEvidenceProvider()
        )
        self._allow_fixture_study_evidence = allow_fixture_study_evidence
        self.env: managym.Env | None = None
        self.obs: managym.Observation | None = None
        self.villain_policy: VillainPolicy | None = None
        self.trace: Trace | None = None
        self.trace_id: str | None = None
        # Display names for the current matchup, echoed on every payload.
        self.deck_names: dict[str, str] | None = None
        self.asset_pack: dict[str, str] | None = None
        self.content_hash = CONTENT_HASH
        self.asset_manifest_hash = ASSET_MANIFEST_HASH
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
        # Protocol-v1 adapter state. The engine remains authoritative; this
        # layer only gives each surfaced positional action a stable envelope.
        self.match_id = self._id_factory("match")
        self.revision = 0
        self._next_prompt_id = 1
        self.published_prompt: PublishedPrompt | None = None
        self.accepted_commands: dict[str, dict[str, Any]] = {}
        self.presentation = PresentationProjector()
        self.presentation_events: list[dict[str, Any]] = []
        self._last_presentation_cursor = 0
        # Durable replay truth is unbounded by the live reconnect ledger.
        self.canonical_decisions: list[ReplayDecision] = []
        self.canonical_presentation: dict[int, list[dict[str, Any]]] = {
            HERO_PLAYER_INDEX: [],
            1: [],
        }
        self._authority_command_seq = 0
        self._study_roots: dict[int, managym.Env] = {}
        self._study_provider: StudyForkProvider | None = None
        self.authority_transitions: list[AuthorityTransition] = []
        self.authority_fallback_counters = {
            "legacy_fixed_action": 0,
            "card_name_dispatch": 0,
            "candidate_cap": 0,
            "client_legality": 0,
        }
        self._study_attempts: dict[str, StudyAttempt] = {}
        self._study_attempt_seq = 0

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

        manifest = self.env.content_pack_manifest()
        compiled = manifest.get("compiled_semantics")
        selected_pack = curated_pack_for_matchup(
            config.hero_deck_name,
            config.villain_deck_name,
        )
        if selected_pack is not None:
            expected_pack_key = CURATED_SEMANTIC_PACK_KEYS[selected_pack.pack_id]
            actual_pack_key = None if compiled is None else compiled.get("pack_key")
            if actual_pack_key != expected_pack_key:
                raise RuntimeError(
                    "Curated matchup did not select its compiled semantic pack: "
                    f"expected {expected_pack_key!r}, got {actual_pack_key!r}"
                )
            self.asset_manifest_hash = selected_pack.manifest_sha256
            self.content_hash = (
                CONTENT_HASH
                if selected_pack is CURATED_PACK
                else str(manifest["content_digest"])
            )
        else:
            self.content_hash = CONTENT_HASH
            self.asset_manifest_hash = ASSET_MANIFEST_HASH

        self.deck_names = {
            "hero": DECK_DISPLAY_NAMES.get(config.hero_deck_name, "Custom"),
            "villain": DECK_DISPLAY_NAMES.get(config.villain_deck_name, "Custom"),
        }
        self.asset_pack = dict(config.asset_pack) if config.asset_pack else None
        self.trace = Trace(
            config=config,
            events=[],
            final_observation={},
            winner=None,
            end_reason="disconnect",
            timestamp=self._clock(),
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
        self.match_id = self._id_factory("match")
        self.revision = 0
        self._next_prompt_id = 1
        self.published_prompt = None
        self.accepted_commands = {}
        self.presentation.reset()
        self.presentation_events = []
        self._last_presentation_cursor = 0
        self.canonical_decisions = []
        self.canonical_presentation = {HERO_PLAYER_INDEX: [], 1: []}
        self._authority_command_seq = 0
        self._study_roots = {}
        self._study_provider = None
        self.authority_transitions = []
        self.authority_fallback_counters = {
            "legacy_fixed_action": 0,
            "card_name_dispatch": 0,
            "candidate_cap": 0,
            "client_legality": 0,
        }
        self._close_study_attempts()
        self._study_attempt_seq = 0

        # Setup itself is an authority snapshot. Clear staged facts before any
        # auto-play transition so each subsequent drain belongs to one exact
        # engine action/revision pair.
        self.presentation.drain(
            from_revision=self.revision,
            to_revision=self.revision,
            caused_by=None,
        )
        self._advance()
        self._last_presentation_cursor = self.presentation.next_seq
        return self._wire_message(reason="initial_connect")

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

        context = self._publish_current_prompt()
        if context is None:
            raise ValueError("No current decision is available.")
        if action_index < 0 or action_index >= len(context.actions):
            raise ValueError(f"Action index out of range: {action_index}")

        selected = next(
            (
                offer_id
                for offer_id, engine_index in context.action_by_offer.items()
                if engine_index == action_index
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("Hero action has no unique authority offer.")
        self.authority_fallback_counters["legacy_fixed_action"] += 1
        self.authority_fallback_counters["client_legality"] += 1
        command = self._authority_command(context, selected, "compat")
        batch_cursor = self.presentation.next_seq
        self._apply_bound_command(context, command, source="client")
        self._advance()
        self._last_presentation_cursor = batch_cursor
        return self._wire_message()

    def hero_command(self, raw: Any) -> dict[str, Any]:
        """Validate and apply one revision-bound protocol-v1 command.

        Offer IDs are lowered through ``PublishedPrompt``. The client never
        chooses or replays a raw engine action index on this path.
        """
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if not isinstance(raw, dict):
            raise ValueError("command must be an object.")

        command_id = raw.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a non-empty string.")
        if command_id.startswith("authority."):
            return self._reject_command(raw, "reserved_command_id", recover=False)

        # Check retries before any revision/match validation. A response may
        # have been lost after the original command committed.
        receipt = self.accepted_commands.get(command_id)
        if receipt is not None:
            return {
                "type": "command_outcome",
                "status": "duplicate",
                "receipt": dict(receipt),
                "recovery": self.current_recovery("duplicate_command"),
            }

        if raw.get("match_id") != self.match_id:
            return self._reject_command(raw, "wrong_match", recover=False)

        context = self.published_prompt
        if context is None:
            return self._reject_command(raw, "authority_busy", recover=True)
        if raw.get("expected_revision") != context.revision:
            return self._reject_command(raw, "stale_revision", recover=True)
        if raw.get("prompt_id") != context.prompt_id:
            return self._reject_command(raw, "stale_prompt", recover=True)

        offer_id = raw.get("offer_id")
        if context.action_by_offer.get(offer_id) is None:
            return self._reject_command(raw, "unknown_offer", recover=True)
        if raw.get("answers") != []:
            return self._reject_command(raw, "invalid_selection", recover=False)
        if self.obs.game_over:
            return self._reject_command(raw, "authority_busy", recover=True)
        if not _is_hero_turn(self.obs):
            return self._reject_command(raw, "not_actor", recover=False)

        try:
            command = Command.model_validate(raw).model_dump(mode="json")
        except Exception:
            return self._reject_command(raw, "invalid_command", recover=False)

        base_revision = context.revision
        batch_cursor = self.presentation.next_seq
        self._apply_bound_command(context, command, source="client")
        self._advance()
        self._last_presentation_cursor = batch_cursor
        presentation = [
            dict(event)
            for event in self.presentation_events
            if int(event["seq"]) >= batch_cursor
        ]
        frame = self._experience_frame()
        transient = self._drain_surface_metadata()
        receipt = {
            "command_id": command_id,
            "actor": HERO_PLAYER_INDEX,
            "accepted_at": base_revision,
            "resulting_revision": self.revision,
            "resulting_frame_hash": frame["frame_hash"],
        }
        self.accepted_commands[command_id] = receipt
        while len(self.accepted_commands) > MAX_ACCEPTED_COMMANDS:
            self.accepted_commands.pop(next(iter(self.accepted_commands)))

        return {
            "type": "command_outcome",
            "status": "accepted",
            "update": {
                "base_revision": base_revision,
                "frame": frame,
                "presentation": presentation,
                "receipt": receipt,
                **transient,
            },
        }

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

        # Stop configuration is versioned authority state even though it is not
        # a player decision and therefore never receives a replay ordinal.
        self._advance_protocol_revision()
        batch_cursor = self.presentation.next_seq
        self._advance()
        self._last_presentation_cursor = batch_cursor
        return self._wire_message()

    def pass_turn(self) -> dict[str, Any]:
        """F6: yield every hero priority window (through stops and stack)
        until the current turn ends or a non-priority decision surfaces."""
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if self.obs.game_over:
            raise ValueError("Game is already over. Start a new game.")

        if _is_hero_turn(self.obs) and _is_priority_space(self.obs):
            batch_cursor = self.presentation.next_seq
            self._f6_turn = int(self.obs.turn.turn_number)
            self._advance()
            self._last_presentation_cursor = batch_cursor
        return self._wire_message()

    def current_message(self, presentation_cursor: int | None = None) -> dict[str, Any]:
        if self.obs is None:
            raise ValueError("No active game session. Send new_game first.")
        if presentation_cursor is None:
            presentation_cursor, _ = self._presentation_recovery_tail(None)
        return self._wire_message(
            reason="reconnect",
            presentation_cursor=presentation_cursor,
        )

    def _authority_command(
        self,
        context: DecisionContext,
        offer_id: int,
        namespace: str,
    ) -> dict[str, Any]:
        """Create a collision-free authority-local bound command."""
        self._authority_command_seq += 1
        return {
            "command_id": (
                f"authority.{namespace}.{self.match_id}.{self._authority_command_seq}"
            ),
            "match_id": self.match_id,
            "expected_revision": context.revision,
            "prompt_id": context.prompt_id,
            "offer_id": offer_id,
            "answers": [],
        }

    def _apply_bound_command(
        self,
        context: DecisionContext,
        raw_command: dict[str, Any],
        *,
        source: str,
    ) -> None:
        """Validate, capture, and lower one deliberate authority command."""
        command = Command.model_validate(raw_command)
        if (
            command.match_id != self.match_id
            or command.expected_revision != context.revision
            or command.prompt_id != context.prompt_id
        ):
            raise RuntimeError("Bound command identity differs from decision context.")
        action_index = context.action_by_offer.get(command.offer_id)
        matching_offers = [
            offer for offer in context.offers if int(offer["id"]) == command.offer_id
        ]
        if action_index is None or len(matching_offers) != 1:
            raise RuntimeError("Bound command does not lower through one exact offer.")
        actor = "hero" if context.viewer == HERO_PLAYER_INDEX else "villain"
        self._step_and_record(
            actor=actor,
            action_index=action_index,
            actions=context.actions,
            context=context,
            command=command.model_dump(mode="json"),
            source=source,
        )

    def _step_and_record(
        self,
        actor: str,
        action_index: int,
        actions: list[dict[str, Any]],
        auto: bool = False,
        context: DecisionContext | None = None,
        command: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            raise RuntimeError("Cannot step without an active game.")

        observation = serialize_observation(self.obs)
        before_observation = self.obs
        state_before = (
            self.env.state_digest() if self._capture_authority_evidence else None
        )
        actor_index = HERO_PLAYER_INDEX if actor == "hero" else 1
        selected_offer: dict[str, Any] | None = None
        if not auto:
            if context is None or command is None or source not in {"client", "policy"}:
                raise RuntimeError("Deliberate engine actions require a bound command.")
            if context.viewer != actor_index:
                raise RuntimeError("Decision context actor differs from engine actor.")
            offer_id = int(command["offer_id"])
            selected_offer = next(
                offer for offer in context.offers if int(offer["id"]) == offer_id
            )
            row = ReplayDecision.model_validate(
                {
                    "ordinal": len(self.canonical_decisions),
                    "viewer": actor_index,
                    "source": source,
                    "revision": context.revision,
                    "prompt_id": context.prompt_id,
                    "offer_id": offer_id,
                    "command_id": command["command_id"],
                    "presentation_cursor": len(
                        self.canonical_presentation[actor_index]
                    ),
                    "frame": context.frame,
                    "offer": selected_offer,
                    "command": command,
                }
            )
            self._study_roots[int(row.ordinal)] = self.env.clone_env()
            self.canonical_decisions.append(row)
        action_description = actions[action_index]["description"]
        if self.presentation.note_action(
            self.obs,
            actions[action_index],
            actor_index=actor_index,
        ):
            self.authority_fallback_counters["card_name_dispatch"] += 1
        from_revision = self.revision
        if auto:
            next_obs, reward, _, _, _ = self.env.step(action_index)
        else:
            assert command is not None
            semantic_frame = json.loads(self.env.semantic_decision_frame_json())
            if (
                int(semantic_frame["actor"]) != actor_index
                or len(semantic_frame["offers"]) != len(actions)
                or int(semantic_frame["offers"][action_index]["id"])
                != int(command["offer_id"])
            ):
                raise RuntimeError("Etude decision drifted from semantic authority.")
            semantic_command = SemanticCommand(
                command_id=str(command["command_id"]),
                expected_revision=int(semantic_frame["revision"]),
                offer_id=int(command["offer_id"]),
            )
            transition_json, next_obs, reward, _, _, _ = self.env.step_semantic_command(
                semantic_command.to_json()
            )
            semantic_transition = SemanticTransition.from_json(transition_json)
            if semantic_transition.receipt.command_id != str(command["command_id"]):
                raise RuntimeError("Semantic receipt command identity drifted.")
        state_after: str | None = None
        semantic_events: list[dict[str, Any]] = []
        encountered_definition_ids: list[int] = []
        if self._capture_authority_evidence:
            state_after = self.env.state_digest()
            definition_ids = {
                **_definition_ids_by_object(before_observation),
                **_definition_ids_by_object(next_obs),
            }
            semantic_events = [
                _semantic_event_payload(event, definition_ids)
                for event in next_obs.recent_events
            ]
            encountered_definition_ids = _encountered_definition_ids(
                before_observation,
                next_obs,
                actions[action_index],
                semantic_events,
            )
        self.presentation.observe(next_obs)
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
        self._advance_protocol_revision()
        caused_by = None if command is None else str(command["command_id"])
        presentation_events = self._commit_step_presentation(
            from_revision=from_revision,
            actor=actor_index,
            caused_by=caused_by,
        )
        if self._capture_authority_evidence:
            assert state_before is not None and state_after is not None
            self.authority_transitions.append(
                AuthorityTransition(
                    actor=actor_index,
                    source=source,
                    automatic=auto,
                    from_revision=from_revision,
                    to_revision=self.revision,
                    action_space=(
                        context.action_space
                        if context is not None
                        else _enum_name(
                            ActionSpaceEnum,
                            before_observation.action_space.action_space_type,
                        )
                    ),
                    action_type=str(actions[action_index]["type"]),
                    legal_action_count=len(actions),
                    offer_count=0 if context is None else len(context.offers),
                    prompt_id=None if context is None else context.prompt_id,
                    offer=deepcopy(selected_offer),
                    command=deepcopy(command),
                    state_before=state_before,
                    state_after=state_after,
                    semantic_events=semantic_events,
                    presentation_events=deepcopy(presentation_events),
                    encountered_definition_ids=encountered_definition_ids,
                )
            )

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

            context = self._build_decision_context(self.obs, viewer=1)
            if self._villain_offer_policy is not None:
                offer_id = int(self._villain_offer_policy(context))
                action_index = context.action_by_offer.get(offer_id)
                if action_index is None:
                    raise RuntimeError(
                        f"Villain offer policy selected unknown offer: {offer_id}"
                    )
            else:
                self.authority_fallback_counters["legacy_fixed_action"] += 1
                action_index = int(self.villain_policy(self.env, self.obs))
                if action_index < 0 or action_index >= len(context.actions):
                    raise RuntimeError(
                        f"Villain policy selected invalid action index: {action_index}"
                    )
                matching = [
                    candidate_offer_id
                    for candidate_offer_id, engine_index in context.action_by_offer.items()
                    if engine_index == action_index
                ]
                if len(matching) != 1:
                    raise RuntimeError(
                        "Villain policy action does not map to one authority offer."
                    )
                offer_id = matching[0]
            command = self._authority_command(context, offer_id, "policy")
            self._apply_bound_command(
                context,
                command,
                source="policy",
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

    def _drain_surface_metadata(self) -> dict[str, Any]:
        """Drain compatibility narration outside the canonical frame."""
        payload: dict[str, Any] = {}
        log = self._drain_pending_villain_log()
        auto_passed = self._auto_passed_since_surface
        self._auto_passed_since_surface = 0
        if log:
            payload["log"] = log
        if auto_passed:
            payload["auto_passed"] = auto_passed
        return payload

    def _advance_protocol_revision(self) -> None:
        self.revision += 1
        self.published_prompt = None

    def _commit_step_presentation(
        self,
        *,
        from_revision: int,
        actor: int,
        caused_by: str | None,
    ) -> list[dict[str, Any]]:
        """Commit one engine step into both authorized semantic tracks."""
        authority_events = self.presentation.drain(
            from_revision=from_revision,
            to_revision=self.revision,
            caused_by=caused_by,
        )
        viewer_events: dict[int, list[dict[str, Any]]] = {}
        for viewer in (HERO_PLAYER_INDEX, 1):
            projected: list[dict[str, Any]] = []
            for event in authority_events:
                copy = deepcopy(event)
                if viewer != actor:
                    copy["caused_by"] = None
                projected.append(copy)
            self.canonical_presentation[viewer].extend(projected)
            viewer_events[viewer] = projected

        live_events = viewer_events[HERO_PLAYER_INDEX]
        self.presentation_events.extend(live_events)
        if len(self.presentation_events) > MAX_PRESENTATION_EVENTS:
            self.presentation_events = self.presentation_events[
                -MAX_PRESENTATION_EVENTS:
            ]
        if live_events and self.trace is not None and self.trace.events:
            index = len(self.trace.events) - 1
            self.trace.events[index] = replace(
                self.trace.events[index],
                presentation=deepcopy(live_events),
            )
        return authority_events

    def _publish_current_prompt(self) -> PublishedPrompt | None:
        if self.obs is None or self.obs.game_over:
            self.published_prompt = None
            return None
        if self.published_prompt is not None:
            return self.published_prompt

        self.published_prompt = self._build_decision_context(
            self.obs,
            viewer=HERO_PLAYER_INDEX,
        )
        return self.published_prompt

    def _build_decision_context(
        self,
        obs: managym.Observation,
        *,
        viewer: int,
    ) -> DecisionContext:
        """Build one exact actor-safe frame and raw-action lowering map."""
        actions = describe_actions(obs)
        action_space = _enum_name(
            ActionSpaceEnum, int(obs.action_space.action_space_type)
        )
        prompt_id = self._next_prompt_id
        self._next_prompt_id += 1
        action_by_offer = {
            offer_id: int(action["index"]) for offer_id, action in enumerate(actions)
        }
        offers = [
            {
                "id": offer_id,
                "actor": viewer,
                "verb": _offer_verb(str(action.get("type", ""))),
                "source": None,
                "label": action["description"],
                "help": None,
                "choices": [],
                "confirm_label": action["description"],
                # Temporary direct-manipulation bridge for the existing table.
                "action_type": action.get("type", ""),
                "focus": list(action.get("focus", [])),
            }
            for offer_id, action in enumerate(actions)
        ]
        if len(offers) != len(actions):
            self.authority_fallback_counters["candidate_cap"] += 1
        prompt_view = {
            "id": prompt_id,
            "actor": viewer,
            "kind": action_space.lower(),
            "title": "Your priority"
            if action_space == "PRIORITY"
            else "Choose an action",
            "instruction": "Choose an action",
        }
        frame = self._build_frame(
            obs,
            viewer=viewer,
            prompt=prompt_view,
            offers=offers,
            action_space=action_space,
        )
        return DecisionContext(
            viewer=viewer,
            revision=self.revision,
            prompt_id=prompt_id,
            action_space=action_space,
            action_by_offer=action_by_offer,
            actions=actions,
            offers=offers,
            frame=frame,
        )

    def _build_frame(
        self,
        obs: managym.Observation,
        *,
        viewer: int,
        prompt: dict[str, Any] | None,
        offers: list[dict[str, Any]],
        action_space: str,
    ) -> dict[str, Any]:
        frame_without_hash: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "match_id": self.match_id,
            "revision": self.revision,
            "content_hash": self.content_hash,
            "asset_manifest_hash": self.asset_manifest_hash,
            "status": "game_over" if obs.game_over else "ready",
            "prompt": prompt,
            "projection": viewer_view(obs, viewer),
            "offers": offers,
            "winner": _winner_for_hero(obs) if viewer == HERO_PLAYER_INDEX else None,
            "action_space": action_space,
            "stops": self._stops_payload(),
            "deck_names": dict(self.deck_names) if self.deck_names else None,
            "asset_pack": dict(self.asset_pack) if self.asset_pack else None,
            "log": None,
            "auto_passed": None,
        }
        frame_without_hash["frame_hash"] = hashlib.sha256(
            json.dumps(
                frame_without_hash,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ExperienceFrame.model_validate(frame_without_hash).model_dump(
            mode="json"
        )

    def _experience_frame(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("No observation available.")

        if self.obs.game_over:
            self._finalize_trace(end_reason="game_over")
            return self._build_frame(
                self.obs,
                viewer=HERO_PLAYER_INDEX,
                prompt=None,
                offers=[],
                action_space="",
            )
        prompt = self._publish_current_prompt()
        if prompt is None:
            raise RuntimeError("Non-terminal authority has no decision context.")
        return deepcopy(prompt.frame)

    def _presentation_recovery_tail(
        self, requested_cursor: int | None
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return a contiguous retained tail starting at an exact event address.

        A complete frame makes recovery correct even when the requested cursor
        has fallen out of the bounded in-process ledger. In that case (or when
        a client claims to be ahead) theater resets to the oldest retained
        event instead of guessing at missing semantics.
        """

        head = self.presentation.next_seq
        oldest = (
            int(self.presentation_events[0]["seq"])
            if self.presentation_events
            else head
        )
        cursor = requested_cursor
        if cursor is None or cursor < oldest or cursor > head:
            cursor = oldest
        tail = [
            event for event in self.presentation_events if int(event["seq"]) >= cursor
        ]
        return cursor, tail

    def current_recovery(
        self,
        reason: str,
        presentation_cursor: int | None = None,
    ) -> dict[str, Any]:
        frame = self._experience_frame()
        cursor, presentation_tail = self._presentation_recovery_tail(
            presentation_cursor
        )
        return {
            "protocol": PROTOCOL_VERSION,
            "engine_version": "managym-python-adapter",
            "content_hash": self.content_hash,
            "asset_manifest_hash": self.asset_manifest_hash,
            "reason": reason,
            "frame": frame,
            "presentation_cursor": cursor,
            "presentation_tail": presentation_tail,
            "accepted_commands": list(self.accepted_commands.values()),
            "replay_cursor": len(self.trace.events) if self.trace else 0,
            "checkpoint": None,
        }

    def _wire_message(
        self,
        reason: str = "explicit_resync",
        presentation_cursor: int | None = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper carrying the protocol recovery envelope.

        Existing GUI/table consumers retain their observation fields while the
        protocol client reads the nested atomic frame. This wrapper disappears
        when semantic projection replaces the legacy table shape.
        """
        recovery = self.current_recovery(
            reason,
            self._last_presentation_cursor
            if presentation_cursor is None
            else presentation_cursor,
        )
        frame = recovery["frame"]
        prompt = self.published_prompt
        transient = self._drain_surface_metadata()
        payload: dict[str, Any] = {
            "type": "game_over" if frame["projection"]["game_over"] else "observation",
            "data": frame["projection"],
            "winner": frame["winner"],
            "actions": [] if prompt is None else prompt.actions,
            "action_space": frame["action_space"],
            "stops": frame["stops"],
            "protocol": PROTOCOL_VERSION,
            "frame": frame,
            "recovery": recovery,
        }
        if "deck_names" in frame:
            payload["deck_names"] = frame["deck_names"]
        if "asset_pack" in frame:
            payload["asset_pack"] = frame["asset_pack"]
        payload.update(transient)
        return payload

    def _reject_command(
        self, raw: dict[str, Any], code: str, *, recover: bool
    ) -> dict[str, Any]:
        prompt = self.published_prompt
        rejection = {
            "command_id": raw.get("command_id"),
            "code": code,
            "message": code.replace("_", " ").capitalize() + ".",
            "current_revision": self.revision,
            "current_prompt": None if prompt is None else prompt.prompt_id,
        }
        payload: dict[str, Any] = {
            "type": "command_outcome",
            "status": "rejected",
            "rejection": rejection,
        }
        if recover:
            payload["recovery"] = self.current_recovery("stale_command")
        return payload

    def _finalize_trace(self, end_reason: str) -> None:
        if self.trace is None or self.obs is None:
            return
        if self._trace_saved:
            return

        canonical = self.canonical_replay()
        study_provider = StudyForkProvider(canonical, self._study_roots)
        final_trace = replace(
            self.trace,
            final_observation=serialize_observation(self.obs),
            winner=_winner_for_hero(self.obs),
            end_reason=end_reason,
            canonical_replay=canonical.model_dump(mode="json"),
        )
        path = trace_store.save_trace(final_trace, self.trace_dir)
        self.trace = final_trace
        self.trace_id = path.stem
        self._study_provider = study_provider
        self._trace_saved = True

    def canonical_replay(self) -> CanonicalReplayV1:
        """Project the canonical decisions committed so far.

        The same replay and row identities are persisted when the game ends;
        no unplayed prompt is assigned an address.
        """
        return CanonicalReplayV1(
            version=CANONICAL_REPLAY_VERSION,
            replay_id=f"replay.{self.match_id}",
            match_id=self.match_id,
            content_hash=self.content_hash,
            asset_manifest_hash=self.asset_manifest_hash,
            decisions=[row.model_copy(deep=True) for row in self.canonical_decisions],
            presentation_tracks=[
                ViewerPresentationTrack(
                    viewer=viewer,
                    head=len(events),
                    events=deepcopy(events),
                )
                for viewer, events in sorted(self.canonical_presentation.items())
            ],
        )

    def study_fork_provider(self) -> StudyForkProvider:
        """Expose Game's validated retained-root capability to consumers.

        The provider owns replay lookup and isolated root cloning. Callers do
        not receive or inspect ``GameSession``'s retained-root storage.
        """

        if self._study_provider is not None:
            return self._study_provider
        return StudyForkProvider(self.canonical_replay(), self._study_roots)

    def resolve_live_advisor_decision(
        self,
        raw_address: str,
        authorized_viewer: int,
    ) -> Any:
        """Resolve a committed live-session decision through Game authority."""

        return self.study_fork_provider().resolve_advisor_decision(
            raw_address, authorized_viewer
        )

    def fork_study(self, raw_address: str) -> StudyBranch:
        """Fork one retained player-0 replay decision for ephemeral Study."""
        return self.study_fork_provider().fork(raw_address, HERO_PLAYER_INDEX)

    def retry_study(
        self,
        *,
        trace_id: str,
        replay: CanonicalReplayV1,
        raw_address: str,
        raw_command: Any,
        owner_viewer_id: str = "legacy",
    ) -> dict[str, Any]:
        """Commit one canonical command on an exact retained Study fork."""
        live_trace_id = self.trace_id or f"live.{self.match_id}"
        if live_trace_id != trace_id:
            raise StudyBranchUnavailableError(
                "Study roots are unavailable for this recording."
            )
        if any(
            attempt.owner_viewer_id == owner_viewer_id
            for attempt in self._study_attempts.values()
        ):
            raise StudyCommandUnavailableError("study_attempt_active")
        projection = project_replay(replay, HERO_PLAYER_INDEX)
        restored = restore_decision(replay, raw_address, HERO_PLAYER_INDEX)
        command = Command.model_validate(raw_command)
        prompt = restored.frame.prompt
        if (
            not command.command_id
            or prompt is None
            or command.match_id != restored.frame.match_id
            or command.expected_revision != restored.revision
            or command.prompt_id != prompt.id
            or not any(offer.id == command.offer_id for offer in restored.frame.offers)
        ):
            raise StudyCommandUnavailableError("study_command_identity_mismatch")

        branch = self.fork_study(raw_address)
        try:
            retry_projection, presentation = self._execute_study_command(
                branch,
                restored,
                command,
            )
        except Exception:
            with suppress(StudyBranchUnavailableError):
                branch.return_to_recorded()
            raise

        self._study_attempt_seq += 1
        attempt_id = self._id_factory(f"study-attempt-{self._study_attempt_seq}")
        while not attempt_id or attempt_id in self._study_attempts:
            attempt_id = secrets.token_urlsafe(18)
        attempt = StudyAttempt(
            attempt_id=attempt_id,
            owner_viewer_id=owner_viewer_id,
            trace_id=trace_id,
            address=raw_address,
            projection=projection,
            restored=restored,
            branch=branch,
            retry_command=command,
            retry_projection=retry_projection,
            retry_presentation=presentation,
        )
        self._study_attempts[attempt_id] = attempt
        return {
            "attempt_id": attempt_id,
            "trace_id": trace_id,
            "address": raw_address,
            "retry": {
                "command": command.model_dump(mode="json"),
                "projection": deepcopy(retry_projection),
                "presentation": deepcopy(presentation),
            },
            "return_to": {
                "address": raw_address,
                "ordinal": restored.ordinal,
                "presentation_cursor": restored.presentation_cursor,
            },
        }

    def reveal_study(
        self, attempt_id: str, owner_viewer_id: str | None = None
    ) -> dict[str, Any]:
        """Release evidence only after the attempt owns an accepted Retry."""
        attempt = self._require_study_attempt(attempt_id, owner_viewer_id)
        request = HistoricalStudyEvidenceRequest(
            projection=attempt.projection.model_copy(deep=True),
            source_replay_sha256=canonical_projection_sha256(attempt.projection),
            address=attempt.address,
            restored=attempt.restored.model_copy(deep=True),
        )
        raw_artifact = self._historical_evidence_provider.artifact_for(request)
        joined = join_historical_study_evidence(
            request,
            raw_artifact,
            allow_fixture_evidence=self._allow_fixture_study_evidence,
        )
        attempt.evidence = joined
        return {
            "attempt_id": attempt_id,
            "artifact": joined.artifact.model_dump(mode="json"),
        }

    def preview_study_plan(
        self,
        attempt_id: str,
        kind: StudyPlanKind,
        owner_viewer_id: str | None = None,
    ) -> dict[str, Any]:
        """Project one labelled plan without reusing the player's Retry fork."""
        attempt = self._require_study_attempt(attempt_id, owner_viewer_id)
        if attempt.evidence is None:
            raise StudyEvidenceUnavailableError("study_evidence_not_revealed")
        selection = select_study_plan(attempt.evidence.landmark, kind)

        if kind == "played":
            projection = self._recorded_continuation_projection(attempt)
            presentation = [
                event.model_dump(mode="json") for event in attempt.restored.continuation
            ]
        else:
            preview = self.fork_study(attempt.address)
            try:
                projection, presentation = self._execute_study_command(
                    preview,
                    attempt.restored,
                    selection.command,
                )
            finally:
                with suppress(StudyBranchUnavailableError):
                    preview.return_to_recorded()

        return {
            "attempt_id": attempt_id,
            "plan": kind,
            "alternative_id": selection.alternative_id,
            "command": selection.command.model_dump(mode="json"),
            "offer": selection.offer.model_dump(mode="json"),
            "projection": deepcopy(projection),
            "presentation": deepcopy(presentation),
            "return_to": {
                "address": attempt.address,
                "ordinal": attempt.restored.ordinal,
                "presentation_cursor": attempt.restored.presentation_cursor,
            },
        }

    def return_from_study(
        self, attempt_id: str, owner_viewer_id: str | None = None
    ) -> dict[str, Any]:
        """Close one branch and restore the exact canonical decision."""
        attempt = self._require_study_attempt(attempt_id, owner_viewer_id)
        self._study_attempts.pop(attempt_id)
        if attempt is None:
            raise StudyAttemptNotFoundError("study_attempt_not_found")
        receipt = attempt.branch.return_to_recorded()
        restored = RestoredReplayDecision.model_validate(
            receipt.model_dump(mode="python", exclude={"source_digest", "execution"})
        )
        if restored != attempt.restored:
            raise RuntimeError("Study return identity drifted.")
        return restored.model_dump(mode="json")

    def has_study_attempt(
        self, attempt_id: str, owner_viewer_id: str | None = None
    ) -> bool:
        attempt = self._study_attempts.get(attempt_id)
        return attempt is not None and (
            owner_viewer_id is None or attempt.owner_viewer_id == owner_viewer_id
        )

    def _require_study_attempt(
        self, attempt_id: str, owner_viewer_id: str | None = None
    ) -> StudyAttempt:
        attempt = self._study_attempts.get(attempt_id)
        if attempt is None or (
            owner_viewer_id is not None and attempt.owner_viewer_id != owner_viewer_id
        ):
            raise StudyAttemptNotFoundError("study_attempt_not_found")
        return attempt

    def _execute_study_command(
        self,
        branch: StudyBranch,
        restored: RestoredReplayDecision,
        command: Command,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        canonical_offer = next(
            (offer for offer in restored.frame.offers if offer.id == command.offer_id),
            None,
        )
        if canonical_offer is None:
            raise StudyCommandUnavailableError("study_command_offer_missing")

        try:
            structured = branch.structured_offers()
        except (managym.AgentError, StudyBranchUnavailableError) as exc:
            raise StudyCommandUnavailableError("study_command_not_structured") from exc
        matches = [
            offer
            for offer in structured.get("offers", [])
            if offer.get("actor") == canonical_offer.actor
            and offer.get("verb") == canonical_offer.verb.value
            and offer.get("source")
            == (
                None
                if canonical_offer.source is None
                else canonical_offer.source.model_dump(mode="json")
            )
            and offer.get("choices")
            == [choice.model_dump(mode="json") for choice in canonical_offer.choices]
        ]
        if len(matches) != 1:
            raise StudyCommandUnavailableError("study_command_not_structured")

        before = branch.current_observation()
        projector = PresentationProjector()
        projector.note_action(
            before,
            {
                "type": canonical_offer.action_type,
                "focus": list(canonical_offer.focus),
            },
            actor_index=restored.viewer,
        )
        try:
            observation, _, _, _, _, legacy_actions = branch.submit(
                {
                    "offer_id": matches[0]["id"],
                    "answers": command.model_dump(mode="json")["answers"],
                }
            )
        except (managym.AgentError, StudyBranchUnavailableError) as exc:
            raise StudyCommandUnavailableError("study_command_rejected") from exc
        projector.observe(observation)
        presentation = projector.drain(
            from_revision=restored.revision,
            to_revision=restored.revision + max(1, legacy_actions),
            caused_by=command.command_id,
        )
        return viewer_view(observation, restored.viewer), presentation

    @staticmethod
    def _recorded_continuation_projection(
        attempt: StudyAttempt,
    ) -> dict[str, Any]:
        later = next(
            (
                row
                for row in attempt.projection.decisions
                if row.ordinal > attempt.restored.ordinal
            ),
            None,
        )
        frame = attempt.restored.frame if later is None else later.frame
        return deepcopy(frame.projection.model_dump(mode="json"))

    def _close_study_attempts(self) -> None:
        for attempt in self._study_attempts.values():
            with suppress(StudyBranchUnavailableError):
                attempt.branch.return_to_recorded()
        self._study_attempts = {}

    def close(self, end_reason: str) -> None:
        self._close_study_attempts()
        if self.trace is not None and not self._trace_saved:
            self._finalize_trace(end_reason=end_reason)

        self.env = None
        self.obs = None
        self.villain_policy = None


def _error_message(message: str) -> dict[str, str]:
    return {"type": "error", "message": message}


class TableDispatchError(ValueError):
    """A closed table-control request failed before match authority."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _control_error(error: TableDispatchError) -> dict[str, str]:
    return {"type": "control_error", "code": error.code, "message": str(error)}


def _new_game_session() -> GameSession:
    """Construct the normal runtime authority with unavailable Study evidence."""
    return GameSession()


def _invite_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_session_record() -> tuple[SessionRecord, str]:
    while True:
        session_id = secrets.token_urlsafe(12)
        if session_id not in SESSION_REGISTRY:
            break

    invite_token = secrets.token_urlsafe(24)
    record = SessionRecord(
        session_id=session_id,
        resume_token=secrets.token_urlsafe(24),
        game=_new_game_session(),
        watcher_invite_digest=_invite_digest(invite_token),
    )
    SESSION_REGISTRY[session_id] = record
    return record, invite_token


def _drop_session(session_id: str, end_reason: str) -> None:
    record = SESSION_REGISTRY.pop(session_id, None)
    if record is None:
        return
    record.game.close(end_reason=end_reason)
    for participant in record.participants.values():
        participant.websocket = None
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


def _participant_from_resume(
    raw_session_id: Any,
    raw_resume_token: Any,
) -> tuple[SessionRecord, ParticipantLease]:
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise ValueError("resume messages require a non-empty 'session_id'.")
    if not isinstance(raw_resume_token, str) or not raw_resume_token:
        raise ValueError("resume messages require a non-empty 'resume_token'.")

    record = SESSION_REGISTRY.get(raw_session_id)
    if record is None:
        raise ValueError("Session not found or expired. Start a new game.")
    participants = [
        participant
        for participant in record.participants.values()
        if secrets.compare_digest(participant.resume_token, raw_resume_token)
    ]
    if len(participants) != 1:
        raise ValueError("Invalid resume credentials. Start a new game.")
    if record.expires_at <= _now_utc():
        _drop_session(raw_session_id, end_reason=SESSION_EXPIRED_END_REASON)
        raise ValueError("Session has expired. Start a new game.")
    return record, participants[0]


def _participant_from_invite(
    raw_table_id: Any,
    raw_invite_token: Any,
) -> tuple[SessionRecord, ParticipantLease]:
    if not isinstance(raw_table_id, str) or not raw_table_id:
        raise ValueError("join_table messages require a non-empty 'table_id'.")
    if not isinstance(raw_invite_token, str) or not raw_invite_token:
        raise ValueError("join_table messages require a non-empty 'invite_token'.")
    record = SESSION_REGISTRY.get(raw_table_id)
    if record is None or record.expires_at <= _now_utc():
        raise ValueError("Table not found or expired.")
    digest = record.watcher_invite_digest
    if digest is None or not secrets.compare_digest(
        digest, _invite_digest(raw_invite_token)
    ):
        raise ValueError("Invalid or consumed watcher invite.")
    if len(record.participants) >= 2:
        raise ValueError("This table already has its permitted watcher.")
    viewer_id = f"viewer.{secrets.token_urlsafe(12)}"
    participant = ParticipantLease(
        viewer_id=viewer_id,
        resume_token=secrets.token_urlsafe(24),
        role=ViewerRole.WATCHER,
    )
    record.participants[viewer_id] = participant
    record.watcher_invite_digest = None
    record.table_revision += 1
    record.touch()
    return record, participant


def _parse_presentation_cursor(raw_cursor: Any) -> int | None:
    if raw_cursor is None:
        return None
    if (
        isinstance(raw_cursor, bool)
        or not isinstance(raw_cursor, int)
        or raw_cursor < 0
        or raw_cursor > MAX_UINT64
    ):
        raise ValueError(
            "resume presentation_cursor must be an unsigned 64-bit integer."
        )
    return raw_cursor


def _visible_beliefs(
    record: SessionRecord, participant: ParticipantLease
) -> list[dict[str, Any]]:
    visible = [deepcopy(value) for value in record.shared_beliefs.values()]
    visible.extend(deepcopy(value) for value in participant.personal_beliefs.values())
    return sorted(visible, key=lambda value: str(value["id"]))


def _live_decision_summaries(record: SessionRecord) -> list[dict[str, Any]]:
    if not record.game.canonical_decisions:
        return []
    projection = project_replay(record.game.canonical_replay(), HERO_PLAYER_INDEX)
    addressed = projection_with_addresses(projection)
    return [
        {
            "address": decision["address"],
            "ordinal": decision["ordinal"],
            "revision": decision["revision"],
            "prompt_id": decision["prompt_id"],
            "offer_id": decision["offer_id"],
        }
        for decision in addressed["decisions"]
    ]


def _table_opponent_label(record: SessionRecord) -> str | None:
    trace = record.game.trace
    if trace is None:
        return None
    config = trace.config
    if config.villain_type == "search":
        return f"Search {config.villain_sims}"
    if config.villain_type == "checkpoint":
        return "Checkpoint"
    return config.villain_type.capitalize()


def _table_snapshot(
    record: SessionRecord,
    participant: ParticipantLease,
    *,
    watcher_invite: str | None = None,
) -> dict[str, Any]:
    capabilities = list(ROLE_CAPABILITIES[participant.role])
    snapshot = TableSnapshot(
        table_id=record.session_id,
        table_revision=record.table_revision,
        mode=(
            "study"
            if record.game.obs is not None and record.game.obs.game_over
            else "live"
        ),
        access=ViewerAccess(
            identity=ViewerIdentity(
                viewer_id=participant.viewer_id,
                table_id=record.session_id,
            ),
            role=participant.role,
            capabilities=capabilities,
            grant_revision=participant.grant_revision,
        ),
        participants=[
            {
                "viewer_id": candidate.viewer_id,
                "role": candidate.role,
                "connected": candidate.websocket is not None,
            }
            for candidate in sorted(
                record.participants.values(), key=lambda value: value.viewer_id
            )
        ],
        beliefs=_visible_beliefs(record, participant),
        decisions=_live_decision_summaries(record),
        opponent_label=_table_opponent_label(record),
        watcher_invite=(
            f"#table={record.session_id}&watch={watcher_invite}"
            if watcher_invite is not None and participant.role == ViewerRole.PILOT
            else None
        ),
    )
    return snapshot.model_dump(mode="json")


def _response_with_participant(
    response: dict[str, Any],
    record: SessionRecord,
    participant: ParticipantLease,
    *,
    watcher_invite: str | None = None,
) -> dict[str, Any]:
    payload = deepcopy(response)
    payload["table"] = _table_snapshot(
        record, participant, watcher_invite=watcher_invite
    )
    if response.get("type") in {"observation", "game_over", "table_snapshot"}:
        payload["session_id"] = record.session_id
        payload["resume_token"] = participant.resume_token
        payload["participant_id"] = participant.viewer_id
    return payload


async def _attach_participant_websocket(
    record: SessionRecord,
    participant: ParticipantLease,
    websocket: WebSocket,
) -> None:
    previous_websocket = participant.websocket
    participant.websocket = websocket
    if participant.role == ViewerRole.PILOT:
        record.resume_token = participant.resume_token
        record.websocket = websocket
    record.touch()
    if previous_websocket is not None and previous_websocket is not websocket:
        with suppress(Exception):
            await previous_websocket.close(code=4000)


async def _get_or_create_attached_participant(
    attached_table_id: str | None,
    attached_viewer_id: str | None,
    websocket: WebSocket,
) -> tuple[SessionRecord, ParticipantLease, str | None]:
    if attached_table_id is not None and attached_viewer_id is not None:
        existing = SESSION_REGISTRY.get(attached_table_id)
        participant = (
            None if existing is None else existing.participants.get(attached_viewer_id)
        )
        if existing is not None and participant is not None:
            await _attach_participant_websocket(existing, participant, websocket)
            return existing, participant, None

    record, invite_token = _create_session_record()
    participant = record.pilot()
    await _attach_participant_websocket(record, participant, websocket)
    return record, participant, invite_token


def _detach_participant_websocket(
    table_id: str,
    viewer_id: str,
    websocket: WebSocket,
) -> None:
    record = SESSION_REGISTRY.get(table_id)
    if record is None:
        return
    participant = record.participants.get(viewer_id)
    if participant is None:
        return
    if participant.websocket is websocket:
        participant.websocket = None
        if participant.role == ViewerRole.PILOT:
            record.websocket = None
        record.touch()


DISPATCH_CAPABILITIES: dict[str, TableCapability] = {
    "new_game": TableCapability.CONFIGURE_MATCH,
    "rematch": TableCapability.CONFIGURE_MATCH,
    "command": TableCapability.SUBMIT_LIVE_COMMAND,
    "action": TableCapability.SUBMIT_LIVE_COMMAND,
    "pass_turn": TableCapability.SUBMIT_LIVE_COMMAND,
    "set_stops": TableCapability.CONFIGURE_MATCH,
    "transfer_pilot": TableCapability.TRANSFER_PILOT,
    "author_belief": TableCapability.AUTHOR_BELIEF,
    "share_belief": TableCapability.SHARE_BELIEF,
    "restore_decision": TableCapability.EXPLORE_STUDY,
    "retry_decision": TableCapability.EXPLORE_STUDY,
    "branch_reveal": TableCapability.EXPLORE_STUDY,
    "branch_preview": TableCapability.EXPLORE_STUDY,
    "return_from_branch": TableCapability.EXPLORE_STUDY,
    "return_to_live": TableCapability.EXPLORE_STUDY,
}

if set(DISPATCH_CAPABILITIES) | {"join_table", "resume"} != set(REQUEST_TYPES):
    raise RuntimeError("testing-house request vocabulary and dispatch drifted")


def _parse_table_request(raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise TableDispatchError(
            "invalid_message", "WebSocket payload must be a JSON object."
        )
    message_type = raw.get("type")
    if not isinstance(message_type, str) or message_type not in REQUEST_TYPES:
        raise TableDispatchError(
            "unsupported_message", f"Unsupported message type: {message_type}"
        )
    try:
        return REQUEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise TableDispatchError(
            "invalid_message", f"Invalid {message_type} message."
        ) from exc


def _authorize_table_request(
    record: SessionRecord,
    participant: ParticipantLease,
    request: Any,
) -> None:
    message_type = request.type
    grant_revision = getattr(request, "grant_revision", None)
    if not (grant_revision is None and len(record.participants) == 1) and (
        grant_revision != participant.grant_revision
    ):
        raise TableDispatchError(
            "stale_grant", "Participant role changed; recover current table access."
        )
    capability = DISPATCH_CAPABILITIES[message_type]
    if capability not in ROLE_CAPABILITIES[participant.role]:
        raise TableDispatchError(
            "forbidden", f"{participant.role.value.capitalize()} cannot {message_type}."
        )


def _author_belief(
    record: SessionRecord,
    participant: ParticipantLease,
    scenario_id: str,
) -> dict[str, Any]:
    meta = advice_meta().model_dump(mode="json")
    scenarios = {
        str(scenario["landmark_id"]): scenario for scenario in meta["scenarios"]
    }
    if scenario_id not in scenarios:
        raise TableDispatchError("not_found", "Pinned belief scenario not found.")
    record.table_revision += 1
    belief_id = f"belief.{secrets.token_urlsafe(12)}"
    belief = {
        "id": belief_id,
        "author_viewer_id": participant.viewer_id,
        "source": {
            "decision_address": meta["address"],
            "gam6_scenario_id": scenario_id,
            "advice_identity": meta["identity"],
        },
        "audience": {"kind": "personal"},
        "provenance": {
            "kind": "player_authored",
            "created_at_table_revision": record.table_revision,
            "shared_at_table_revision": None,
        },
    }
    participant.personal_beliefs[belief_id] = belief
    return deepcopy(belief)


def _share_belief(
    record: SessionRecord,
    participant: ParticipantLease,
    belief_id: str,
) -> dict[str, Any]:
    belief = participant.personal_beliefs.pop(belief_id, None)
    if belief is None:
        raise TableDispatchError("not_found", "Personal belief not found.")
    record.table_revision += 1
    belief["audience"] = {"kind": "table", "table_id": record.session_id}
    belief["provenance"]["shared_at_table_revision"] = record.table_revision
    record.shared_beliefs[belief_id] = belief
    return deepcopy(belief)


def _dispatch_table_request(
    record: SessionRecord,
    participant: ParticipantLease,
    request: Any,
) -> tuple[dict[str, Any], bool]:
    """Authorize and execute one closed operation under the table lock."""
    if isinstance(request, CommandMessage):
        command_id = request.command.command_id
        if command_id in record.game.accepted_commands:
            submitter = record.accepted_command_submitters.get(command_id)
            if submitter is None and len(record.participants) == 1:
                submitter = participant.viewer_id
                record.accepted_command_submitters[command_id] = submitter
            if submitter != participant.viewer_id:
                raise TableDispatchError(
                    "forbidden", "Command id belongs to another participant."
                )
            return record.game.hero_command(
                request.command.model_dump(mode="json")
            ), False

    _authorize_table_request(record, participant, request)

    if isinstance(request, (NewGameMessage, RematchMessage)):
        response = record.game.new_game(request.config)
        record.accepted_command_submitters.clear()
        record.shared_beliefs.clear()
        for candidate in record.participants.values():
            candidate.personal_beliefs.clear()
            candidate.study_attempts.clear()
        record.table_revision += 1
        return response, True
    if isinstance(request, CommandMessage):
        response = record.game.hero_command(request.command.model_dump(mode="json"))
        if response.get("status") == "accepted":
            record.accepted_command_submitters[request.command.command_id] = (
                participant.viewer_id
            )
        return response, True
    if isinstance(request, ActionMessage):
        return record.game.hero_action(request.index), True
    if isinstance(request, PassTurnMessage):
        return record.game.pass_turn(), True
    if isinstance(request, SetStopsMessage):
        return (
            record.game.set_stops(
                request.stops,
                request.stop_on_stack,
                request.auto_pass,
            ),
            True,
        )
    if isinstance(request, TransferPilotMessage):
        target = record.participants.get(request.target_viewer_id)
        if target is None or target is participant:
            raise TableDispatchError("not_found", "Watcher participant not found.")
        if target.role != ViewerRole.WATCHER:
            raise TableDispatchError("conflict", "Target is not the watcher.")
        participant.role = ViewerRole.WATCHER
        target.role = ViewerRole.PILOT
        participant.grant_revision += 1
        target.grant_revision += 1
        record.resume_token = target.resume_token
        record.websocket = target.websocket
        record.table_revision += 1
        return {"type": "role_changed"}, True
    if isinstance(request, AuthorBeliefMessage):
        belief = _author_belief(record, participant, request.scenario_id)
        return {"type": "belief_changed", "belief": belief}, False
    if isinstance(request, ShareBeliefMessage):
        belief = _share_belief(record, participant, request.belief_id)
        return {"type": "belief_changed", "belief": belief}, True
    if isinstance(request, RestoreDecisionMessage):
        replay = record.game.canonical_replay()
        try:
            restored = restore_decision(replay, request.address, HERO_PLAYER_INDEX)
        except (InvalidAddressError, DecisionNotFoundError) as exc:
            raise TableDispatchError("not_found", "Decision not found.") from exc
        return {
            "type": "decision_restored",
            "address": request.address,
            "restored": restored.model_dump(mode="json"),
        }, False
    if isinstance(request, RetryDecisionMessage):
        replay = record.game.canonical_replay()
        trace_id = record.game.trace_id or f"live.{record.game.match_id}"
        try:
            retry = record.game.retry_study(
                trace_id=trace_id,
                replay=replay,
                raw_address=request.address,
                raw_command=request.command.model_dump(mode="json"),
                owner_viewer_id=participant.viewer_id,
            )
        except (ValueError, StudyBranchUnavailableError) as exc:
            raise TableDispatchError("conflict", str(exc)) from exc
        participant.study_attempts.add(str(retry["attempt_id"]))
        return {
            "type": "branch_updated",
            "attempt_id": retry["attempt_id"],
            "phase": "retry",
            "payload": retry,
        }, False
    if isinstance(request, BranchRevealMessage):
        if request.attempt_id not in participant.study_attempts:
            raise TableDispatchError("not_found", "Study attempt not found.")
        try:
            reveal = record.game.reveal_study(request.attempt_id, participant.viewer_id)
        except ValueError as exc:
            raise TableDispatchError("conflict", str(exc)) from exc
        return {
            "type": "branch_updated",
            "attempt_id": request.attempt_id,
            "phase": "revealed",
            "payload": reveal,
        }, False
    if isinstance(request, BranchPreviewMessage):
        if request.attempt_id not in participant.study_attempts:
            raise TableDispatchError("not_found", "Study attempt not found.")
        try:
            preview = record.game.preview_study_plan(
                request.attempt_id, request.plan, participant.viewer_id
            )
        except ValueError as exc:
            raise TableDispatchError("conflict", str(exc)) from exc
        return {
            "type": "branch_updated",
            "attempt_id": request.attempt_id,
            "phase": "preview",
            "payload": preview,
        }, False
    if isinstance(request, ReturnFromBranchMessage):
        if request.attempt_id not in participant.study_attempts:
            raise TableDispatchError("not_found", "Study attempt not found.")
        try:
            restored = record.game.return_from_study(
                request.attempt_id, participant.viewer_id
            )
        except ValueError as exc:
            raise TableDispatchError("not_found", str(exc)) from exc
        participant.study_attempts.discard(request.attempt_id)
        return {"type": "branch_returned", "restored": restored}, False
    if isinstance(request, ReturnToLiveMessage):
        return {"type": "table_snapshot"}, False
    raise RuntimeError(f"closed dispatch missing handler for {request.type}")


async def _send_table_response(
    record: SessionRecord,
    initiator: ParticipantLease,
    response: dict[str, Any],
    *,
    broadcast: bool,
    watcher_invite: str | None = None,
) -> None:
    recipients = list(record.participants.values()) if broadcast else [initiator]
    for recipient in recipients:
        socket = recipient.websocket
        if socket is None:
            continue
        payload = _response_with_participant(
            response,
            record,
            recipient,
            watcher_invite=(watcher_invite if recipient is initiator else None),
        )
        try:
            await socket.send_json(payload)
        except Exception:
            if recipient.websocket is socket:
                recipient.websocket = None


def _current_table_response(record: SessionRecord) -> dict[str, Any]:
    if record.game.obs is None:
        return {"type": "table_snapshot"}
    return record.game.current_message()


@app.websocket("/ws/play")
async def play_socket(websocket: WebSocket) -> None:
    _cleanup_expired_sessions()
    await websocket.accept()
    attached_table_id: str | None = None
    attached_viewer_id: str | None = None

    try:
        while True:
            try:
                request = await websocket.receive_json()
            except WebSocketDisconnect:
                if attached_table_id is not None and attached_viewer_id is not None:
                    _detach_participant_websocket(
                        attached_table_id, attached_viewer_id, websocket
                    )
                raise

            _cleanup_expired_sessions()
            try:
                if isinstance(request, dict) and request.get("type") in {
                    "join_table",
                    "resume",
                }:
                    _parse_presentation_cursor(request.get("presentation_cursor"))
                parsed = _parse_table_request(request)
                if isinstance(parsed, JoinTableMessage):
                    presentation_cursor = _parse_presentation_cursor(
                        parsed.presentation_cursor
                    )
                    record, participant = _participant_from_invite(
                        parsed.table_id, parsed.invite_token
                    )
                    await _attach_participant_websocket(record, participant, websocket)
                    attached_table_id = record.session_id
                    attached_viewer_id = participant.viewer_id
                    response = (
                        record.game.current_message(presentation_cursor)
                        if record.game.obs is not None
                        else {"type": "table_snapshot"}
                    )
                    await _send_table_response(
                        record, participant, response, broadcast=True
                    )
                    continue
                if isinstance(parsed, ResumeMessage):
                    presentation_cursor = _parse_presentation_cursor(
                        parsed.presentation_cursor
                    )
                    record, participant = _participant_from_resume(
                        parsed.session_id, parsed.resume_token
                    )
                    await _attach_participant_websocket(record, participant, websocket)
                    attached_table_id = record.session_id
                    attached_viewer_id = participant.viewer_id
                    response = (
                        record.game.current_message(presentation_cursor)
                        if record.game.obs is not None
                        else {"type": "table_snapshot"}
                    )
                    await _send_table_response(
                        record, participant, response, broadcast=True
                    )
                    continue

                invite_token: str | None = None
                if isinstance(parsed, NewGameMessage) and attached_table_id is None:
                    (
                        record,
                        participant,
                        invite_token,
                    ) = await _get_or_create_attached_participant(
                        attached_table_id,
                        attached_viewer_id,
                        websocket,
                    )
                    attached_table_id = record.session_id
                    attached_viewer_id = participant.viewer_id
                else:
                    if attached_table_id is None or attached_viewer_id is None:
                        raise ValueError("No active game session. Send new_game first.")
                    record = SESSION_REGISTRY.get(attached_table_id)
                    participant = (
                        None
                        if record is None
                        else record.participants.get(attached_viewer_id)
                    )
                    if record is None or participant is None:
                        attached_table_id = None
                        attached_viewer_id = None
                        raise ValueError("Session expired. Start a new game.")

                async with record.lock:
                    response, broadcast = _dispatch_table_request(
                        record, participant, parsed
                    )
                    record.touch()
                    await _send_table_response(
                        record,
                        participant,
                        response,
                        broadcast=broadcast,
                        watcher_invite=invite_token,
                    )
                continue
            except TableDispatchError as exc:
                await websocket.send_json(_control_error(exc))
                continue
            except ValueError as exc:
                await websocket.send_json(_error_message(str(exc)))
                continue
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if attached_table_id is not None:
            _drop_session(attached_table_id, end_reason="error")
        with suppress(Exception):
            await websocket.send_json(_error_message(str(exc)))
        with suppress(Exception):
            await websocket.close()


@app.get("/api/traces")
async def list_traces() -> list[dict[str, Any]]:
    return trace_store.list_trace_summaries()


def _load_trace_replay(trace_id: str) -> CanonicalReplayV1:
    try:
        return load_canonical_replay(trace_store.load_trace(trace_id))
    except ValueError as exc:
        if isinstance(exc, CanonicalReplayUnavailableError):
            raise HTTPException(
                status_code=409,
                detail="canonical_replay_unavailable",
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="decision_not_found") from exc


def _live_record_for_trace(trace_id: str) -> SessionRecord:
    _cleanup_expired_sessions()
    matches = [
        record
        for record in SESSION_REGISTRY.values()
        if record.game.trace_id == trace_id
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=409, detail="study_branch_unavailable")
    record = matches[0]
    record.touch()
    return record


def _live_record_for_attempt(attempt_id: str) -> SessionRecord:
    _cleanup_expired_sessions()
    matches = [
        record
        for record in SESSION_REGISTRY.values()
        if record.game.has_study_attempt(attempt_id)
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="study_attempt_not_found")
    record = matches[0]
    record.touch()
    return record


def _rest_study_owner(
    record: SessionRecord,
    participant_token: str | None,
) -> str | None:
    """Authorize legacy single-user Study or one exact shared participant."""
    if len(record.participants) == 1 and participant_token is None:
        return None
    if participant_token is None:
        raise HTTPException(status_code=403, detail="participant_credential_required")
    matches = [
        participant
        for participant in record.participants.values()
        if secrets.compare_digest(participant.resume_token, participant_token)
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=403, detail="participant_credential_invalid")
    if TableCapability.EXPLORE_STUDY not in ROLE_CAPABILITIES[matches[0].role]:
        raise HTTPException(status_code=403, detail="participant_forbidden")
    return matches[0].viewer_id


@app.get("/api/traces/{trace_id}/decisions")
async def get_trace_decisions(trace_id: str) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    return projection_with_addresses(project_replay(replay, HERO_PLAYER_INDEX))


@app.get("/api/traces/{trace_id}/decisions/{address}")
async def get_trace_decision(trace_id: str, address: str) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    try:
        restored = restore_decision(replay, address, HERO_PLAYER_INDEX)
    except InvalidAddressError as exc:
        raise HTTPException(status_code=400, detail="invalid_address") from exc
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="decision_not_found") from exc
    return restored.model_dump(mode="json")


@app.post("/api/traces/{trace_id}/decisions/{address}/retry")
async def retry_trace_decision(
    trace_id: str,
    address: str,
    payload: dict[str, Any],
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    record = _live_record_for_trace(trace_id)
    owner_viewer_id = _rest_study_owner(record, x_etude_participant_token)
    try:
        kwargs: dict[str, Any] = {}
        if owner_viewer_id is not None:
            kwargs["owner_viewer_id"] = owner_viewer_id
        return record.game.retry_study(
            trace_id=trace_id,
            replay=replay,
            raw_address=address,
            raw_command=payload.get("command"),
            **kwargs,
        )
    except InvalidAddressError as exc:
        raise HTTPException(status_code=400, detail="invalid_address") from exc
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="decision_not_found") from exc
    except StudyBranchUnavailableError as exc:
        raise HTTPException(status_code=409, detail="study_branch_unavailable") from exc
    except StudyCommandUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="study_command_invalid") from exc


@app.post("/api/study-attempts/{attempt_id}/reveal")
async def reveal_study_attempt(
    attempt_id: str,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    record = _live_record_for_attempt(attempt_id)
    owner_viewer_id = _rest_study_owner(record, x_etude_participant_token)
    try:
        return record.game.reveal_study(attempt_id, owner_viewer_id)
    except StudyEvidenceUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail="study_evidence_unavailable",
        ) from exc
    except StudyEvidenceMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/study-attempts/{attempt_id}/preview")
async def preview_study_attempt(
    attempt_id: str,
    payload: dict[str, Any],
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    raw_plan = payload.get("plan")
    if raw_plan not in {"played", "policy", "search"}:
        raise HTTPException(status_code=400, detail="study_plan_invalid")
    record = _live_record_for_attempt(attempt_id)
    owner_viewer_id = _rest_study_owner(record, x_etude_participant_token)
    try:
        return record.game.preview_study_plan(attempt_id, raw_plan, owner_viewer_id)
    except StudyEvidenceUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (StudyPlanUnavailableError, StudyCommandUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/study-attempts/{attempt_id}/return")
async def return_from_study_attempt(
    attempt_id: str,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    record = _live_record_for_attempt(attempt_id)
    owner_viewer_id = _rest_study_owner(record, x_etude_participant_token)
    try:
        return record.game.return_from_study(attempt_id, owner_viewer_id)
    except StudyAttemptNotFoundError as exc:
        raise HTTPException(status_code=404, detail="study_attempt_not_found") from exc


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


@app.get("/api/advice")
async def get_advice_meta() -> dict[str, Any]:
    """Bootstrap for the shared decision-advice surface.

    Returns the compatibility bootstrap plus the checked advice-v1 comparison
    request consumed by both live play and Study.
    """
    return advice_meta().model_dump(mode="json")


@app.post("/api/advice", response_model=None)
async def post_advice(payload: AdviceRequest) -> dict[str, Any] | Response:
    """Serve legacy GAM-6 or canonical advice-v1 through one endpoint.

    Versioned success bytes are returned from the provider serializer without
    framework re-encoding. Domain incompatibilities remain typed unavailable;
    malformed request bodies use ordinary transport validation.
    """
    if not payload.is_legacy:
        return Response(
            content=request_versioned_fixture_advice(payload),
            media_type="application/json",
        )
    assert payload.scenario_id is not None
    assert payload.identity is not None
    response = request_advice(payload.address, payload.scenario_id, payload.identity)
    return response.model_dump(mode="json")
