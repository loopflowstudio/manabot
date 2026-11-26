"""
server.py
FastAPI server for interactive managym play over WebSocket.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import concurrent.futures
from contextlib import ExitStack, suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any, Callable, Literal, TypedDict
import warnings

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from starlette.websockets import WebSocketState

import managym
from managym.academy.card_art import card_art
from managym.academy.card_faces import CardFace, card_faces
from managym.bots import BotDecision, BotPolicy, FlatSearchPolicy, choose_bot
from managym.combat import CombatPlan, preview_combat
from managym.decision import (
    Command as SemanticCommand,
    DecisionFrame as SemanticDecisionFrame,
    SemanticTransition,
    TransitionReceipt,
)
from managym.enums import (
    ActionSpaceEnum,
    EventTypeEnum,
    StepEnum,
)
from managym.experience import (
    PROTOCOL_VERSION,
    UNVERSIONED_CONTENT_HASH,
    Command,
    ExperienceFrame,
    PresentationEvent,
)
from managym.experience_projection import (
    decision_surface_view,
    enum_name,
    object_records,
    observation_definition_id,
    observation_object_id,
    offer_verb,
    redact_observation,
    serialize_observation,
)
from managym.payment import preview_payment
from managym.policy import PolicyRunner
from managym.search import (
    SearchConfig,
    SearchError,
    descriptor_search_unavailable,
)

from . import trace as trace_store, villain as villain_module
from .advice import (
    AdviceRequest,
    AdvisorUnavailable,
    advice_meta,
    request_advice,
    request_versioned_fixture_advice,
)
from .analysis import (
    AnalysedDecision,
    AnalysisChoices,
    AnalysisStop,
    LiveAnalysis,
    agent_line,
    analysed_events,
    analysing_policies,
    analysis_choices,
    policy_admissions,
    run_analysis,
)
from .append_json import encode_list, splice
from .attempts import (
    AttemptStore,
    RecordedPlayer,
    RetainedAttempt,
    decode_cursor,
    page_cursor,
    shown_status,
)
from .curated_pack import (
    CURATED_PACK,
    CURATED_PACK_CATALOG,
    ELEMENTAL_WORLD_PACK,
    JEONG_INCREMENT_PACK,
    curated_pack_for_matchup,
)
from .deck_import import DeckImportInput
from .deck_pools import (
    deck_file_payload,
    deck_pool_catalog,
    deck_pools,
    deck_to_source,
    ensure_default_deck_pools,
    is_pool_key,
    resolve_deck_pool,
    set_deck_pools,
)
from .game_setup import continuation_config, setup_draft, shown_setup
from .numbers import game_numbers
from .opponent import (
    Opponent,
    catalog_opponent,
    opponent_catalog,
    opponent_kind,
    policy_registry,
    policy_settings_fit,
)
from .play_protocol import (
    REQUEST_ADAPTER,
    REQUEST_TYPES,
    ActionMessage,
    CommandMessage,
    NewGameMessage,
    PassTurnMessage,
    RematchMessage,
    ResumeMessage,
    RetryMirrorMessage,
    SetStopsMessage,
    ShowAnalysisMessage,
)
from .presentation import PresentationProjector
from .recording import Recording, person_board, read_cache, step_matches
from .replay_index import (
    CANONICAL_REPLAY_VERSION,
    CanonicalReplayProjectionV1,
    CanonicalReplayUnavailableError,
    CanonicalReplayV1,
    DecisionAddressV2,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecision,
    RestoredReplayDecision,
    ViewerPresentationTrack,
    canonical_projection_sha256,
    load_canonical_replay,
    parse_decision_address,
    project_replay,
    projection_with_addresses,
    restore_decision,
)
from .step_boards import MAX_STEP_BOARDS, StepBoard, WireStepBoard, wire_boards
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
from .trace import GameConfig, PlayedFrom, Trace, TraceEvent
from .villain import build_villain_policy
from .work import Work, WorkStatus

# Interactive remains a lightweight mirror of manabot.verify.util. Curated
# manifests are checked against the compiled rules setup without importing torch.
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
    **{
        key: dict(ELEMENTAL_WORLD_PACK.player_config(key, key).decklist)
        for key in ELEMENTAL_WORLD_PACK.deck_ids
    },
}
DECK_DISPLAY_NAMES = {
    "interactive": "Interactive",
    CURATED_PACK.hero_deck_id: CURATED_PACK.hero_display_name,
    CURATED_PACK.villain_deck_id: CURATED_PACK.villain_display_name,
    JEONG_INCREMENT_PACK.hero_deck_id: JEONG_INCREMENT_PACK.hero_display_name,
    "custom": "Custom",
    "imported": "Imported deck",
    **{
        row["deck_id"]: row["display_name"]
        for row in ELEMENTAL_WORLD_PACK.manifest["roster"]
    },
}
# Default matchup: the Milestone-1 two-deck slice, UR as hero vs GW villain.
DEFAULT_HERO_DECK_NAME = CURATED_PACK.hero_deck_id
DEFAULT_VILLAIN_DECK_NAME = CURATED_PACK.villain_deck_id
CLOSED_DECK_NAME = "Closed decklist"
# Backwards-compatible alias (tests and older callers).
DEFAULT_DECK = INTERACTIVE_DECK

MAX_AUTOPLAY_STEPS = 1024
VILLAIN_TYPES = {"passive", "random", "search", "policy"}
MAX_ACCEPTED_COMMANDS = 64
MAX_PRESENTATION_EVENTS = 256
MAX_UINT64 = 2**64 - 1
ASSET_MANIFEST_HASH = CURATED_PACK.manifest_sha256

# Arena-style priority stops. Stop keys are the human-facing step names; they
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
    **{engine: stop for stop, engine in STOP_STEP_TO_ENGINE_STEP.items()},
    "COMBAT_FIRST_STRIKE_DAMAGE": "combat_damage",
}
STOP_SIDES = ("my", "opponent")
# Combat steps whose stops only hold while at least one creature is attacking,
# matching Arena and CR 508.8 without altering engine step progression.
ACTIVE_COMBAT_STOP_STEPS = frozenset({"declare_attackers", "declare_blockers"})
# Arena-style defaults: act in your own main phases, pause in active combat
# steps when you have a non-mana play, hold instants at the opponent's end
# step, and stop when the stack is non-empty.
DEFAULT_STOPS: dict[str, list[str]] = {
    "my": ["main1", "declare_attackers", "declare_blockers", "main2"],
    "opponent": ["declare_attackers", "declare_blockers", "end_step"],
}
app = FastAPI(title="Academy")

# Capture loaded runtime identity at process startup, before later source edits
# can be mistaken for the code that actually served a retained game.
PLAY_SOURCE_SHA256 = hashlib.sha256(
    b"".join(
        f"{path.parent.name}/{path.name}".encode() + b"\0" + path.read_bytes()
        for path in sorted(
            [
                *Path(__file__).parent.glob("*.py"),
                Path(managym.__file__).with_name("experience.py"),
                Path(managym.__file__).with_name("experience_projection.py"),
            ]
        )
    )
).hexdigest()
RULES_BINARY_SHA256 = hashlib.sha256(
    Path(managym._managym.__file__).read_bytes()
).hexdigest()
SESSION_TTL = timedelta(minutes=15)
SESSION_EXPIRED_END_REASON = "session_expired"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LiveGame:
    """A game held in memory: its authority, the one browser connection
    playing it, and the lock that takes its requests one at a time.

    Dropping it from memory ends nothing for a played game: the record is
    saved, and the game is rebuilt from it when it is next named."""

    game: "GameSession"
    websocket: WebSocket | None = None
    expires_at: datetime = field(default_factory=lambda: _now_utc() + SESSION_TTL)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The last reply, for a connection that arrives while a request runs.
    published: dict[str, Any] | None = None

    def touch(self) -> None:
        self.expires_at = _now_utc() + SESSION_TTL


@dataclass(frozen=True)
class Launch:
    """A game or Watch setup configured before the browser opened."""

    mode: Literal["play", "watch"]
    config: dict[str, Any]


# Live games by game id (the attempt id, which is also the trace id).
LIVE_GAMES: dict[str, LiveGame] = {}
# Private, process-local configurations. Never included in viewer frames.
LAUNCHES: dict[str, Launch] = {}
# The game each Play launch started: a launch starts one game however often
# its address is opened.
LAUNCH_GAMES: dict[str, str] = {}


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


@dataclass(frozen=True)
class PendingCanonicalDecision:
    """Prompt-level canonical identity and retained root before selection."""

    address: DecisionAddressV2
    frame: ExperienceFrame
    root: managym.Env


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
    semantic_receipt: TransitionReceipt | None
    state_before: str
    state_after: str
    semantic_events: list[dict[str, Any]]
    presentation_events: list[dict[str, Any]]
    viewer_presentation_events: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    encountered_definition_ids: list[int]


class StudyAttemptNotFoundError(ValueError):
    """No live Study attempt exists under this authority session."""


class StudyCommandUnavailableError(ValueError):
    """A canonical replay offer cannot lower through the retained Study fork."""


@dataclass
class StudyAttempt:
    attempt_id: str
    trace_id: str
    address: str
    projection: CanonicalReplayProjectionV1
    restored: RestoredReplayDecision
    branch: StudyBranch
    retry_command: Command
    retry_projection: dict[str, Any]
    retry_presentation: list[dict[str, Any]]
    evidence: JoinedStudyEvidence | None = None


def _definition_ids_by_object(obs: managym.Observation) -> dict[int, int]:
    """Resolve live card, permanent, and stack identities to typed definitions."""

    result = {
        observation_object_id(card): observation_definition_id(card)
        for card in object_records(obs)
    }
    for permanent in (*obs.agent_permanents, *obs.opponent_permanents):
        if permanent.object_id is not None:
            result[int(permanent.id)] = int(permanent.registry_key)
    for stack_object in obs.stack_objects:
        result[int(stack_object.stack_object_id)] = int(
            stack_object.source_card_registry_key
        )
    return result


def _semantic_event_payload(
    event: managym.EventData,
    definition_ids: dict[int, int],
) -> dict[str, Any]:
    related_definitions: set[int] = set()
    for kind, identity in (
        (event.source_kind, int(event.source_id)),
        (event.target_kind, int(event.target_id)),
    ):
        if kind == managym.EventEntityKindEnum.DEFINITION:
            related_definitions.add(identity)
        elif (
            kind
            in (
                managym.EventEntityKindEnum.CARD,
                managym.EventEntityKindEnum.PERMANENT,
                managym.EventEntityKindEnum.OBJECT,
            )
            and identity in definition_ids
        ):
            related_definitions.add(definition_ids[identity])
    return {
        "event_type": enum_name(EventTypeEnum, event.event_type),
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
        "definition_ids": sorted(related_definitions),
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
        for permanent in (
            *observation.agent_permanents,
            *observation.opponent_permanents,
        ):
            if permanent.object_id is not None:
                target[int(permanent.id)] = (
                    int(permanent.registry_key),
                    bool(permanent.tapped),
                )
    for identity, (definition_id, was_tapped) in before_taps.items():
        current = after_taps.get(identity)
        if current is not None and not was_tapped and current[1]:
            definitions.add(definition_id)
    return sorted(definitions)


def viewer_view(
    obs: managym.Observation,
    viewer: int,
    *,
    winner: int | None,
    include_details: bool = False,
) -> dict[str, Any]:
    """Serialize an observation for one acting viewer.

    Two guarantees, regardless of whose perspective the engine observation is
    from:
      1. ``agent`` is always ``viewer`` and ``opponent`` the other player.
      2. The opponent's hand is redacted (libraries are never serialized).

    ``winner`` is the game's winning seat, ``None`` while it is running or
    after a draw: the other seat's ``won: false`` does not say whether the
    viewer won.
    """
    data = serialize_observation(obs, include_details=include_details)
    if int(data["agent"]["player_index"]) != viewer:
        data["agent"], data["opponent"] = data["opponent"], data["agent"]
        data["won"] = bool(obs.game_over) and winner == viewer

    redact_observation(data)
    return data


def _is_hero_turn(obs: managym.Observation, human_seat: int = 0) -> bool:
    return int(obs.agent.player_index) == human_seat


def _is_priority_space(obs: managym.Observation) -> bool:
    return obs.action_space.action_space_type == managym.ActionSpaceEnum.PRIORITY


def _has_non_mana_play(obs: managym.Observation) -> bool:
    """Whether the priority window offers a legal action beyond pass and mana abilities."""
    return any(
        action.action_type != managym.ActionEnum.PRIORITY_PASS_PRIORITY
        and not action.is_mana_ability
        for action in obs.action_space.actions
    )


def _has_attackers(obs: managym.Observation) -> bool:
    """Whether at least one permanent on either battlefield is currently attacking."""
    return any(permanent.attacking for permanent in obs.agent_permanents) or any(
        permanent.attacking for permanent in obs.opponent_permanents
    )


def _hero_is_active_player(obs: managym.Observation, human_seat: int = 0) -> bool:
    """Whether the hero is the active player (it is 'my' turn).

    Perspective-independent: finds the hero side by player_index rather than
    assuming ``obs.agent`` is the hero.
    """
    hero = obs.agent if int(obs.agent.player_index) == human_seat else obs.opponent
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


def _select_opponent(sha256: Any) -> Opponent:
    """Resolve the requested catalog checkpoint by exact digest, verified.

    Refusals carry the mismatch that makes the checkpoint unplayable here.
    """

    if not isinstance(sha256, str):
        raise ValueError("opponent_sha256 must name a registered policy identity.")
    opponent = catalog_opponent(sha256)
    if opponent is None:
        raise ValueError("The selected opponent is no longer available.")
    return opponent


def _bundled_deck_file(name: str, seat: str) -> dict[str, str]:
    deck, resolved_name = _normalize_deck(
        name, DEFAULT_HERO_DECK_NAME if seat == "hero" else DEFAULT_VILLAIN_DECK_NAME
    )
    pack = next(
        (
            candidate
            for candidate in CURATED_PACK_CATALOG
            if resolved_name in candidate.deck_ids
        ),
        None,
    )
    sideboard = (
        {} if pack is None else dict(pack.player_config(resolved_name, seat).sideboard)
    )
    return {
        "source": deck_to_source(managym.Deck(deck, sideboard)),
        "filename": f"{resolved_name}.txt",
        "name": DECK_DISPLAY_NAMES.get(resolved_name, resolved_name),
    }


def resolve_pool_decks(data: dict[str, Any], *, seed: int | None) -> dict[str, Any]:
    """Expand ``pool:<id>`` selections and one-sided ``deck_import`` payloads."""
    raw_import = data.get("deck_import")
    raw_hero = data.get("hero_deck", DEFAULT_HERO_DECK_NAME)
    raw_villain = data.get("villain_deck", DEFAULT_VILLAIN_DECK_NAME)

    if (is_pool_key(raw_hero) or is_pool_key(raw_villain)) and not deck_pools():
        defaults = ensure_default_deck_pools()
        if defaults:
            set_deck_pools(defaults)

    if isinstance(raw_import, dict):
        has_hero = raw_import.get("hero") is not None
        has_villain = raw_import.get("villain") is not None
        if has_hero and has_villain:
            return data
        if has_hero or has_villain:
            import random as _random

            rng = (
                _random.Random(seed)
                if isinstance(seed, int)
                else _random.SystemRandom()
            )
            hero_payload = raw_import.get("hero")
            villain_payload = raw_import.get("villain")
            hero_pool = resolve_deck_pool(raw_hero) if is_pool_key(raw_hero) else None
            villain_pool = (
                resolve_deck_pool(raw_villain) if is_pool_key(raw_villain) else None
            )
            if hero_payload is None:
                if hero_pool is not None:
                    sampled = hero_pool.sample(rng)
                    hero_payload = deck_file_payload(sampled.deck, name=sampled.name)
                elif is_pool_key(raw_hero):
                    raise ValueError(f"Unknown deck pool {raw_hero!r}.")
                else:
                    hero_payload = _bundled_deck_file(str(raw_hero), "hero")
            if villain_payload is None:
                if villain_pool is not None:
                    sampled = villain_pool.sample(rng)
                    villain_payload = deck_file_payload(sampled.deck, name=sampled.name)
                elif is_pool_key(raw_villain):
                    raise ValueError(f"Unknown deck pool {raw_villain!r}.")
                else:
                    villain_payload = _bundled_deck_file(str(raw_villain), "villain")
            resolved_partial = dict(data)
            resolved_partial["deck_import"] = {
                "hero": hero_payload,
                "villain": villain_payload,
            }
            if hero_pool is not None:
                resolved_partial["hero_deck_pool"] = hero_pool.key
            if villain_pool is not None:
                resolved_partial["villain_deck_pool"] = villain_pool.key
            return resolved_partial
        return data

    hero_pool = resolve_deck_pool(raw_hero) if is_pool_key(raw_hero) else None
    villain_pool = resolve_deck_pool(raw_villain) if is_pool_key(raw_villain) else None
    if is_pool_key(raw_hero) and hero_pool is None:
        raise ValueError(f"Unknown deck pool {raw_hero!r}.")
    if is_pool_key(raw_villain) and villain_pool is None:
        raise ValueError(f"Unknown deck pool {raw_villain!r}.")
    if hero_pool is None and villain_pool is None:
        return data

    import random as _random

    rng: _random.Random = (
        _random.Random(seed ^ 0x517CC1B727220A95)
        if seed is not None
        else _random.SystemRandom()
    )
    if (
        hero_pool is not None
        and villain_pool is not None
        and hero_pool.id == villain_pool.id
    ):
        hero_named, villain_named = hero_pool.sample_pair(rng)
        hero_payload = deck_file_payload(hero_named.deck, name=hero_named.name)
        villain_payload = deck_file_payload(villain_named.deck, name=villain_named.name)
    else:
        if hero_pool is not None:
            hero_named = hero_pool.sample(rng)
            hero_payload = deck_file_payload(hero_named.deck, name=hero_named.name)
        else:
            hero_payload = _bundled_deck_file(str(raw_hero), "hero")
        if villain_pool is not None:
            villain_named = villain_pool.sample(rng)
            villain_payload = deck_file_payload(
                villain_named.deck, name=villain_named.name
            )
        else:
            villain_payload = _bundled_deck_file(str(raw_villain), "villain")

    return {
        **data,
        "deck_import": {"hero": hero_payload, "villain": villain_payload},
        "hero_deck_pool": hero_pool.key if hero_pool is not None else None,
        "villain_deck_pool": villain_pool.key if villain_pool is not None else None,
    }


def _parse_game_config(config: Any) -> GameConfig:
    data = config or {}
    if not isinstance(data, dict):
        raise ValueError("new_game.config must be an object.")

    villain_type = data.get("villain_type", "search")
    if villain_type == "checkpoint":
        warnings.warn(
            "checkpoint player names are deprecated; use registered policies (removed in 0.3.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        villain_type = "policy"
    if villain_type not in VILLAIN_TYPES:
        raise ValueError(
            "villain_type must be one of: " + ", ".join(sorted(VILLAIN_TYPES)) + "."
        )

    seed_value = data.get("seed")
    if seed_value is None:
        seed: int | None = None
    else:
        if type(seed_value) is not int and not (
            isinstance(seed_value, str)
            and seed_value.isascii()
            and seed_value.isdecimal()
        ):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        try:
            seed = int(seed_value)
        except Exception as exc:
            raise ValueError("seed must be an integer.") from exc
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer.")

    decklists = data.get("decklists", "open")
    if decklists not in ("open", "closed"):
        raise ValueError("decklists must be 'open' or 'closed'.")
    if decklists == "closed" and villain_type == "search":
        raise ValueError(
            "The search opponent samples hidden cards from open decklists; "
            "choose open decklists or another opponent."
        )

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

    # Private import provenance (see `GameConfig.hero_source`); empty for named decks.
    data = resolve_pool_decks(data, seed=seed)
    hero_deck_pool = data.get("hero_deck_pool")
    villain_deck_pool = data.get("villain_deck_pool")
    if hero_deck_pool is not None and not isinstance(hero_deck_pool, str):
        raise ValueError("hero_deck_pool must be a string or null.")
    if villain_deck_pool is not None and not isinstance(villain_deck_pool, str):
        raise ValueError("villain_deck_pool must be a string or null.")
    provenance: dict[str, Any] = {}
    if data.get("deck_import") is not None:
        deck_import = DeckImportInput.model_validate(data["deck_import"])
        hero_import, villain_import = deck_import.admit()
        provenance = {
            "hero_source": hero_import.decklist.to_dict(),
            "villain_source": villain_import.decklist.to_dict(),
        }
        hero_deck, villain_deck = hero_import.deck.main, villain_import.deck.main
        hero_deck_name = deck_import.hero.display_name()
        villain_deck_name = deck_import.villain.display_name()
        selected_pack = None
        sideboards = {
            "hero": hero_import.deck.sideboard,
            "villain": villain_import.deck.sideboard,
        }
    else:
        hero_deck, hero_deck_name = _normalize_deck(
            data.get("hero_deck"), DEFAULT_HERO_DECK_NAME
        )
        villain_deck, villain_deck_name = _normalize_deck(
            data.get("villain_deck"), DEFAULT_VILLAIN_DECK_NAME
        )

        selected_pack = curated_pack_for_matchup(hero_deck_name, villain_deck_name)
        if selected_pack is None and any(
            name in ELEMENTAL_WORLD_PACK.deck_ids
            for name in (hero_deck_name, villain_deck_name)
        ):
            raise ValueError(
                "Elemental decks require two setups from the Elemental roster"
            )

        sideboards = {}
        for seat, deck_name in (
            ("hero", hero_deck_name),
            ("villain", villain_deck_name),
        ):
            pack = selected_pack
            if pack is None:
                pack = next(
                    (
                        candidate
                        for candidate in CURATED_PACK_CATALOG
                        if deck_name in candidate.deck_ids
                    ),
                    None,
                )
            sideboard = (
                {}
                if pack is None
                else dict(pack.player_config(deck_name, seat).sideboard)
            )
            # Named setup cannot silently change when a persisted config is reused.
            supplied = data.get(f"{seat}_sideboard", sideboard)
            if (
                not isinstance(supplied, dict)
                or supplied != sideboard
                or any(type(count) is not int for count in supplied.values())
            ):
                raise ValueError(f"{seat}_sideboard must match the named deck setup")
            sideboards[seat] = sideboard

    opponent: Opponent | None = None
    if villain_type == "policy":
        opponent = _select_opponent(data.get("opponent_sha256"))

    mirror = TypeAdapter(SearchConfig | None).validate_python(data.get("mirror"))
    if mirror is not None:
        if opponent is not None:
            reason = descriptor_search_unavailable(opponent.descriptor)
            if reason:
                raise ValueError(reason)
        else:
            raise ValueError("Search requires a registered search-capable policy.")
    parsed = GameConfig(
        **provenance,
        hero_deck=hero_deck,
        villain_deck=villain_deck,
        hero_sideboard=sideboards["hero"],
        villain_sideboard=sideboards["villain"],
        hero_deck_name=hero_deck_name,
        villain_deck_name=villain_deck_name,
        hero_deck_pool=hero_deck_pool,
        villain_deck_pool=villain_deck_pool,
        asset_pack=selected_pack.reference if selected_pack is not None else None,
        villain_type=villain_type,
        human_seat=data.get("human_seat", 0),
        mana_floats=data.get("mana_floats", "off"),
        mana_payment=data.get("mana_payment", "manual"),
        seed=seed,
        seed_was_requested=seed_value is not None,
        villain_sims=villain_sims,
        opponent=opponent.identity(mirror) if opponent is not None else None,
        opponent_checkpoint=opponent.record(mirror) if opponent is not None else None,
        stops=_parse_stops(data.get("stops")),
        stop_on_stack=_parse_flag(data, "stop_on_stack", True),
        auto_pass=_parse_flag(data, "auto_pass", True),
        decklists=decklists,
        mirror=mirror,
    )
    if opponent is not None:
        policy_registry().admit(opponent.descriptor.id, parsed.to_rust())
    return parsed


def observe_policies(
    policies: Iterable[BotPolicy | None], env: managym.Env, obs: managym.Observation
) -> None:
    """Show each registered policy its own seat's view of the current state."""
    for policy in policies:
        if isinstance(policy, PolicyRunner):
            if int(obs.agent.player_index) == policy.seat:
                view_obs = obs
            else:
                view_obs = env.observation_for_player(policy.seat)
                view_obs.recent_events = list(obs.recent_events)
            policy.observe(view_obs)


def close_policies(policies: Iterable[BotPolicy | None]) -> None:
    with ExitStack() as resources:
        for policy in policies:
            if isinstance(policy, PolicyRunner):
                resources.callback(policy.close)


class RecordDiverged(Exception):
    """The engine no longer plays a stored game as recorded, at `step`."""

    def __init__(self, step: int) -> None:
        super().__init__(f"Step {step + 1} no longer plays as recorded.")
        self.step = step


def replay_stored_steps(
    env: managym.Env,
    obs: managym.Observation,
    events: list[dict[str, Any]],
    human_seat: int,
    policy: BotPolicy | None,
    projectors: tuple[PresentationProjector, PresentationProjector],
) -> tuple[managym.Observation, dict[int, managym.Env]]:
    """Step `env` (in place) through a played game's stored steps, recording nothing.

    Each step must be the stored one (`step_matches`: the same seat acts and
    the stored action has its stored description), else RecordDiverged. The
    opponent is shown every position and, where it keeps memory between
    decisions, asked at each of its own, so it holds the history it had.
    `projectors` consume each step's presentation facts and are left at each
    viewer's next address, with the ancestry a later step's events refer to.
    Returns the observation after the last step and a Study root per
    deliberate decision, by decision ordinal: the position before it.
    """
    roots: dict[int, managym.Env] = {}
    for index, event in enumerate(events):
        if not step_matches(obs, event, human_seat):
            raise RecordDiverged(index)
        try:
            if not event.get("auto"):
                if (
                    event["actor"] == "villain"
                    and isinstance(policy, PolicyRunner)
                    and policy.base_descriptor.recurrent
                ):
                    choose_bot(policy, env, obs)
                roots[len(roots)] = env.clone_env()
            obs = env.step(int(event["action"]))[0]
        except (ValueError, managym.AgentError) as exc:
            raise RecordDiverged(index) from exc
        for viewer, projector in enumerate(projectors):
            projector.project(
                env.presentation_facts_json(viewer),
                from_revision=index,
                to_revision=index + 1,
                caused_by=None,
            )
        env.take_presentation_positions()
        observe_policies((policy,), env, obs)
    return obs, roots


class PendingBotDecision(TypedDict):
    code: str
    message: str
    match_id: str
    expected_revision: int


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
        watch_mode: bool = False,
    ) -> None:
        self.trace_dir = trace_dir or trace_store.TRACES_DIR
        self.watch_mode = watch_mode
        self.work = Work()
        self.human_seat = 0
        self.watch_policies: tuple[BotPolicy, BotPolicy] | None = None
        self._bot_decision: BotDecision | None = None
        # A policy reading the person's decisions, while they show it; off at
        # the start of every game (`show_analysis`).
        self._live_analysis: LiveAnalysis | None = None
        self.mirror_retry: PendingBotDecision | None = None
        self._id_factory = id_factory or (lambda _kind: secrets.token_urlsafe(16))
        self._clock = clock or trace_store.utc_now_iso
        self._preparing_demo = False
        self.attempt_store = AttemptStore(self.trace_dir / "play.sqlite")
        self.record_players: list[RecordedPlayer] | None = None
        self._villain_offer_policy = villain_offer_policy
        self._capture_authority_evidence = capture_authority_evidence
        self._historical_evidence_provider = (
            historical_evidence_provider or UnavailableHistoricalStudyEvidenceProvider()
        )
        self._allow_fixture_study_evidence = allow_fixture_study_evidence
        self.env: managym.Env | None = None
        self.obs: managym.Observation | None = None
        self.villain_policy: BotPolicy | None = None
        self.trace: Trace | None = None
        self.trace_id: str | None = None
        self._demo_name: str | None = None
        # Display names for the current matchup, echoed on every payload.
        self.deck_names: dict[str, str] | None = None
        self.asset_pack: dict[str, str] | None = None
        self.decklists_closed = False
        self.content_hash = UNVERSIONED_CONTENT_HASH
        self.asset_manifest_hash = ASSET_MANIFEST_HASH
        self._trace_saved = False
        # The (match, revision, events) last written while active: projecting
        # the same frame again (recovery caches, the committed boundary, the
        # settled reply) does not rewrite the whole trace.
        self._persisted: tuple[str, int, int] | None = None
        # Each trace event's, decision's and presentation event's JSON, so a
        # save encodes only what the last step added (`append_json`).
        self._encoded_events: list[tuple[TraceEvent, str]] = []
        self._encoded_decisions: list[tuple[ReplayDecision, str]] = []
        self._encoded_presentation: dict[int, list[tuple[dict[str, Any], str]]] = {
            0: [],
            1: [],
        }
        # The person's board after each step since the last surfaced message (`step_boards`).
        self._step_boards: list[StepBoard] = []
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
        self.presentation = (PresentationProjector(), PresentationProjector())
        self.presentation_events: list[dict[str, Any]] = []
        self._last_presentation_cursor = 0
        # Durable replay truth is unbounded by the live reconnect ledger.
        self.canonical_decisions: list[ReplayDecision] = []
        self.canonical_presentation: dict[int, list[dict[str, Any]]] = {
            0: [],
            1: [],
        }
        self._authority_command_seq = 0
        self._study_roots: dict[int, managym.Env] = {}
        self._pending_decision: PendingCanonicalDecision | None = None
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
        """Deal the game `raw_config` asks for and play it to its first
        decision. A failure leaves nothing started."""
        from .play_demo import load_play_demo, replay_checked_prefix

        demo_name = (raw_config if isinstance(raw_config, dict) else {}).get("demo")
        demo = None
        if demo_name is not None:
            demo = load_play_demo(demo_name)
            raw_config = {**raw_config, **demo.config}
        played_from: PlayedFrom | None = None
        source_events: list[dict[str, Any]] = []
        if isinstance(raw_config, dict) and raw_config.get("played_from") is not None:
            played_from, source = self._played_from_source(raw_config["played_from"])
            source_events = source["events"]
            # The source's own setup and seed.
            raw_config = {
                **setup_draft(GameConfig(**source["config"])),
                "seed": str(source["config"]["seed"]),
            }

        config = _parse_game_config(raw_config)
        seed = config.seed if config.seed is not None else secrets.randbits(32)
        config.seed = seed
        env, obs, policy = self._open_game(config)
        # A watched game names both of its bots before it starts.
        if not (self.watch_mode and self.record_players is not None):
            self.record_players = AttemptStore.players_for_game(config.to_record())
        self._begin(
            config,
            env,
            obs,
            policy,
            match_id=self._id_factory("match"),
            timestamp=self._clock(),
            played_from=played_from,
        )
        self._demo_name = demo_name
        self._persist_attempt()
        # A checked demo prefix is setup, not a sequence of surfaced human
        # decisions. Retain every Command and Study root in memory, then persist
        # its complete final snapshot once instead of rewriting the growing
        # replay at each internal setup step.
        self._preparing_demo = demo is not None or played_from is not None
        try:
            if played_from is not None:
                self._play_recorded_prefix(source_events, played_from.event)
            else:
                self._advance()
            self._last_presentation_cursor = self.presentation[self.human_seat].next_seq
            update = self._wire_message(reason="initial_connect")
            if demo is not None:
                replay_checked_prefix(self, update, demo, demo_name)
        finally:
            self._preparing_demo = False
        if demo is not None or played_from is not None:
            update = self._wire_message(reason="initial_connect")
        return update

    def _open_game(
        self, config: GameConfig
    ) -> tuple[managym.Env, managym.Observation, BotPolicy | None]:
        """Deal `config`'s game and load its opponent, leaving this session's
        current game untouched. Raises ValueError naming what could not load."""
        self.work.stage("load")
        try:
            policy = (
                None
                if self.watch_mode
                else build_villain_policy(config, seat=1 - config.human_seat)
            )
        except SearchError as exc:
            raise ValueError(f"{exc.reason}: {exc}") from exc
        except ValueError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to load opponent policy")
            raise ValueError(
                f"Failed to load the selected opponent ({type(exc).__name__}); see the server log."
            ) from exc

        with ExitStack() as prepared:
            if isinstance(policy, PolicyRunner):
                prepared.callback(policy.close)
            try:
                # Native skipping resolves sole legal actions, as in every runner;
                # priority-stop automation still records its Commands here.
                # Bot policies may read the viewer's decision memory, which the
                # engine only captures when asked.
                env = managym.Env(seed=config.seed, decision_memory=True)
                obs, _ = env.reset(config.to_rust())
                env.enable_presentation()
            except BaseException as exc:
                # PyO3's PanicException derives directly from BaseException and has
                # no importable Python class. A failed disposable replacement must
                # not close the existing authority; process interrupts still escape.
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                logging.getLogger(__name__).exception(
                    "Failed to prepare game environment"
                )
                raise ValueError(
                    "Could not prepare the requested game. The current game is unchanged."
                ) from exc
            try:
                observe_policies((policy,), env, obs)
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Bot observation failed during game setup"
                )
                raise ValueError(
                    f"Bot observation failed ({type(exc).__name__}); see the server log."
                ) from exc
            prepared.pop_all()
        return env, obs, policy

    def _begin(
        self,
        config: GameConfig,
        env: managym.Env,
        obs: managym.Observation,
        policy: BotPolicy | None,
        *,
        match_id: str,
        timestamp: str,
        played_from: PlayedFrom | None,
    ) -> None:
        """Make an opened game this session's game, at its deal with nothing
        recorded. The game it replaces is released, not ended."""
        self.release()
        self.env, self.obs, self.villain_policy = env, obs, policy
        self._demo_name = None
        self.human_seat = config.human_seat
        self.decklists_closed = config.decklists == "closed"
        selected_pack = curated_pack_for_matchup(
            config.hero_deck_name,
            config.villain_deck_name,
        )
        # A pack identifies both decks, so closed games use the generic identity.
        # Cards still resolve their treatments by name from their public zones.
        if selected_pack is not None and not self.decklists_closed:
            self.asset_manifest_hash = selected_pack.manifest_sha256
            self.content_hash = (
                UNVERSIONED_CONTENT_HASH
                if selected_pack is CURATED_PACK
                else str(env.content_pack_manifest()["content_digest"])
            )
        else:
            self.content_hash = UNVERSIONED_CONTENT_HASH
            self.asset_manifest_hash = ASSET_MANIFEST_HASH

        self.deck_names = {
            "hero": DECK_DISPLAY_NAMES.get(
                config.hero_deck_name,
                "Custom"
                if config.hero_deck_name == "custom"
                else config.hero_deck_name,
            ),
            "villain": DECK_DISPLAY_NAMES.get(
                config.villain_deck_name,
                "Custom"
                if config.villain_deck_name == "custom"
                else config.villain_deck_name,
            ),
        }
        self.asset_pack = (
            dict(config.asset_pack)
            if config.asset_pack and not self.decklists_closed
            else None
        )
        self.trace = Trace(
            config=config,
            events=[],
            final_observation={},
            winner=None,
            end_reason="active",
            timestamp=timestamp,
            played_from=played_from,
        )
        self._trace_saved = False
        self._persisted = None
        self._encoded_events = []
        self._encoded_decisions = []
        self._encoded_presentation = {0: [], 1: []}
        self._step_boards = []
        self.stops = {
            side: list(steps) for side, steps in (config.stops or DEFAULT_STOPS).items()
        }
        self.stop_on_stack = config.stop_on_stack
        self.auto_pass = config.auto_pass
        self._f6_turn = None
        self._auto_passed_since_surface = 0
        self.match_id = self.trace_id = match_id
        self.revision = 0
        self._next_prompt_id = 1
        self.published_prompt = None
        self.accepted_commands = {}
        self.presentation = (PresentationProjector(), PresentationProjector())
        self.presentation_events = []
        self._last_presentation_cursor = 0
        self.canonical_decisions = []
        self.canonical_presentation = {0: [], 1: []}
        self._authority_command_seq = 0
        self._study_roots = {}
        self._pending_decision = None
        self._study_provider = None
        self.authority_transitions = []
        self.authority_fallback_counters = {
            "legacy_fixed_action": 0,
            "card_name_dispatch": 0,
            "candidate_cap": 0,
            "client_legality": 0,
        }
        self._study_attempt_seq = 0

    def resume(self, game: str) -> dict[str, Any]:
        """Take up an unfinished played game from its stored record.

        The engine replays the stored steps (`replay_stored_steps`) and the
        record is installed as it is stored, never regenerated: the game keeps
        its id, its events, its decision addresses and its presentation track,
        and what is played next is appended to them. Then the game advances,
        so a game stopped during the opponent's turn gets its reply now.

        Raises ValueError and changes nothing where the game is not an
        unfinished played game (`game_not_found`, `game_is_watch`,
        `game_finished`, `game_stopped`) or its opponent cannot be loaded, so
        it can be resumed once that is put right. Where the engine no longer
        plays the stored steps, the game ends as `record_diverged` and stays
        replayable.
        """
        store = self.attempt_store
        metadata = store.metadata(game)
        if not metadata:
            raise ValueError("game_not_found: That game is not saved here.")
        if not any(player["kind"] == "human" for player in metadata["players"]):
            raise ValueError("game_is_watch: A watched game has no seat of yours.")
        if metadata["status"] == "completed":
            raise ValueError("game_finished: That game is over.")
        retained = store.load_attempt(game) if metadata["resumable"] else None
        if retained is None:
            raise ValueError("game_stopped: That game stopped and cannot be resumed.")
        source = retained.trace
        replay = source.get("canonical_replay")
        config = GameConfig(**source["config"])
        if config.watch_players is not None or config.seed is None or replay is None:
            raise ValueError("resume_unavailable: That game has no record to resume.")
        env, obs, policy = self._open_game(config)
        tracks = {
            int(track["viewer"]): track["events"]
            for track in replay["presentation_tracks"]
        }
        projectors = (PresentationProjector(), PresentationProjector())
        try:
            obs, roots = replay_stored_steps(
                env, obs, source["events"], config.human_seat, policy, projectors
            )
            if len(roots) != len(replay["decisions"]) or any(
                projector.next_seq != len(tracks.get(viewer, []))
                for viewer, projector in enumerate(projectors)
            ):
                raise RecordDiverged(len(source["events"]))
        except RecordDiverged as exc:
            close_policies((policy,))
            store.diverge(game)
            raise ValueError(
                f"record_diverged: This game no longer plays as recorded at step {exc.step + 1}."
            ) from exc
        except BaseException:
            close_policies((policy,))
            raise

        played_from = source.get("played_from")
        self._begin(
            config,
            env,
            obs,
            policy,
            match_id=game,
            timestamp=source["timestamp"],
            played_from=None if played_from is None else PlayedFrom(**played_from),
        )
        # Each stored item keeps its stored JSON (`append_json`), so a save
        # writes the stored prefix byte for byte and encodes only what follows.
        events = [trace_store.stored_event(record) for record in source["events"]]
        self.trace.events = events
        self._encoded_events = [
            (event, json.dumps(record))
            for event, record in zip(events, source["events"], strict=True)
        ]
        self.canonical_decisions = [
            ReplayDecision.model_validate(row) for row in replay["decisions"]
        ]
        self._encoded_decisions = [
            (row, json.dumps(record))
            for row, record in zip(
                self.canonical_decisions, replay["decisions"], strict=True
            )
        ]
        for viewer in (0, 1):
            track = tracks.get(viewer, [])
            self.canonical_presentation[viewer] = track
            self._encoded_presentation[viewer] = [
                (event, json.dumps(event)) for event in track
            ]
        self.presentation = projectors
        self.presentation_events = list(
            self.canonical_presentation[self.human_seat][-MAX_PRESENTATION_EVENTS:]
        )
        self._study_roots = roots
        # Taking a game up is a new revision, stored with the reopening, so a
        # Command held from before it is stale wherever the game was rebuilt.
        self.revision = store.reopen(game)
        # Above every id the record holds, so nothing recorded is named twice.
        self._next_prompt_id = 1 + max(
            (row.prompt_id for row in self.canonical_decisions), default=0
        )
        self._authority_command_seq = len(events)
        self.record_players = [
            RecordedPlayer(**player) for player in metadata["players"]
        ]
        if source["end_reason"] == "active":
            self._persisted = (game, self.revision, len(events))
        self._advance()
        return self.current_message()

    def _played_from_source(self, raw: Any) -> tuple[PlayedFrom, dict[str, Any]]:
        """The game to play on from and its stored trace: any saved played
        game, finished or not, at one of the person's deliberate decisions."""
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("trace_id"), str)
            or type(raw.get("event")) is not int
        ):
            raise ValueError("played_from needs a game and a decision.")
        trace_id, event = raw["trace_id"], raw["event"]
        source = self.attempt_store.load(trace_id)
        if source is None:
            raise ValueError("played_from_unavailable: That game is not saved here.")
        if source["config"].get("watch_players") is not None:
            raise ValueError(
                "played_from_unavailable: A watched game has no seat of yours."
            )
        events = source["events"]
        if (
            not 0 <= event < len(events)
            or events[event]["actor"] != "hero"
            or events[event].get("auto")
        ):
            raise ValueError(
                "played_from_decision: That step is not one of your decisions."
            )
        return PlayedFrom(trace_id, event), source

    def _play_recorded_prefix(self, events: list[dict[str, Any]], upto: int) -> None:
        """Play a recorded game's first `upto` steps through the Command path.

        A game played on is a new game with its own record, so its opening
        steps are recorded again under its own id; `resume` continues the
        same game and only steps the engine (`replay_stored_steps`).

        The opponent's recorded choices go through the offer-policy hook (a
        recurrent opponent is still asked at each, so it has the history it
        would have had) and the person's through `hero_command`. Every window
        surfaces meanwhile, so the steps automation took replay as recorded,
        now as Commands. The game is left at decision `upto`, the person's.
        Raises ValueError where the game no longer plays as recorded.
        """
        assert self.trace is not None

        def recorded(actor: str, context: DecisionContext) -> int:
            index = len(self.trace.events)
            event = events[index] if index < upto else None
            offers = [
                offer
                for offer, engine in context.action_by_offer.items()
                if event is not None and engine == int(event["action"])
            ]
            if (
                event is None
                or event["actor"] != actor
                or len(offers) != 1
                or not step_matches(self.obs, event, self.human_seat)
            ):
                raise ValueError(
                    f"played_from_diverged: Step {index + 1} no longer plays as recorded."
                )
            return offers[0]

        def opponent(context: DecisionContext) -> int:
            policy = self.villain_policy
            if isinstance(policy, PolicyRunner) and policy.base_descriptor.recurrent:
                choose_bot(policy, self.env, self.obs)
            return recorded("villain", context)

        saved = self._villain_offer_policy, self.auto_pass
        self._villain_offer_policy, self.auto_pass = opponent, False
        try:
            self._advance()
            while len(self.trace.events) < upto:
                context = self._publish_current_prompt()
                if context is None:
                    raise ValueError("played_from_diverged: The game ended early.")
                result = self.hero_command(
                    {
                        "command_id": f"played-from.{len(self.trace.events)}",
                        "match_id": self.match_id,
                        "expected_revision": context.revision,
                        "prompt_id": context.prompt_id,
                        "offer_id": recorded("hero", context),
                        "answers": [],
                    }
                )
                if result["status"] != "accepted":
                    raise ValueError(
                        "played_from_diverged: A recorded move was refused."
                    )
        finally:
            self._villain_offer_policy, self.auto_pass = saved
        # The prompt published during the prefix carries its stops; publish it
        # again under the game's own. Nobody has seen it, so nothing goes stale.
        self.published_prompt = None
        if len(self.trace.events) != upto or self._publish_current_prompt() is None:
            raise ValueError("played_from_diverged: The decision was not reached.")

    def hero_action(self, raw_index: Any) -> dict[str, Any]:
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if self.obs.game_over:
            raise ValueError("Game is already over. Start a new game.")
        if not _is_hero_turn(self.obs, self.human_seat):
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
        batch_cursor = self.presentation[self.human_seat].next_seq
        self._apply_bound_command(context, command, source="client")
        self._advance()
        self._last_presentation_cursor = batch_cursor
        return self._wire_message()

    def hero_command(
        self, raw: Any, *, committed: Callable[[dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
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
        if command_id.startswith(("authority.", "compound-")):
            return self._reject_command(raw, "reserved_command_id", recover=False)

        # Check retries before any revision/match validation. A response may
        # have been lost after the original command committed.
        receipt = self.accepted_commands.get(command_id)
        if receipt is not None:
            return {
                "type": "command_outcome",
                "status": "duplicate",
                "receipt": deepcopy(receipt),
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
        if not _is_hero_turn(self.obs, self.human_seat):
            return self._reject_command(raw, "not_actor", recover=False)

        try:
            command = Command.model_validate(raw).model_dump(mode="json")
        except Exception:
            return self._reject_command(raw, "invalid_command", recover=False)

        base_revision = context.revision
        batch_cursor = self.presentation[self.human_seat].next_seq
        self._apply_bound_command(context, command, source="client")
        # The receipt names the human transition, before any opponent proposal.
        # A later failure must not make an accepted Command appear unexecuted.
        committed_frame = self._experience_frame()
        receipt = {
            "command_id": command_id,
            "actor": self.human_seat,
            "accepted_at": base_revision,
            "resulting_revision": self.revision,
            "resulting_frame_hash": committed_frame["frame_hash"],
        }
        self.accepted_commands[command_id] = receipt
        while len(self.accepted_commands) > MAX_ACCEPTED_COMMANDS:
            self.accepted_commands.pop(next(iter(self.accepted_commands)))

        if committed is not None:
            committed(self._accepted_outcome(base_revision, batch_cursor, receipt))
            base_revision = self.revision
            batch_cursor = self.presentation[self.human_seat].next_seq
        self._advance()
        self._last_presentation_cursor = batch_cursor
        return self._accepted_outcome(base_revision, batch_cursor, receipt)

    def _accepted_outcome(
        self, base_revision: int, cursor: int, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        """Publish events since the last boundary with the immutable receipt.

        The same envelope serves early commitment and the settled response;
        later frames never change the transition named by the receipt.
        """
        frame = self._experience_frame()
        return {
            "type": "command_outcome",
            "status": "accepted",
            "update": {
                "base_revision": base_revision,
                "frame": frame,
                "presentation": [
                    dict(event)
                    for event in self.presentation_events
                    if int(event["seq"]) >= cursor
                ],
                "receipt": deepcopy(receipt),
                **self._drain_surface_metadata(frame["projection"]),
            },
        }

    def retry_mirror(self, match_id: str, expected_revision: int) -> dict[str, Any]:
        """Retry only the bound, uncommitted bot decision after a deadline."""
        if (
            self.watch_mode
            or self.mirror_retry is None
            or self.match_id != match_id
            or self.revision != expected_revision
        ):
            raise ValueError("The pending search changed. Recover the current table.")
        self.mirror_retry = None
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

        # The record holds the stops last set, so the game stops in the same
        # windows when it is continued from its record.
        self.trace.config = replace(
            self.trace.config,
            stops={side: list(steps) for side, steps in self.stops.items()},
            stop_on_stack=self.stop_on_stack,
            auto_pass=self.auto_pass,
        )

        # Stop configuration is versioned authority state even though it is not
        # a player decision and therefore never receives a replay ordinal.
        self._advance_protocol_revision()
        batch_cursor = self.presentation[self.human_seat].next_seq
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

        if _is_hero_turn(self.obs, self.human_seat) and _is_priority_space(self.obs):
            batch_cursor = self.presentation[self.human_seat].next_seq
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
        native_command: str | None = None,
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
        actor = "hero" if context.viewer == self.human_seat else "villain"
        self._step_and_record(
            actor=actor,
            action_index=action_index,
            actions=context.actions,
            context=context,
            command=command.model_dump(mode="json"),
            source=source,
            native_command=native_command,
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
        native_command: str | None = None,
    ) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            raise RuntimeError("Cannot step without an active game.")

        observation = decision_surface_view(self.env, int(self.obs.agent.player_index))
        analysis_observation = None
        if self.watch_mode:
            analysis_observation = deepcopy(observation)
            other = serialize_observation(
                self.env.observation_for_player(1 - int(self.obs.agent.player_index))
            )
            analysis_observation["opponent"]["hand"] = other["agent"]["hand"]
            analysis_observation["opponent"]["hand_hidden_count"] = 0
        before_observation = self.obs
        state_before = (
            self.env.state_digest() if self._capture_authority_evidence else None
        )
        actor_index = self.human_seat if actor == "hero" else 1 - self.human_seat
        selected_offer: dict[str, Any] | None = None
        if auto:
            if context is None:
                context = self._build_decision_context(
                    self.obs,
                    viewer=actor_index,
                )
            matching_offer_ids = [
                offer_id
                for offer_id, engine_index in context.action_by_offer.items()
                if engine_index == action_index
            ]
            if len(matching_offer_ids) != 1:
                raise RuntimeError("Automatic action has no unique authority offer.")
            command = self._authority_command(
                context,
                matching_offer_ids[0],
                "auto",
            )
            source = "auto"
        else:
            if context is None or command is None or source not in {"client", "policy"}:
                raise RuntimeError("Deliberate engine actions require a bound command.")
            if context.viewer != actor_index:
                raise RuntimeError("Decision context actor differs from engine actor.")
        assert context is not None and command is not None
        offer_id = int(command["offer_id"])
        selected_offer = next(
            offer for offer in context.offers if int(offer["id"]) == offer_id
        )
        if not auto:
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
                    "native_command": native_command,
                }
            )
            retained_root = self.env.clone_env()
            if actor_index == self.human_seat:
                pending = self._pending_decision
                if pending is None:
                    raise RuntimeError(
                        "Hero command has no pending canonical decision."
                    )
                committed_address = DecisionAddressV2.from_row(
                    replay_id=f"replay.{self.match_id}",
                    match_id=self.match_id,
                    row=row,
                )
                if committed_address != pending.address:
                    raise RuntimeError(
                        "Committed decision identity drifted from live prompt."
                    )
                retained_root = pending.root.clone_env()
            self._study_roots[int(row.ordinal)] = retained_root
            self.canonical_decisions.append(row)
        action_description = actions[action_index]["description"]
        from_revision = self.revision
        semantic_frame = json.loads(self.env.semantic_decision_frame_json())
        if (
            int(semantic_frame["actor"]) != actor_index
            or len(semantic_frame["offers"]) != len(actions)
            or int(semantic_frame["offers"][action_index]["id"])
            != int(command["offer_id"])
        ):
            raise RuntimeError("Academy decision drifted from semantic authority.")
        semantic_command = (
            SemanticCommand(**json.loads(native_command))
            if native_command is not None
            else SemanticCommand(
                command_id=str(command["command_id"]),
                expected_revision=int(semantic_frame["revision"]),
                offer_id=int(command["offer_id"]),
                answers=tuple(command["answers"]),
            )
        )
        transition_json, next_obs, reward, _, _, _ = self.env.step_semantic_command(
            semantic_command.to_json()
        )
        semantic_transition = SemanticTransition.from_json(transition_json)
        if semantic_transition.receipt.command_id != str(command["command_id"]):
            raise RuntimeError("Semantic receipt command identity drifted.")
        if (
            semantic_transition.receipt.before_revision
            != semantic_command.expected_revision
        ):
            raise RuntimeError("Semantic receipt source revision drifted.")
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
        self.trace.events.append(
            TraceEvent(
                actor=actor,
                observation=observation,
                actions=actions,
                action=action_index,
                action_description=action_description,
                reward=float(reward),
                auto=auto,
                bot_decision=self._bot_decision,
                analysis_observation=analysis_observation,
                committed_at=trace_store.utc_now_iso(),
                analysis_shown=(
                    actor == "hero"
                    and not auto
                    and self._live_analysis is not None
                    and self._live_analysis.shown(len(self.trace.events))
                ),
            )
        )
        if self._live_analysis is not None:
            self._live_analysis.advance(action_index)
        self.obs = next_obs
        self._bot_decision = None
        self._advance_protocol_revision()
        caused_by = None if command is None or auto else str(command["command_id"])
        viewer_presentation_events = self._commit_step_presentation(
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
                        else enum_name(
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
                    semantic_receipt=(
                        None
                        if semantic_transition is None
                        else semantic_transition.receipt
                    ),
                    state_before=state_before,
                    state_after=state_after,
                    semantic_events=semantic_events,
                    presentation_events=deepcopy(
                        viewer_presentation_events[self.human_seat]
                    ),
                    viewer_presentation_events=(
                        deepcopy(viewer_presentation_events[0]),
                        deepcopy(viewer_presentation_events[1]),
                    ),
                    encountered_definition_ids=encountered_definition_ids,
                )
            )

        try:
            self._observe_policies()
        finally:
            if next_obs.game_over:
                self._close_policies()

    def _choose_bot(self, policy: BotPolicy) -> BotDecision | None:
        """Retain failed proposals on the same native decision and policy session.

        Only proposal work belongs here. Native commitment, observer delivery and
        recording failures must still interrupt: they may follow a mutation.
        The mirror_retry wire name is retained for existing clients.
        """
        assert self.env is not None and self.obs is not None
        self._bot_decision = None
        self.mirror_retry = None
        self.work.stage(
            "search"
            if isinstance(policy, FlatSearchPolicy)
            or isinstance(policy, PolicyRunner)
            and policy.search is not None
            else "choose",
            self.match_id,
            self.revision,
            int(self.obs.agent.player_index),
        )
        try:
            proposal = choose_bot(policy, self.env, self.obs)
        except Exception as exc:
            code = exc.reason if isinstance(exc, SearchError) else "policy_failed"
            if isinstance(exc, (SearchError, ValueError)):
                message = str(exc)
            else:
                logging.getLogger(__name__).exception("Bot proposal failed")
                message = (
                    f"Bot decision failed ({type(exc).__name__}); see the server log."
                )
            self.mirror_retry = {
                "code": code,
                "message": message,
                "match_id": self.match_id,
                "expected_revision": self.revision,
            }
            return None
        self.work.stage("advance", self.match_id, self.revision)
        self._bot_decision = proposal
        return proposal

    def _auto_play_villain(self) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            return
        if self.villain_policy is None:
            raise RuntimeError("Villain policy not initialized.")

        steps = 0
        while not self.obs.game_over and not _is_hero_turn(self.obs, self.human_seat):
            if steps >= MAX_AUTOPLAY_STEPS:
                raise RuntimeError("Villain auto-play exceeded safety step limit.")
            steps += 1

            context = self._build_decision_context(self.obs, viewer=1 - self.human_seat)
            if self._villain_offer_policy is not None:
                offer_id = int(self._villain_offer_policy(context))
                action_index = context.action_by_offer.get(offer_id)
                if action_index is None:
                    raise RuntimeError(
                        f"Villain offer policy selected unknown offer: {offer_id}"
                    )
            else:
                self.authority_fallback_counters["legacy_fixed_action"] += 1
                proposal = self._choose_bot(self.villain_policy)
                if proposal is None:
                    return
                action_index = proposal.action
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

        Alternates villain play with automated hero priority passes. Native
        skipping already resolved sole legal actions. Each automatic pass uses
        the same recorded authority path as a click.
        Bounded so a stuck engine cannot spin the server forever.
        """
        if self.mirror_retry is not None:
            return
        if (
            self.watch_mode
            or self.env is None
            or self.obs is None
            or self.trace is None
        ):
            return

        hero_steps = 0
        while True:
            self._auto_play_villain()
            if self.mirror_retry is not None:
                return
            if self.obs.game_over:
                self._f6_turn = None
                return

            # Hero is to act from here on.
            self._maybe_clear_f6()
            if self._should_surface():
                return

            if hero_steps >= MAX_AUTOPLAY_STEPS:
                raise RuntimeError("Hero automatic play exceeded safety step limit.")
            hero_steps += 1

            actions = trace_store.academy_actions(self.obs, self.human_seat)
            action_index = _pass_priority_index(actions)
            if action_index is None:
                return
            self._step_and_record(
                actor="hero",
                action_index=action_index,
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
        if not _has_non_mana_play(obs):
            return False
        if self.stop_on_stack and len(obs.stack_objects) > 0:
            return True

        step_name = enum_name(StepEnum, obs.turn.step)
        stop_step = ENGINE_STEP_TO_STOP_STEP.get(step_name)
        if stop_step is None:
            # Untap/cleanup/end-of-combat: no stop exists for these steps.
            return False
        side = "my" if _hero_is_active_player(obs, self.human_seat) else "opponent"
        if stop_step not in self.stops.get(side, []):
            return False
        if stop_step in ACTIVE_COMBAT_STOP_STEPS and not _has_attackers(obs):
            return False
        return True

    def _stops_payload(self) -> dict[str, Any]:
        return {
            "my": list(self.stops.get("my", [])),
            "opponent": list(self.stops.get("opponent", [])),
            "stop_on_stack": self.stop_on_stack,
            "auto_pass": self.auto_pass,
        }

    def _drain_surface_metadata(self, projection: dict[str, Any]) -> dict[str, Any]:
        """Drain step boards and auto-pass count outside the canonical frame.

        `projection` is the frame these travel with; step boards are patches
        back from it (`step_boards.wire_boards`).
        """
        payload: dict[str, Any] = {}
        auto_passed = self._auto_passed_since_surface
        self._auto_passed_since_surface = 0
        boards = wire_boards(self._step_boards, projection)
        self._step_boards = []
        if auto_passed:
            payload["auto_passed"] = auto_passed
        if boards:
            payload["boards"] = boards
        return payload

    def _advance_protocol_revision(self) -> None:
        self.revision += 1
        if self.mirror_retry is not None:
            self.mirror_retry["expected_revision"] = self.revision
        self.published_prompt = None
        self._pending_decision = None

    def _commit_step_presentation(
        self,
        *,
        from_revision: int,
        actor: int,
        caused_by: str | None,
    ) -> list[list[dict[str, Any]]]:
        """Commit one engine step into both authorized semantic tracks."""
        assert self.env is not None
        viewer_events = [
            projector.project(
                self.env.presentation_facts_json(viewer),
                from_revision=from_revision,
                to_revision=self.revision,
                caused_by=caused_by if viewer == actor else None,
            )
            for viewer, projector in enumerate(self.presentation)
        ]
        positions = self.env.take_presentation_positions()
        for viewer, events in enumerate(viewer_events):
            self.canonical_presentation[viewer].extend(events)

        live_events = viewer_events[self.human_seat]
        self.presentation_events.extend(live_events)
        step_boards = self._keep_step_board(live_events, positions)
        if len(self.presentation_events) > MAX_PRESENTATION_EVENTS:
            self.presentation_events = self.presentation_events[
                -MAX_PRESENTATION_EVENTS:
            ]
        if live_events and self.trace is not None and self.trace.events:
            trace_boards: list[WireStepBoard] = []
            if step_boards and self.obs is not None:
                post_viewer = (
                    self.human_seat
                    if self.obs.game_over
                    else int(self.obs.agent.player_index)
                )
                post_board = person_board(
                    decision_surface_view(self.env, post_viewer),
                    self.human_seat,
                    winner=self.env.winner_index() if self.obs.game_over else None,
                )
                trace_boards = wire_boards(step_boards, post_board)
            index = len(self.trace.events) - 1
            self.trace.events[index] = replace(
                self.trace.events[index],
                presentation=deepcopy(live_events),
                boards=trace_boards,
            )
        return viewer_events

    def _keep_step_board(
        self, events: list[dict[str, Any]], positions: list[managym.Env]
    ) -> list[StepBoard]:
        """Keep the person's board after each line of a step's `events`.

        Each event names the engine position showing the board just after it
        (`PresentationProjector.position`); a run of events on one position
        shares its board, and events after the last position are on the
        step's own board. Live play only: Watch and Replay have their own
        transport.
        """
        if self.watch_mode or self.env is None:
            return []
        events = [event for event in events if event["kind"]["kind"] != "passed"]
        if not events:
            return []
        runs: list[tuple[int | None, list[dict[str, Any]]]] = []
        for event in events:
            position = self.presentation[self.human_seat].position(int(event["seq"]))
            if runs and runs[-1][0] == position:
                runs[-1][1].append(event)
            else:
                runs.append((position, [event]))
        kept: list[StepBoard] = []
        for position, run in runs:
            env = self.env if position is None else positions[position]
            board = StepBoard(
                first=int(run[0]["seq"]),
                last=int(run[-1]["seq"]),
                projection=decision_surface_view(env, self.human_seat),
            )
            kept.append(board)
            self._step_boards.append(board)
        del self._step_boards[:-MAX_STEP_BOARDS]
        return kept

    def _publish_current_prompt(self) -> PublishedPrompt | None:
        if self.obs is None or self.obs.game_over:
            self.published_prompt = None
            return None
        if self.published_prompt is not None:
            return self.published_prompt

        self.published_prompt = self._build_decision_context(
            self.obs,
            viewer=self.human_seat,
        )
        frame = ExperienceFrame.model_validate(self.published_prompt.frame)
        address = DecisionAddressV2.from_frame(
            replay_id=f"replay.{self.match_id}",
            match_id=self.match_id,
            ordinal=len(self.canonical_decisions),
            viewer=self.human_seat,
            frame=frame,
            presentation_cursor=len(self.canonical_presentation[self.human_seat]),
        )
        self._pending_decision = PendingCanonicalDecision(
            address=address,
            frame=frame,
            root=self.env.clone_env(),
        )
        if self._live_analysis is not None and self.trace is not None:
            self._live_analysis.decide(len(self.trace.events), self.revision)
        return self.published_prompt

    def _build_decision_context(
        self,
        obs: managym.Observation,
        *,
        viewer: int,
    ) -> DecisionContext:
        """Build one exact actor-safe frame and raw-action lowering map."""
        actions = trace_store.academy_actions(obs, self.human_seat)
        action_space = enum_name(
            ActionSpaceEnum, int(obs.action_space.action_space_type)
        )
        prompt_id = self._next_prompt_id
        self._next_prompt_id += 1
        action_by_offer = {
            offer_id: int(action["index"]) for offer_id, action in enumerate(actions)
        }
        subjects = json.loads(self.env.decision_offer_subjects_json(viewer))
        offers = [
            {
                "id": offer_id,
                **({"subjects": subjects[offer_id]} if subjects[offer_id] else {}),
                "actor": viewer,
                "verb": offer_verb(str(action.get("type", ""))),
                "source": next(
                    (
                        subject["object"]["subject"]
                        for subject in subjects[offer_id]
                        if subject["role"] == "source"
                    ),
                    None,
                ),
                "label": action["description"],
                "help": None,
                "choices": [],
                "confirm_label": action["description"],
                # Retained readers use raw fields; the table navigates typed subjects.
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

    def _deck_names_for(self, viewer: int) -> dict[str, str] | None:
        """Display names `viewer` may know; a closed opponent's deck is unnamed."""
        if self.deck_names is None:
            return None
        names = dict(self.deck_names)
        if self.decklists_closed:
            names["villain" if viewer == self.human_seat else "hero"] = CLOSED_DECK_NAME
        return names

    def _build_frame(
        self,
        obs: managym.Observation,
        *,
        viewer: int,
        prompt: dict[str, Any] | None,
        offers: list[dict[str, Any]],
        action_space: str,
    ) -> dict[str, Any]:
        projection = decision_surface_view(self.env, viewer)
        context = projection.get("decision_context")
        if context is not None and prompt is not None:
            prompt = {
                **prompt,
                "instruction": context["instruction"],
                "title": context["stage"].replace("_", " ").capitalize(),
            }
        frame_without_hash: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "match_id": self.match_id,
            "revision": self.revision,
            "content_hash": self.content_hash,
            "asset_manifest_hash": self.asset_manifest_hash,
            "status": "game_over"
            if obs.game_over
            else ("ready" if prompt else "thinking"),
            "prompt": prompt,
            "projection": projection,
            "offers": offers,
            "winner": self.env.winner_index() if obs.game_over else None,
            "action_space": action_space,
            "stops": self._stops_payload(),
            "deck_names": self._deck_names_for(viewer),
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

        if not self.obs.game_over and not _is_hero_turn(self.obs, self.human_seat):
            # The pending choice belongs to the bot. Publish the human's view
            # without a human prompt or any of the bot's private legal offers.
            self.published_prompt = None
            self._persist_attempt()
            return self._build_frame(
                self.obs,
                viewer=self.human_seat,
                prompt=None,
                offers=[],
                action_space="",
            )

        if self.obs.game_over:
            self._persist_attempt(end_reason="game_over")
            return self._build_frame(
                self.obs,
                viewer=self.human_seat,
                prompt=None,
                offers=[],
                action_space="",
            )
        if not _is_hero_turn(self.obs, self.human_seat):
            # A policy failure can leave a committed command awaiting another
            # seat. Recovery must never publish that seat's private offers.
            self._persist_attempt()
            return self._build_frame(
                self.obs,
                viewer=self.human_seat,
                prompt=None,
                offers=[],
                action_space="",
            )
        prompt = self._publish_current_prompt()
        if prompt is None:
            raise RuntimeError("Non-terminal authority has no decision context.")
        self._persist_attempt()
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

        head = self.presentation[self.human_seat].next_seq
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
            "accepted_commands": deepcopy(list(self.accepted_commands.values())),
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
        transient = self._drain_surface_metadata(frame["projection"])
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
        if self._live_analysis is not None:
            payload["analysis"] = {"policy": self._live_analysis.policy}
        payload.update(transient)
        payload["mirror_retry"] = self.mirror_retry
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

    def _persist_attempt(
        self, *, end_reason: str | None = None, config_changed: bool = False
    ) -> None:
        """Save the attempt once per committed change.

        A live save is skipped when the revision and event count match the
        last write; a caller that edits `trace.config` without stepping passes
        `config_changed` so that edit still reaches the store.
        """
        if self._preparing_demo and end_reason is None:
            return
        if self.trace is None or self.obs is None or self.env is None:
            return
        finished = end_reason is not None
        if finished and self._trace_saved:
            return
        if end_reason == "game_over" and not self.obs.game_over:
            raise ValueError("Only an authoritative terminal game can be completed.")
        written = (self.match_id, self.revision, len(self.trace.events))
        if not finished and not config_changed and written == self._persisted:
            return

        canonical = self._canonical_replay_snapshot()
        study_provider = (
            StudyForkProvider(canonical, self._study_roots) if finished else None
        )
        retained = replace(
            self.trace,
            final_observation=decision_surface_view(self.env, self.human_seat),
            winner=self.env.winner_index() if self.obs.game_over else None,
            end_reason=end_reason if finished else self.trace.end_reason,
            # Only a finished trace is kept; a live save encodes the replay incrementally.
            canonical_replay=canonical.model_dump(mode="json") if finished else None,
        )
        if finished:
            self.trace = retained
        world = {
            **self.env.content_pack_manifest(),
            "rules_binary_sha256": RULES_BINARY_SHA256,
            "experience_source_sha256": PLAY_SOURCE_SHA256,
        }
        self.attempt_store.save(
            self.match_id,
            self.revision,
            world,
            retained.to_record(),
            trace_json=retained.to_json(
                self._encoded_events, self._canonical_replay_json(canonical)
            ),
            finished=finished,
            players=self.record_players,
            origin=(
                (os.getenv("ACADEMY_PLAY_RECORD_ORIGIN") or "bot_evaluation")
                if self.watch_mode
                else "automated_validation"
                if self._demo_name
                else None
            ),
        )
        self._persisted = written
        if finished:
            self._study_provider = study_provider
            self._trace_saved = True
            if not self.watch_mode:
                save_game_numbers(
                    self.attempt_store,
                    self.match_id,
                    RetainedAttempt(retained.to_record(), json.dumps(world)),
                )

    def canonical_replay(self) -> CanonicalReplayV1:
        """Project the canonical decisions committed so far.

        The same replay and row identities are persisted when the game ends;
        no unplayed prompt is assigned an address.
        """
        return self._canonical_replay_snapshot().model_copy(deep=True)

    def _canonical_replay_snapshot(self) -> CanonicalReplayV1:
        """Internal snapshot of append-only records; callers never mutate rows.

        Persistence serializes this directly. Public consumers receive a deep
        copy so they cannot change retained frames, offers or Command bindings.
        """
        return CanonicalReplayV1(
            version=CANONICAL_REPLAY_VERSION,
            replay_id=f"replay.{self.match_id}",
            match_id=self.match_id,
            content_hash=self.content_hash,
            asset_manifest_hash=self.asset_manifest_hash,
            decisions=list(self.canonical_decisions),
            presentation_tracks=[
                ViewerPresentationTrack(
                    viewer=viewer,
                    head=len(events),
                    events=list(events),
                )
                for viewer, events in sorted(self.canonical_presentation.items())
            ],
        )

    def _canonical_replay_json(self, canonical: CanonicalReplayV1) -> str:
        """`json.dumps(canonical.model_dump(mode="json"))`, encoding only new rows.

        Decisions and presentation events are append-only, so each is encoded
        once. Decision rows are the snapshot's own objects; a track's events
        are validated from `canonical_presentation`'s dicts, so they are
        encoded from those dicts through the same model.
        """
        head = canonical.model_dump(
            mode="json", exclude={"decisions", "presentation_tracks"}
        )
        tracks = [
            splice(
                {"viewer": viewer, "head": len(events)},
                {
                    "events": encode_list(
                        events,
                        self._encoded_presentation.setdefault(viewer, []),
                        lambda event: json.dumps(
                            PresentationEvent.model_validate(event).model_dump(
                                mode="json"
                            )
                        ),
                    )
                },
            )
            for viewer, events in sorted(self.canonical_presentation.items())
        ]
        return splice(
            head,
            {
                "decisions": encode_list(
                    canonical.decisions,
                    self._encoded_decisions,
                    lambda row: json.dumps(row.model_dump(mode="json")),
                ),
                "presentation_tracks": "[" + ", ".join(tracks) + "]",
            },
        )

    def study_fork_provider(self) -> StudyForkProvider:
        """Expose Game's validated retained-root capability to consumers.

        The provider owns replay lookup and isolated root cloning. Callers do
        not receive or inspect ``GameSession``'s retained-root storage.
        """

        if self._study_provider is not None:
            return self._study_provider
        return StudyForkProvider(self.canonical_replay(), self._study_roots)

    def fork_study(self, raw_address: str) -> StudyBranch:
        """Fork an authorized acting-player decision for ephemeral Study."""
        viewer = (
            int(parse_decision_address(raw_address).viewer)
            if self.watch_mode
            else self.human_seat
        )
        return self.study_fork_provider().fork(raw_address, viewer)

    def retry_study(
        self,
        *,
        trace_id: str,
        replay: CanonicalReplayV1,
        raw_address: str,
        raw_command: Any,
    ) -> dict[str, Any]:
        """Commit one canonical command on an exact retained Study fork. A
        game has one Study attempt at a time."""
        live_trace_id = self.trace_id or f"live.{self.match_id}"
        if live_trace_id != trace_id:
            raise StudyBranchUnavailableError(
                "Study roots are unavailable for this recording."
            )
        if self._study_attempts:
            raise StudyCommandUnavailableError("study_attempt_active")
        viewer = (
            int(parse_decision_address(raw_address).viewer)
            if self.watch_mode
            else self.human_seat
        )
        projection = project_replay(replay, viewer)
        restored = restore_decision(replay, raw_address, viewer)
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

    def reveal_study(self, attempt_id: str) -> dict[str, Any]:
        """Release evidence only after the attempt owns an accepted Retry."""
        attempt = self._require_study_attempt(attempt_id)
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
    ) -> dict[str, Any]:
        """Project one labelled plan without reusing the player's Retry fork."""
        attempt = self._require_study_attempt(attempt_id)
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

    def return_from_study(self, attempt_id: str) -> dict[str, Any]:
        """Close one branch and restore the exact canonical decision."""
        attempt = self._require_study_attempt(attempt_id)
        self._study_attempts.pop(attempt_id)
        receipt = attempt.branch.return_to_recorded()
        restored = RestoredReplayDecision.model_validate(
            receipt.model_dump(mode="python", exclude={"source_digest", "execution"})
        )
        if restored != attempt.restored:
            raise RuntimeError("Study return identity drifted.")
        return restored.model_dump(mode="json")

    def has_study_attempt(self, attempt_id: str) -> bool:
        return attempt_id in self._study_attempts

    def _require_study_attempt(self, attempt_id: str) -> StudyAttempt:
        attempt = self._study_attempts.get(attempt_id)
        if attempt is None:
            raise StudyAttemptNotFoundError("study_attempt_not_found")
        return attempt

    def _execute_study_command(
        self,
        branch: StudyBranch,
        restored: RestoredReplayDecision,
        command: Command,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not any(offer.id == command.offer_id for offer in restored.frame.offers):
            raise StudyCommandUnavailableError("study_command_offer_missing")
        try:
            branch.submit_semantic(command)
        except (managym.AgentError, StudyBranchUnavailableError) as exc:
            raise StudyCommandUnavailableError("study_command_rejected") from exc
        presentation = PresentationProjector().project(
            branch.presentation_facts(),
            from_revision=restored.revision,
            to_revision=restored.revision + 1,
            caused_by=command.command_id,
        )
        return branch.current_projection(), presentation

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

    def _observe_policies(self) -> None:
        if self.env is not None and self.obs is not None:
            observe_policies(
                self.watch_policies or (self.villain_policy,), self.env, self.obs
            )

    def _close_policies(self) -> None:
        self._end_live_analysis()
        close_policies((self.villain_policy, *(self.watch_policies or ())))

    def _end_live_analysis(self) -> None:
        if self._live_analysis is not None:
            self._live_analysis.close()
            self._live_analysis = None

    def show_analysis(
        self,
        shown: bool,
        policy: str | None,
        publish: Callable[[dict[str, Any]], concurrent.futures.Future[bool]],
    ) -> dict[str, Any]:
        """Show or hide a policy's reading of the person's decisions in this game.

        Showing starts a `LiveAnalysis` from the game so far; its readings go
        to `publish`, never into a frame. Hiding stops it: decisions from the
        next one on are not marked `analysis_shown`.
        """
        if self.env is None or self.obs is None or self.trace is None:
            raise ValueError("No active game session. Send new_game first.")
        if self.watch_mode:
            raise PlayRequestError("conflict", "A watched game has no seat of yours.")
        self._end_live_analysis()
        # Admission takes seconds the first time; the connection admits off the
        # game lock before dispatching here (`policy_admissions`).
        choices = analysing_policies(self.trace.config, policy_admissions())
        state: dict[str, Any] = {
            "type": "analysis_state",
            "shown": False,
            "policy": None,
            "policies": choices.policies,
            "unavailable": None
            if choices.policies
            else "No registered policy can read this game.",
        }
        if not shown or not choices.policies:
            return state
        key = policy or choices.default or choices.policies[0]["sha256"]
        chosen = next((p for p in choices.policies if p["sha256"] == key), None)
        if chosen is None:
            raise PlayRequestError("not_found", "That policy cannot read this game.")
        config = self.trace.config
        setup = config.to_rust()
        seat = self.human_seat
        events = [
            {
                "actor": event.actor,
                "auto": event.auto,
                "action": event.action,
                "action_description": event.action_description,
            }
            for event in self.trace.events
        ]
        snapshot = RetainedAttempt(
            {
                "config": config.to_record(),
                "events": events,
                "end_reason": "active",
                "winner": None,
            },
            json.dumps(self.env.content_pack_manifest()),
        )
        self._live_analysis = LiveAnalysis(
            snapshot,
            seat,
            chosen,
            lambda: policy_registry().load(
                key, setup, seat=seat, seed=int(config.seed or 0)
            ),
            publish,
        )
        if self.published_prompt is not None:
            self._live_analysis.decide(len(self.trace.events), self.revision)
        return {**state, "shown": True, "policy": chosen}

    def close(self, end_reason: str) -> None:
        """End the game with `end_reason` and release it."""
        try:
            self._close_study_attempts()
            self._persist_attempt(end_reason=end_reason)
        finally:
            self.release()

    def release(self) -> None:
        """Free the Env and policies, ending nothing: the record is already
        saved, and a played game is taken up again from it (`resume`)."""
        try:
            self._close_study_attempts()
            self._close_policies()
        finally:
            self.env = None
            self.obs = None
            self.villain_policy = None
            self.watch_policies = None
            self._bot_decision = None
            self.mirror_retry = None


def _error_message(message: str) -> dict[str, str]:
    return {"type": "error", "message": message}


class PlayRequestError(ValueError):
    """A `/ws/play` request refused before it reached the game."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _control_error(error: PlayRequestError) -> dict[str, str]:
    return {"type": "control_error", "code": error.code, "message": str(error)}


def _new_game_session() -> GameSession:
    """Construct the normal runtime authority with unavailable Study evidence."""
    return GameSession()


def end_game(game_id: str, end_reason: str) -> None:
    """End a live game with `end_reason` and drop it from memory."""
    live = LIVE_GAMES.pop(game_id, None)
    if live is None:
        return
    live.game.close(end_reason=end_reason)
    live.websocket = None


def evict_game(game_id: str, watch_end_reason: str) -> None:
    """Drop an idle or shut-down game from memory. A played game ends
    nothing and stays unfinished; a watched game cannot be rebuilt from its
    record, so it ends with `watch_end_reason`."""
    live = LIVE_GAMES.get(game_id)
    if live is None:
        return
    if live.game.watch_mode:
        end_game(game_id, end_reason=watch_end_reason)
        return
    del LIVE_GAMES[game_id]
    live.game.release()
    live.websocket = None


def _cleanup_expired_games() -> None:
    """Release idle games. A game with a connection is never idle: the person
    may be thinking, and its board still asks for previews."""
    now = _now_utc()
    expired = [
        game_id
        for game_id, live in LIVE_GAMES.items()
        if live.expires_at <= now and live.websocket is None and not live.lock.locked()
    ]
    for game_id in expired:
        evict_game(game_id, SESSION_EXPIRED_END_REASON)


# One rebuild of a game at a time: a second would replay and extend the same
# stored record.
_RESUMING: dict[str, asyncio.Lock] = {}


async def take_up_game(game_id: str) -> LiveGame:
    """The live game `game_id`, rebuilt from its stored record when it is not
    in memory. Raises ValueError where it is not an unfinished played game or
    cannot be rebuilt (`GameSession.resume`)."""
    async with _RESUMING.setdefault(game_id, asyncio.Lock()):
        live = LIVE_GAMES.get(game_id)
        if live is None:
            game = _new_game_session()
            try:
                await asyncio.to_thread(game.resume, game_id)
            finally:
                # The rebuild loaded an opponent and advanced the game; once it
                # returns, no reply may describe that as work still running.
                game.work.finish()
            live = LIVE_GAMES[game_id] = LiveGame(game)
    if live.game.watch_mode:
        raise ValueError("game_is_watch: A watched game has no seat of yours.")
    return live


def _opponent_label(config: GameConfig) -> str:
    """The opponent's name and how it decides: "Search 64 · search",
    "Heuristic · scripted", "<checkpoint> · checkpoint w17"."""
    if config.villain_type == "search":
        return f"Search {config.villain_sims} · search"
    if config.opponent is not None:
        policy = (config.opponent_checkpoint or {}).get("policy")
        kind = opponent_kind(policy) if isinstance(policy, dict) else None
        return " · ".join(filter(None, (config.opponent["name"], kind)))
    if config.villain_type == "passive":
        return "Passive · scripted"
    return config.villain_type.capitalize()


def _reply(live: LiveGame, response: dict[str, Any]) -> dict[str, Any]:
    """A response as the browser reads it: with the game's id, opponent and
    setup, and the work in progress."""
    game = live.game
    payload = deepcopy(response)
    payload["work"] = game.work.payload()
    payload["mirror_retry"] = game.mirror_retry
    trace = game.trace
    payload["mirror"] = asdict(trace.config).get("mirror") if trace else None
    if trace is not None:
        payload["game"] = game.match_id
        payload["opponent"] = trace.config.opponent
        payload["opponent_label"] = _opponent_label(trace.config)
        # What Rematch, Swap sides and Play again start from.
        payload["setup"] = shown_setup(
            trace.config, game.obs is not None and game.obs.game_over
        )
    return payload


async def _attach_websocket(live: LiveGame, websocket: WebSocket) -> None:
    """Make `websocket` the game's one connection; the one it replaces closes."""
    previous = live.websocket
    live.websocket = websocket
    live.touch()
    if previous is not None and previous is not websocket:
        with suppress(Exception):
            await previous.close(code=4000)


def _detach_websocket(game_id: str | None, websocket: WebSocket) -> None:
    live = LIVE_GAMES.get(game_id) if game_id is not None else None
    if live is not None and live.websocket is websocket:
        live.websocket = None
        live.touch()


def _parse_request(raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise PlayRequestError(
            "invalid_message", "WebSocket payload must be a JSON object."
        )
    message_type = raw.get("type")
    if not isinstance(message_type, str) or message_type not in REQUEST_TYPES:
        raise PlayRequestError(
            "unsupported_message", f"Unsupported message type: {message_type}"
        )
    try:
        return REQUEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise PlayRequestError(
            "invalid_message", f"Invalid {message_type} message."
        ) from exc


def _dispatch_request(
    game: GameSession,
    request: Any,
    committed: Callable[[dict[str, Any]], None],
    publish_analysis: Callable[[dict[str, Any]], concurrent.futures.Future[bool]],
) -> dict[str, Any]:
    """Run one request on its game, under the game's lock.

    `committed` delivers an accepted Command before the opponent replies;
    `publish_analysis` delivers a live analysis's readings without waiting."""
    if isinstance(request, CommandMessage):
        return game.hero_command(
            request.command.model_dump(mode="json"), committed=committed
        )
    if isinstance(request, ActionMessage):
        return game.hero_action(request.index)
    if isinstance(request, PassTurnMessage):
        return game.pass_turn()
    if isinstance(request, RetryMirrorMessage):
        return game.retry_mirror(request.match_id, request.expected_revision)
    if isinstance(request, SetStopsMessage):
        return game.set_stops(request.stops, request.stop_on_stack, request.auto_pass)
    if isinstance(request, ShowAnalysisMessage):
        return game.show_analysis(request.shown, request.policy, publish_analysis)
    raise RuntimeError(f"closed dispatch missing handler for {request.type}")


def _launch_metadata(launch_id: object) -> dict[str, Any]:
    """Return setup to the requesting tab even when a known launch is refused."""
    launch = LAUNCHES.get(launch_id) if isinstance(launch_id, str) else None
    if launch is None:
        return {}
    return {
        "launch": launch_id,
        "launch_setup": {
            key: value for key, value in launch.config.items() if key != "seed"
        },
    }


def _request_error(request: object, message: str) -> dict[str, Any]:
    """An error reply; a refused new game keeps its launch setup for a retry."""
    error: dict[str, Any] = dict(_error_message(message))
    if isinstance(request, dict) and request.get("type") == "new_game":
        config = request.get("config")
        if isinstance(config, dict):
            error.update(_launch_metadata(config.get("launch")))
    return error


def _start_config(request: NewGameMessage | RematchMessage) -> dict[str, Any]:
    """The setup a start request asks for. A rematch reads the setup its game
    was played with, as recorded (stops changed during play included)."""
    if isinstance(request, NewGameMessage):
        config = dict(request.config)
        launch_id = config.pop("launch", None)
        if launch_id is None:
            return config
        launch = LAUNCHES.get(launch_id) if isinstance(launch_id, str) else None
        if launch is None:
            raise ValueError(
                "launch_unknown: This launch is unknown; run the launch command again or choose a new game."
            )
        if launch.mode != "play":
            raise ValueError(
                "launch_not_play: This launch opens Watch setup, not a game to play."
            )
        return {**config, **launch.config}
    if {"launch", "demo", "played_from"}.intersection(request.config):
        raise ValueError("Rematch setup contains reserved fields.")
    live = LIVE_GAMES.get(request.game)
    if live is not None and live.game.trace is not None:
        played = live.game.trace.config
    else:
        source = AttemptStore(trace_store.TRACES_DIR / "play.sqlite").load(request.game)
        if source is None:
            raise ValueError("rematch_unavailable: That game is not saved here.")
        played = GameConfig(**source["config"])
    if played.watch_players is not None:
        raise ValueError("rematch_unavailable: A watched game has no seat of yours.")
    return continuation_config(setup_draft(played), request.config, request.intent)


def _cache_recovery(live: LiveGame) -> None:
    """Project the game once at a safe worker boundary, for a connection that
    arrives while its next request is still running."""
    if live.game.obs is None:
        live.published = None
        return
    recovery = live.game.current_recovery("reconnect")
    frame = recovery["frame"]
    live.published = _reply(
        live,
        {
            "type": "game_over" if frame["projection"]["game_over"] else "observation",
            "data": frame["projection"],
            "actions": [],
            "winner": frame["winner"],
            "frame": frame,
            "recovery": recovery,
        },
    )


async def _publish_work(live: LiveGame, status: WorkStatus) -> None:
    # Status has no projection path. Resume uses the last permitted publication.
    if live.websocket is not None:
        with suppress(Exception):
            await live.websocket.send_json(
                {
                    "type": "work_status",
                    "game": live.game.match_id if live.game.trace else None,
                    "work": asdict(status),
                }
            )


async def _send_live_analysis(live: LiveGame, message: dict[str, Any]) -> bool:
    """Send a live analysis reading to the game's connection; true once sent."""
    socket = live.websocket
    if socket is None:
        return False
    try:
        await socket.send_json(message)
    except Exception:
        if live.websocket is socket:
            live.websocket = None
        return False
    return True


async def _send_reply(live: LiveGame, response: dict[str, Any], **extra: Any) -> None:
    _cache_recovery(live)
    socket = live.websocket
    if socket is None:
        return
    try:
        await socket.send_json({**_reply(live, response), **extra})
    except Exception:
        if live.websocket is socket:
            live.websocket = None


async def _start_game(
    request: NewGameMessage | RematchMessage, websocket: WebSocket
) -> LiveGame:
    """Start the game a request asks for and make `websocket` its connection.

    A configured launch starts one game: sent again, it takes that game up.
    A failure starts nothing and leaves every other game as it was."""
    launch_id = (
        request.config.get("launch") if isinstance(request, NewGameMessage) else None
    )
    config = _start_config(request)
    started = LAUNCH_GAMES.get(launch_id) if isinstance(launch_id, str) else None
    if started is not None:
        try:
            live = await take_up_game(started)
        except ValueError as exc:
            raise ValueError(
                f"launch_played: This launch already played game {started}; open its replay or start a new game."
            ) from exc
        await _attach_websocket(live, websocket)
        async with live.lock:
            await _send_reply(
                live, live.game.current_message(), **_launch_metadata(launch_id)
            )
        return live

    game = _new_game_session()
    live = LiveGame(game, websocket)
    loop = asyncio.get_running_loop()

    def publish_work(status: WorkStatus) -> None:
        # Called while the worker is paused at a safe boundary.
        asyncio.run_coroutine_threadsafe(_publish_work(live, status), loop).result()

    game.work.publish = publish_work
    started_ok = False
    try:
        response = await asyncio.to_thread(game.new_game, config)
        started_ok = True
    finally:
        game.work.publish = None
        game.work.finish(failed=not started_ok or game.mirror_retry is not None)
        if not started_ok:
            # A game that failed after its first save is over, never unfinished.
            if game.trace is not None:
                with suppress(Exception):
                    game.close("setup_failed")
            else:
                game.release()
    LIVE_GAMES[game.match_id] = live
    if isinstance(launch_id, str):
        LAUNCH_GAMES[launch_id] = game.match_id
    await _send_reply(live, response, **_launch_metadata(launch_id))
    return live


@app.websocket("/ws/play")
async def play_socket(websocket: WebSocket) -> None:
    _cleanup_expired_games()
    await websocket.accept()
    # The game this connection last took up or started.
    attached: str | None = None
    # The game a request is running on: an unexpected failure there ends it.
    running: str | None = None
    request: Any = None

    try:
        while True:
            try:
                # A failed send (the client left during a long Command, such as
                # a searched bot reply) closes the application side; that is a
                # disconnect, not a game error.
                if websocket.application_state is not WebSocketState.CONNECTED:
                    raise WebSocketDisconnect(code=1006)
                request = await websocket.receive_json()
            except WebSocketDisconnect:
                _detach_websocket(attached, websocket)
                raise

            _cleanup_expired_games()
            try:
                parsed = _parse_request(request)
                if isinstance(parsed, ResumeMessage):
                    live = await take_up_game(parsed.game)
                    if not parsed.take_over and live.websocket not in (
                        None,
                        websocket,
                    ):
                        raise ValueError(
                            "game_open_elsewhere: This game is open in another tab."
                        )
                    if attached != parsed.game:
                        _detach_websocket(attached, websocket)
                    await _attach_websocket(live, websocket)
                    attached = parsed.game
                    if live.lock.locked() and live.published is not None:
                        await websocket.send_json(
                            {**live.published, "work": live.game.work.payload()}
                        )
                        continue
                    async with live.lock:
                        await _send_reply(
                            live, live.game.current_message(parsed.presentation_cursor)
                        )
                    continue

                if isinstance(parsed, (NewGameMessage, RematchMessage)):
                    try:
                        live = await _start_game(parsed, websocket)
                    except sqlite3.Error as exc:
                        # Nothing started; the game in play is untouched.
                        raise ValueError(
                            "Game recording failed; this game may not have been saved."
                        ) from exc
                    previous, attached = attached, live.game.match_id
                    if previous != attached:
                        # The game left behind stays as it is, in memory until
                        # it idles out: unfinished and resumable, or finished
                        # and still open to Study.
                        _detach_websocket(previous, websocket)
                    continue

                if attached is None:
                    raise ValueError("No active game session. Send new_game first.")
                # A game dropped from memory while this connection idled is
                # rebuilt from its record.
                live = await take_up_game(attached)
                if live.websocket is not websocket:
                    await _attach_websocket(live, websocket)
                game = live.game
                if isinstance(parsed, ShowAnalysisMessage) and game.trace is not None:
                    # The first admission of a game's setups takes seconds;
                    # make it here, off the game lock, so play goes on while it
                    # runs and `show_analysis` then reads the cached answer.
                    with suppress(ImportError, OSError, ValueError):
                        await asyncio.to_thread(
                            policy_admissions().admitted, game.trace.config.to_rust()
                        )
                running = attached
                async with live.lock:
                    loop = asyncio.get_running_loop()

                    def committed(
                        outcome: dict[str, Any], live: LiveGame = live
                    ) -> None:
                        # Projection and delivery finish at the safe committed
                        # boundary before the worker may mutate the Env again.
                        asyncio.run_coroutine_threadsafe(
                            _send_reply(live, outcome), loop
                        ).result()

                    def publish_analysis(
                        message: dict[str, Any], live: LiveGame = live
                    ) -> concurrent.futures.Future[bool]:
                        # From the analysis worker: never waits for delivery.
                        return asyncio.run_coroutine_threadsafe(
                            _send_live_analysis(live, message), loop
                        )

                    def publish_work(status: WorkStatus, live: LiveGame = live) -> None:
                        # Called while the worker is paused at a safe boundary.
                        asyncio.run_coroutine_threadsafe(
                            _publish_work(live, status), loop
                        ).result()

                    game.work.stage(
                        "advance",
                        game.match_id if game.obs is not None else None,
                        game.revision if game.obs is not None else None,
                    )
                    status = game.work.read()
                    assert status is not None
                    _cache_recovery(live)
                    await _publish_work(live, status)
                    game.work.publish = publish_work
                    failed = True
                    dispatched = False
                    try:
                        response = await asyncio.to_thread(
                            _dispatch_request, game, parsed, committed, publish_analysis
                        )
                        failed = game.mirror_retry is not None
                        dispatched = True
                    finally:
                        game.work.publish = None
                        game.work.finish(failed=failed)
                        if not dispatched:
                            # A refused request has no reply to end the wait.
                            status = game.work.read()
                            assert status is not None
                            await _publish_work(live, status)
                    live.touch()
                    await _send_reply(live, response)
                running = None
                continue
            except PlayRequestError as exc:
                running = None
                await websocket.send_json(_control_error(exc))
                continue
            except ValueError as exc:
                running = None
                await websocket.send_json(_request_error(request, str(exc)))
                continue
    except WebSocketDisconnect:
        return
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, sqlite3.Error):
            message = "Game recording failed; this game may not have been saved."
        error = _request_error(request, message)
        if running is not None:
            # The game broke: it is over, and the client must not resume it.
            error["session_ended"] = True
            try:
                end_game(running, end_reason="error")
            except Exception:
                logging.getLogger(__name__).exception("Failed to retain stopped game")
                error["message"] = (
                    "Game recording failed; this game may not have been saved."
                )
        with suppress(Exception):
            await websocket.send_json(error)
        with suppress(Exception):
            await websocket.close()


@app.get("/api/card-art/credits")
def card_art_credits(names: list[str] = Query(default=[])) -> Response:
    return JSONResponse(
        card_art().credits(names[:1024]), headers={"Cache-Control": "no-store"}
    )


def _art_response(path: Path | None) -> Response:
    if path is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/card-faces")
def get_card_faces(names: list[str] = Query(default=[])) -> dict[str, CardFace]:
    """Printed cost, type line and keywords for names the viewer already sees."""
    return card_faces(names[:1024])


@app.get("/api/card-art/back.jpg")
def card_back() -> Response:
    return _art_response(card_art().back())


@app.get("/api/card-art/{kind}/{slug}.jpg")
def card_image(kind: Literal["art", "full"], slug: str) -> Response:
    return _art_response(card_art().image(slug, kind))


@app.get("/api/opponents")
async def get_opponents() -> dict[str, Any]:
    return await asyncio.to_thread(opponent_catalog)


@app.get("/api/deck-pools")
async def get_deck_pools() -> dict[str, Any]:
    if not deck_pools():
        defaults = await asyncio.to_thread(ensure_default_deck_pools)
        if defaults and not deck_pools():
            set_deck_pools(defaults)
    return {"pools": deck_pool_catalog()}


@app.get("/api/launch/{launch_id}")
def get_launch_setup(launch_id: str) -> dict[str, Any]:
    """Return a pre-configured launch setup without exposing server internals."""
    launch = LAUNCHES.get(launch_id)
    if launch is None:
        raise HTTPException(404, "launch_not_found")
    return deepcopy(launch.config)


@app.post("/api/policies/check")
def check_policy_setup(config: dict[str, Any]) -> dict[str, Any]:
    """Preview admission without loading a session; Start repeats the same checks."""
    try:
        parsed = _parse_game_config(config)
    except (ValueError, OSError, ImportError) as exc:
        return {"compatible": False, "reason": str(exc), "fit": None}
    fit: dict[str, Any] | None = None
    opp_key = config.get("opponent_sha256")
    if isinstance(opp_key, str) and opp_key:
        planned: dict[str, str] = {"decklists": str(parsed.decklists)}
        mana_payment = config.get("mana_payment")
        if isinstance(mana_payment, str) and mana_payment in ("manual", "auto"):
            planned["mana_payment"] = mana_payment
        mana_floats = config.get("mana_floats")
        if isinstance(mana_floats, str) and mana_floats in ("off", "usable", "all"):
            planned["mana_floats"] = mana_floats
        report = policy_settings_fit(opp_key, planned)
        if report is not None:
            fit = report.to_dict()
    return {"compatible": True, "reason": None, "fit": fit}


@app.get("/api/traces")
async def list_traces(
    response: Response,
    q: str = "",
    bot: str | None = None,
    pairing: str | None = None,
    deck: str | None = None,
    status: str | None = None,
    after: str | None = None,
    before: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Every game in the traces directory, newest first."""
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    try:
        # One extra row tells the client whether another page exists.
        rows = store.summaries(
            q=q,
            bot=bot,
            pairing=pairing,
            deck=deck,
            status=status,
            after=after,
            before=before,
            cursor=cursor,
            limit=limit + 1,
        )
        position = decode_cursor(cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Frozen JSON history has no player identity. Keep it readable and label
    # missing provenance honestly; identity filters cannot match it.
    if not any((bot, pairing, deck)) and status != "active":
        for row in trace_store.list_trace_summaries():
            timestamp = row["timestamp"] or ""
            legacy_status = (
                "completed" if row["end_reason"] == "game_over" else "stopped"
            )
            if (q and q.casefold() not in row["id"].casefold()) or (
                status and legacy_status != status
            ):
                continue
            if (after and timestamp[:10] < after) or (
                before and timestamp[:10] > before
            ):
                continue
            if position and (timestamp, row["id"]) >= position:
                continue
            rows.append(
                {
                    **row,
                    "timestamp": timestamp,
                    "status": legacy_status,
                    "players": [],
                    "record_version": 0,
                    "origin": "unknown",
                    "resumable": False,
                    "revision": None,
                    "ended_at": None,
                }
            )
    rows.sort(key=lambda row: (row["timestamp"], row["id"]), reverse=True)
    page = rows[:limit]
    if len(rows) > limit and page:
        response.headers["X-Etude-Next-Cursor"] = page_cursor(page[-1])
    response.headers["Cache-Control"] = "private, no-store"
    return page


def save_game_numbers(
    store: AttemptStore, trace_id: str, attempt: RetainedAttempt
) -> dict[str, Any] | None:
    """Compute and save the numbers of a game's stored steps; None where it
    does not re-execute here. A failure never reaches the game itself."""
    try:
        numbers = game_numbers(attempt)
    except Exception:  # noqa: BLE001 - numbers are derived; the game is saved
        logging.getLogger(__name__).exception("Summary of %s failed", trace_id)
        return None
    if numbers is None:
        return None
    store.save_numbers(
        trace_id, numbers.model_dump_json(), len(attempt.trace["events"])
    )
    return numbers.model_dump(mode="json")


@app.get("/api/traces/{trace_id}/numbers")
def get_game_numbers(trace_id: str) -> dict[str, Any]:
    """A played game's numbers so far, computed on first read where none are
    saved for its stored steps (null where it does not re-execute here)."""
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    saved = store.numbers(trace_id)
    if saved is None:
        raise HTTPException(404, "recording_not_found")
    _, numbers_json = saved
    if numbers_json is not None:
        return {"numbers": json.loads(numbers_json)}
    attempt = store.load_attempt(trace_id)
    assert attempt is not None
    if attempt.trace["config"].get("watch_players") is not None:
        return {"numbers": None}
    return {"numbers": save_game_numbers(store, trace_id, attempt)}


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


def _live_game_for_trace(trace_id: str) -> LiveGame:
    """The game in memory, which holds the Study roots its forks start from."""
    _cleanup_expired_games()
    live = LIVE_GAMES.get(trace_id)
    if live is None:
        raise HTTPException(status_code=409, detail="study_branch_unavailable")
    live.touch()
    return live


def _live_game_for_attempt(attempt_id: str) -> LiveGame:
    _cleanup_expired_games()
    matches = [
        live for live in LIVE_GAMES.values() if live.game.has_study_attempt(attempt_id)
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="study_attempt_not_found")
    matches[0].touch()
    return matches[0]


def _recording_read(trace_id: str) -> tuple["Recording", dict[str, Any]]:
    try:
        path = trace_store._trace_path(trace_id, trace_store.TRACES_DIR)
    except ValueError as exc:
        raise HTTPException(400, "recording_id_invalid") from exc
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    metadata = store.metadata(trace_id)
    stamp = (
        metadata.get("revision"),
        metadata.get("status"),
        metadata.get("ending_reason"),
    )
    if not metadata:
        try:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        except FileNotFoundError as exc:
            raise HTTPException(404, "recording_not_found") from exc
    try:
        recording = read_cache.load(trace_store.TRACES_DIR, trace_id, stamp)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, "recording_invalid") from exc
    return recording, metadata


def _recorded_mirrors(players: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Read search settings from retained seat identities, without a provider."""
    by_seat = {
        player["seat"]: (player.get("inference") or {})
        .get("runtime", {})
        .get("config", {})
        .get("mirror")
        for player in players
    }
    return [by_seat.get(seat) for seat in (0, 1)]


def _played_from_index(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Source game and starting turn for a game played on from another."""
    played_from = payload.get("played_from")
    if not played_from:
        return None
    event = played_from["event"]
    events = payload["events"]
    observation = (
        events[event]["observation"]
        if 0 <= event < len(events)
        else payload["final_observation"]
        if event == len(events)
        else None
    )
    turn = observation["turn"].get("turn_number") if observation else None
    return {**played_from, "turn": turn}


@app.get("/api/traces/{trace_id}/recording")
def get_recording_index(
    trace_id: str,
    offset: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 200:
        raise HTTPException(400, "invalid_page")
    _cleanup_expired_games()
    recording, metadata = _recording_read(trace_id)
    payload = recording.payload
    # Study forks and a watched game's next step need the game in memory.
    producer = LIVE_GAMES.get(trace_id)
    finished = payload.get("end_reason") == "game_over" and bool(
        payload["final_observation"].get("game_over")
    )
    available = bool(
        producer
        and not finished
        and producer.game.watch_mode
        and producer.game.watch_policies is not None
    )
    opponent = payload["config"].get("opponent") or {}
    # Seat 2's configured checkpoint identity and provenance, as the full trace
    # read redacts them: the checkpoint path stays private.
    provenance = payload["config"].get("opponent_checkpoint")
    if isinstance(provenance, dict):
        provenance = {k: v for k, v in provenance.items() if k != "checkpoint"}
    players = deepcopy(metadata.get("players", []))
    if payload["config"].get("decklists") == "closed":
        for player in players:
            if player["seat"] == 1 - payload["config"].get("human_seat", 0):
                player["deck"] = "closed"
    for player in players:
        deck = player.get("deck")
        if isinstance(deck, str):
            player["deck_name"] = (
                CLOSED_DECK_NAME
                if deck == "closed"
                else DECK_DISPLAY_NAMES.get(deck, deck)
            )
    return {
        "id": trace_id,
        "positions": [
            recording.metadata(position)
            for position in recording.positions[offset : offset + limit]
        ],
        "total": len(recording.positions),
        "next_offset": offset + limit
        if offset + limit < len(recording.positions)
        else None,
        "revision": metadata.get("revision"),
        "started_at": metadata.get("started_at"),
        "ended_at": metadata.get("ended_at"),
        # The stored status: a game dropped from memory is still in progress.
        "status": "completed"
        if finished
        else shown_status(str(metadata.get("status")), bool(metadata.get("resumable"))),
        "resumable": bool(metadata.get("resumable")),
        "can_produce": available,
        "mirror": _recorded_mirrors(players),
        "mirror_retry": producer.game.mirror_retry if available and producer else None,
        "can_study": producer is not None,
        "winner": payload.get("winner") if finished else None,
        "players": players,
        "record_stop": (payload.get("recording_import") or {}).get("stop"),
        # A game played on from another: its source and the turn it starts at.
        "played_from": _played_from_index(payload),
        "source_kind": "play"
        if recording.completed_mirror
        else opponent.get("source_kind"),
        "opponent": payload["config"].get("opponent"),
        "opponent_checkpoint": provenance,
        "watch": recording.watch,
        "track_history": recording.has_track_history(),
        **(
            {
                "history": [
                    event.model_dump(mode="json")
                    for event in recording.tracks.get(recording.human_seat, [])
                ]
            }
            if not recording.watch and offset == 0
            else {}
        ),
    }


@app.get("/api/traces/{trace_id}/positions/{key}")
def get_recording_position(
    trace_id: str,
    key: str,
    reveal: bool = False,
    study: bool = False,
    before: int | None = None,
) -> dict[str, Any]:
    recording, _ = _recording_read(trace_id)
    if key not in recording.by_key:
        raise HTTPException(404, "position_unavailable")
    if before is not None and before < 0:
        raise HTTPException(400, "invalid_history_cursor")
    try:
        position = recording.read(key, reveal=reveal, before=before)
        if study:
            position["study_source"] = recording.study_source(key)
        return position
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ———————————————— Analyses: a policy's reading of your decisions

# Games with an analysis running in this process; one per game at a time.
_ANALYSING: set[str] = set()
_ANALYSING_LOCK = threading.Lock()


def _analysed_attempt(trace_id: str) -> tuple[AttemptStore, RetainedAttempt]:
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    attempt = store.load_attempt(trace_id)
    if attempt is None:
        raise HTTPException(404, "recording_not_found")
    return store, attempt


@app.get("/api/traces/{trace_id}/analyses")
def get_analyses(trace_id: str) -> dict[str, Any]:
    """The game's saved analyses: a read of their rows only, cheap enough to
    follow a running analysis."""
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    if not store.has_attempt(trace_id):
        raise HTTPException(404, "recording_not_found")
    return {"analyses": store.analyses(trace_id)}


@app.get("/api/traces/{trace_id}/analyses/policies")
def get_analysis_policies(trace_id: str) -> dict[str, Any]:
    """Who may analyse this game, read once when Review opens: it re-executes
    the game and admits every registered policy."""
    _, attempt = _analysed_attempt(trace_id)
    try:
        choices = analysis_choices(attempt, policy_admissions())
    except (ImportError, OSError, ValueError) as exc:
        choices = AnalysisChoices([], None, f"Policies could not be listed: {exc}")
    return {
        "policies": choices.policies,
        "default": choices.default,
        "unavailable": choices.unavailable,
        "events": analysed_events(
            attempt.trace, attempt.trace["config"].get("human_seat", 0)
        ),
    }


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: str


@app.post("/api/traces/{trace_id}/analyses")
def start_analysis(trace_id: str, request: AnalysisRequest) -> dict[str, Any]:
    """Analyse the person's seat of a played game, finished or not: the steps
    stored when asked."""
    store, attempt = _analysed_attempt(trace_id)
    registry = policy_registry()
    try:
        descriptor = registry.descriptor(request.policy)
    except ValueError as exc:
        raise HTTPException(404, "analysis_policy_unavailable") from exc
    if descriptor.base_policy is not None:
        raise HTTPException(409, "analysis_by_search_refused")
    choices = analysis_choices(attempt, policy_admissions())
    if choices.unavailable or all(
        policy["sha256"] != descriptor.id for policy in choices.policies
    ):
        raise HTTPException(409, choices.unavailable or "analysis_policy_unavailable")
    config = GameConfig(**attempt.trace["config"])
    seat = config.human_seat
    with _ANALYSING_LOCK:
        if trace_id in _ANALYSING:
            raise HTTPException(409, "analysis_running")
        _ANALYSING.add(trace_id)
    runner: PolicyRunner | None = None
    try:
        try:
            runner = registry.load(
                descriptor.id, config.to_rust(), seat=seat, seed=int(config.seed or 0)
            )
        except (ImportError, OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        analysis = store.start_analysis(
            trace_id,
            seat,
            descriptor.record(),
            len(analysed_events(attempt.trace, seat)),
        )

        def run() -> None:
            try:
                run_analysis(store, analysis, attempt, seat, runner)
            finally:
                with _ANALYSING_LOCK:
                    _ANALYSING.discard(trace_id)

        threading.Thread(target=run, name=f"analysis-{analysis}", daemon=True).start()
    except BaseException:
        if runner is not None:
            runner.close()
        with _ANALYSING_LOCK:
            _ANALYSING.discard(trace_id)
        raise
    return {"id": analysis}


class LineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: int


@app.post("/api/traces/{trace_id}/analyses/{analysis_id}/line")
def see_line(trace_id: str, analysis_id: str, request: LineRequest) -> dict[str, Any]:
    """The analysing policy's line from one of its decisions (`agent_line`):
    its first choice played, then the policy and the game's opponent on to the
    end of the next turn. Computed on each request; nothing is saved."""
    store, attempt = _analysed_attempt(trace_id)
    saved = next((a for a in store.analyses(trace_id) if a["id"] == analysis_id), None)
    if saved is None:
        raise HTTPException(404, "analysis_not_found")
    evidence = next(
        (d for d in saved["decisions"] if d.get("event") == request.event), None
    )
    if evidence is None:
        raise HTTPException(404, "line_decision_not_analysed")
    config = GameConfig(**attempt.trace["config"])
    registry = policy_registry()
    seat = int(saved["seat"])
    runner: PolicyRunner | None = None
    opponent: BotPolicy | None = None
    try:
        runner = registry.load(
            saved["policy"]["id"],
            config.to_rust(),
            seat=seat,
            seed=int(config.seed or 0) + request.event,
        )
        opponent = build_villain_policy(config, seat=1 - seat)
    except (ImportError, OSError, ValueError, KeyError) as exc:
        close_policies((runner, opponent))
        raise HTTPException(409, "line_policy_unavailable") from exc
    try:
        line = agent_line(
            attempt, AnalysedDecision.model_validate(evidence), seat, runner, opponent
        )
    except AnalysisStop as stop:
        raise HTTPException(409, str(stop)) from None
    except (ValueError, managym.AgentError) as exc:
        # Native errors may name private cards: keep only the kind.
        raise HTTPException(409, f"The line failed: {type(exc).__name__}") from exc
    finally:
        close_policies((runner, opponent))
    return line.model_dump(mode="json")


@app.get("/api/traces/{trace_id}/decisions")
async def get_trace_decisions(trace_id: str) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    return projection_with_addresses(
        project_replay(
            replay, trace_store.load_trace(trace_id)["config"].get("human_seat", 0)
        )
    )


@app.get("/api/traces/{trace_id}/decisions/{address}")
async def get_trace_decision(trace_id: str, address: str) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    payload = trace_store.load_trace(trace_id)
    watch_mode = bool(payload["config"].get("watch_players"))
    try:
        viewer = (
            int(parse_decision_address(address).viewer)
            if watch_mode
            else payload["config"].get("human_seat", 0)
        )
        restored = restore_decision(replay, address, viewer)
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
) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    live = _live_game_for_trace(trace_id)
    try:
        return live.game.retry_study(
            trace_id=trace_id,
            replay=replay,
            raw_address=address,
            raw_command=payload.get("command"),
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
async def reveal_study_attempt(attempt_id: str) -> dict[str, Any]:
    live = _live_game_for_attempt(attempt_id)
    try:
        return live.game.reveal_study(attempt_id)
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
) -> dict[str, Any]:
    raw_plan = payload.get("plan")
    if raw_plan not in {"played", "policy", "search"}:
        raise HTTPException(status_code=400, detail="study_plan_invalid")
    live = _live_game_for_attempt(attempt_id)
    try:
        return live.game.preview_study_plan(attempt_id, raw_plan)
    except StudyEvidenceUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (StudyPlanUnavailableError, StudyCommandUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/study-attempts/{attempt_id}/return")
async def return_from_study_attempt(attempt_id: str) -> dict[str, Any]:
    live = _live_game_for_attempt(attempt_id)
    try:
        return live.game.return_from_study(attempt_id)
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
    payload.update(
        AttemptStore(trace_store.TRACES_DIR / "play.sqlite").metadata(trace_id)
    )
    return payload


@app.get("/api/advice")
async def get_advice_meta() -> dict[str, Any]:
    """Bootstrap for the shared decision-advice surface.

    Returns the compatibility bootstrap plus the checked advice-v1 comparison
    request consumed by both live play and Study.
    """
    try:
        return advice_meta().model_dump(mode="json")
    except AdvisorUnavailable as exc:
        raise HTTPException(404, "advice_unavailable") from exc


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


@app.post("/api/watch")
def create_bot_watch(payload: dict[str, Any]) -> dict[str, Any]:
    from .watch import WatchRequest, create_watch

    try:
        return create_watch(WatchRequest.model_validate(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/watch/setups/{handle}")
def start_watch_setup(handle: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .watch import WatchRequest, start_setup

    try:
        return start_setup(handle, WatchRequest.model_validate(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/watch/setups/{handle}")
def read_watch_setup(handle: str) -> dict[str, Any]:
    from .watch import setup_attempt

    return setup_attempt(handle).payload()


@app.post("/api/watch/{trace_id}/step")
async def advance_bot_watch(trace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .watch import live_watch, step_watch

    revision = payload.get("revision")
    if type(revision) is not int:
        raise HTTPException(400, "revision_required")
    live = live_watch(trace_id)
    async with live.lock:
        try:
            return await asyncio.to_thread(
                step_watch, trace_id, revision, payload.get("retry") is True
            )
        except SearchError as exc:
            raise HTTPException(409, {"code": exc.reason, "message": str(exc)}) from exc
        finally:
            live.game.work.finish(failed=live.game.mirror_retry is not None)


@app.get("/api/watch/{trace_id}/work")
def watch_work(trace_id: str) -> dict[str, Any]:
    from .watch import live_watch

    return {"work": live_watch(trace_id).game.work.payload()}


@app.get("/api/watch/{trace_id}/decisions/{ordinal}")
def inspect_bot_watch(
    trace_id: str,
    ordinal: int,
    reveal: bool = False,
) -> dict[str, Any]:
    from .watch import inspect_watch

    if ordinal < -1:
        raise HTTPException(400, "invalid_ordinal")
    try:
        return inspect_watch(trace_id, ordinal, reveal)
    except FileNotFoundError as exc:
        raise HTTPException(404, "watch_not_found") from exc
    except ValueError as exc:
        raise HTTPException(400, "watch_record_invalid") from exc


@app.post("/api/watch/import")
def import_game_watch(payload: dict[str, Any]) -> dict[str, Any]:
    from .watch import import_game

    try:
        return import_game(payload)
    except ValueError as exc:
        raise HTTPException(400, f"Game file rejected: {exc}") from exc
    except (KeyError, TypeError) as exc:
        raise HTTPException(400, "Game file rejected: invalid game fields.") from exc


@app.post("/api/decklists/validate")
def validate_decklists(payload: DeckImportInput) -> dict[str, Any]:
    try:
        payload.admit()
    except managym.DeckImportError as error:
        return {
            "admitted": False,
            "documents": {
                seat: decklist.to_dict() for seat, decklist in error.decklists.items()
            },
        }
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    world = managym.world_identity()
    return {
        "admitted": True,
        "ruleset": managym.DEVELOPMENT_RULESET,
        "card_digest": world["card_digest"],
    }


async def _previewed_game(game_id: str) -> LiveGame:
    """The game a preview reads, rebuilt from its record when it left memory
    (its prompt is then a new one, so the preview answers stale)."""
    live = LIVE_GAMES.get(game_id)
    if live is None:
        try:
            live = await take_up_game(game_id)
        except ValueError as exc:
            raise HTTPException(404, "game_not_found") from exc
    return live


class PaymentPreviewRequest(BaseModel):
    """One current offer, with every Academy authority coordinate."""

    model_config = ConfigDict(extra="forbid")
    match_id: str
    revision: int
    prompt_id: int
    frame_hash: str
    offer_id: int


@app.post("/api/games/{game_id}/payment-preview")
async def get_payment_preview(
    game_id: str,
    payload: PaymentPreviewRequest,
) -> dict[str, Any]:
    live = await _previewed_game(game_id)
    async with live.lock:
        game = live.game
        context = game.published_prompt
        pending = game._pending_decision
        if context is None or pending is None or game.env is None:
            raise HTTPException(409, "payment_preview_stale")
        frame = pending.frame
        # Only the person's own decision is previewed, never the opponent's.
        if context.viewer != game.human_seat:
            raise HTTPException(409, "payment_preview_stale")
        if (
            game.match_id != payload.match_id
            or frame.revision != payload.revision
            or frame.prompt is None
            or frame.prompt.id != payload.prompt_id
            or frame.frame_hash != payload.frame_hash
            or frame.frame_hash != context.frame["frame_hash"]
            or payload.offer_id not in context.action_by_offer
        ):
            raise HTTPException(409, "payment_preview_stale")
        native = SemanticDecisionFrame.from_json(
            pending.root.semantic_decision_frame_json()
        )
        try:
            preview = preview_payment(
                game.env,
                viewer=context.viewer,
                revision=native.revision,
                fingerprint=native.fingerprint,
                action_index=context.action_by_offer[payload.offer_id],
            )
        except managym.AgentError as exc:
            raise HTTPException(409, "payment_preview_stale") from exc
        return {
            **payload.model_dump(),
            "viewer": context.viewer,
            "preview": preview.model_dump(mode="json"),
        }


class CombatPreviewRequest(BaseModel):
    """A drafted declaration at the current combat prompt's authority address."""

    model_config = ConfigDict(extra="forbid")
    match_id: str
    revision: int
    prompt_id: int
    frame_hash: str
    plan: CombatPlan


@app.post("/api/games/{game_id}/combat-preview")
async def get_combat_preview(
    game_id: str,
    payload: CombatPreviewRequest,
) -> dict[str, Any]:
    """Preview a whole attack or block without committing any of it.

    Each step names the Academy offer to send at the decision it predicts:
    Academy offer ids enumerate the native actions in order
    (`_build_decision_context`), so a step's native action index is its offer.
    """
    live = await _previewed_game(game_id)
    async with live.lock:
        game = live.game
        context = game.published_prompt
        pending = game._pending_decision
        if context is None or pending is None or game.env is None:
            raise HTTPException(409, "combat_preview_stale")
        frame = pending.frame
        # Only the person's own decision is previewed, never the opponent's.
        if context.viewer != game.human_seat:
            raise HTTPException(409, "combat_preview_stale")
        if (
            game.match_id != payload.match_id
            or frame.revision != payload.revision
            or frame.prompt is None
            or frame.prompt.id != payload.prompt_id
            or frame.frame_hash != payload.frame_hash
            or frame.frame_hash != context.frame["frame_hash"]
        ):
            raise HTTPException(409, "combat_preview_stale")
        if context.action_space not in {"DECLARE_ATTACKER", "DECLARE_BLOCKER"}:
            raise HTTPException(409, "combat_preview_not_a_declaration")
        native = SemanticDecisionFrame.from_json(
            pending.root.semantic_decision_frame_json()
        )
        try:
            preview = preview_combat(
                game.env,
                viewer=context.viewer,
                revision=native.revision,
                fingerprint=native.fingerprint,
                plan=payload.plan,
            )
        except managym.AgentError as exc:
            raise HTTPException(409, "combat_preview_stale") from exc
        if preview.steps and (
            context.action_by_offer.get(preview.steps[0].action_index)
            != preview.steps[0].action_index
        ):
            raise RuntimeError("Academy combat offers no longer enumerate actions.")
        return {
            **payload.model_dump(mode="json"),
            "viewer": context.viewer,
            "preview": preview.model_dump(mode="json"),
        }
