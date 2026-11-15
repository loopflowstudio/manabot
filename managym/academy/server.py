"""
server.py
FastAPI server for interactive managym play over WebSocket.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from contextlib import ExitStack, suppress
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Callable, Literal, TypedDict
import warnings

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import TypeAdapter, ValidationError
from starlette.websockets import WebSocketState

import managym
from managym.academy.card_art import card_art
from managym.bots import BotDecision, BotPolicy, FlatSearchPolicy, choose_bot
from managym.decision import (
    Command as SemanticCommand,
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
    CommandReceipt,
    DeclarationCommand,
    DeclarationCommitment,
    DeclarationProjection,
    DeclarationReceipt,
    ExperienceFrame,
    InteractionOffer,
)
from managym.experience_projection import (
    HERO_PLAYER_INDEX,
    decision_surface_view,
    describe_actions,
    enum_name,
    object_records,
    observation_definition_id,
    observation_object_id,
    offer_verb,
    redact_observation,
    serialize_observation,
)
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
from .attempts import (
    AttemptStore,
    RecordedPlayer,
    decode_cursor,
    page_cursor,
    participant_id,
)
from .curated_pack import (
    CURATED_PACK,
    CURATED_PACK_CATALOG,
    ELEMENTAL_WORLD_PACK,
    JEONG_INCREMENT_PACK,
    curated_pack_for_matchup,
)
from .deck_import import DeckImportInput
from .opponent import (
    Opponent,
    catalog_opponent,
    opponent_catalog,
    policy_registry,
)
from .play_session import (
    SessionSetupUpdate,
    SessionStart,
    continuation_config,
    request_fingerprint,
    setup_draft,
)
from .presentation import PresentationProjector
from .recording import Recording, read_cache
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
    decision_source_sha256,
    load_canonical_replay,
    parse_decision_address,
    project_replay,
    projection_with_addresses,
    restore_decision,
)
from .shared_table_protocol import (
    REQUEST_ADAPTER,
    REQUEST_TYPES,
    ROLE_CAPABILITIES,
    ActionMessage,
    AuthorBeliefMessage,
    BranchPreviewMessage,
    BranchRevealMessage,
    CommandMessage,
    DeclarationCapacity,
    DeclarationMessage,
    DeclarationValidationEvent,
    JoinTableMessage,
    NewGameMessage,
    PassTurnMessage,
    RematchMessage,
    RestoreDecisionMessage,
    ResumeMessage,
    RetryDecisionMessage,
    RetryMirrorMessage,
    ReturnFromBranchMessage,
    ReturnToLiveMessage,
    SetStopsMessage,
    ShareBeliefMessage,
    TableCapability,
    TableSnapshot,
    TransferPilotMessage,
    ValidateDeclarationMessage,
    ViewerAccess,
    ViewerIdentity,
    ViewerRole,
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
from .trace import GameConfig, Trace, TraceEvent
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
    published: dict[str, dict[str, Any]] = field(default_factory=dict)

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
# Private, process-local configurations. Never included in viewer frames.
LAUNCH_CONFIGS: dict[str, dict[str, Any]] = {}


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
    source_sha256: str
    rules_revision: int


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
    obs: managym.Observation, viewer: int, *, include_details: bool = False
) -> dict[str, Any]:
    """Serialize an observation for one acting viewer.

    Two guarantees, regardless of whose perspective the engine observation is
    from:
      1. ``agent`` is always ``viewer`` and ``opponent`` the other player.
      2. The opponent's hand is redacted (libraries are never serialized).
    """
    data = serialize_observation(obs, include_details=include_details)
    if int(data["agent"]["player_index"]) != viewer:
        data["agent"], data["opponent"] = data["opponent"], data["agent"]
        data["won"] = bool(obs.game_over) and not bool(obs.won)

    redact_observation(data)
    return data


def hero_view(obs: managym.Observation) -> dict[str, Any]:
    """Backwards-compatible player-0 projection for the live table."""
    return viewer_view(obs, HERO_PLAYER_INDEX)


def _is_hero_turn(obs: managym.Observation, human_seat: int = 0) -> bool:
    return int(obs.agent.player_index) == human_seat


def _is_priority_space(obs: managym.Observation) -> bool:
    return obs.action_space.action_space_type == managym.ActionSpaceEnum.PRIORITY


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


def _parse_game_config(config: Any) -> GameConfig:
    data = config or {}
    if not isinstance(data, dict):
        raise ValueError("new_game.config must be an object.")

    villain_type = data.get("villain_type", "search")
    if villain_type == "checkpoint":
        warnings.warn(
            "checkpoint player names are deprecated; use registered policies (removed in 0.4.0)",
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
    provenance: dict[str, Any] = {}
    if data.get("deck_import") is not None:
        deck_import = DeckImportInput.model_validate(data["deck_import"])
        hero_import, villain_import = deck_import.admit()
        provenance = {
            "hero_source": hero_import.decklist.to_dict(),
            "villain_source": villain_import.decklist.to_dict(),
        }
        hero_deck, villain_deck = hero_import.deck.main, villain_import.deck.main
        hero_deck_name = villain_deck_name = "imported"
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
        asset_pack=selected_pack.reference if selected_pack is not None else None,
        villain_type=villain_type,
        human_seat=data.get("human_seat", 0),
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
            policy.observe(
                obs
                if int(obs.agent.player_index) == policy.seat
                else env.observation_for_player(policy.seat)
            )


def close_policies(policies: Iterable[BotPolicy | None]) -> None:
    with ExitStack() as resources:
        for policy in policies:
            if isinstance(policy, PolicyRunner):
                resources.callback(policy.close)


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
        self.interrupted_config: dict[str, Any] | None = None
        self.watch_policies: tuple[BotPolicy, BotPolicy] | None = None
        self._bot_decision: BotDecision | None = None
        self.mirror_retry: PendingBotDecision | None = None
        self._id_factory = id_factory or (lambda _kind: secrets.token_urlsafe(16))
        self._clock = clock or trace_store.utc_now_iso
        self.attempt_owner = participant_id(secrets.token_urlsafe(32))
        self._session_start: SessionStart | None = None
        self._preparing_demo = False
        self.attempt_store = AttemptStore(self.trace_dir / "play.sqlite")
        self.record_players = None
        self._record_player_owner = None
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
        self._pending_action_log: list[str] = []
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
        self._declaration_receipts: list[CommandReceipt] = []
        self.presentation = PresentationProjector()
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
        self._decision_source_sha256: dict[str, str] = {}
        self._decision_rules_revision: dict[str, int] = {}
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

    def new_game(
        self, raw_config: Any, *, continuation_fingerprint: str | None = None
    ) -> dict[str, Any]:
        launch_id = raw_config.get("launch") if isinstance(raw_config, dict) else None
        if launch_id is not None:
            if not isinstance(launch_id, str) or launch_id not in LAUNCH_CONFIGS:
                raise ValueError(
                    "launch_unknown: This launch is unknown; run the launch command again or choose a new game."
                )
            raw_config = {
                **raw_config,
                **LAUNCH_CONFIGS[launch_id],
                "request_id": f"launch.{launch_id}",
            }
        request_id = (
            raw_config.get("request_id") if isinstance(raw_config, dict) else None
        )
        if request_id is not None and (
            not isinstance(request_id, str) or len(request_id) > 128
        ):
            raise ValueError("Invalid new-game request id.")
        request_id = request_id or secrets.token_urlsafe(16)
        fingerprint = continuation_fingerprint or request_fingerprint(raw_config or {})
        if not self.watch_mode:
            previous_attempt = self.attempt_store.continuation_receipt(
                self.attempt_owner, request_id, fingerprint
            )
            if previous_attempt is not None:
                if previous_attempt == self.match_id and self.obs is not None:
                    return self.current_message()
                raise ValueError(
                    f"Request already started recording {previous_attempt}; it cannot be redealt. Open its recording or explicitly start another game."
                )

        if (
            isinstance(raw_config, dict)
            and raw_config.get("play_session_id") is not None
        ):
            retained_session = self.attempt_store.session(
                raw_config["play_session_id"], {self.attempt_owner}
            )
            if retained_session["revision"] != raw_config.get("setup_revision"):
                raise ValueError("Session setup changed; recover before continuing.")

        from .play_demo import load_play_demo, replay_checked_prefix

        demo_name = (raw_config if isinstance(raw_config, dict) else {}).get("demo")
        demo = None
        if demo_name is not None:
            demo = load_play_demo(demo_name)
            raw_config = {**raw_config, **demo.config}

        config = _parse_game_config(raw_config)
        seed = config.seed if config.seed is not None else secrets.randbits(32)
        config.seed = seed
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
            raise ValueError("Failed to load the selected opponent.") from exc

        with ExitStack() as prepared:
            if isinstance(policy, PolicyRunner):
                prepared.callback(policy.close)
            try:
                # Native skipping resolves sole legal actions, as in every runner;
                # priority-stop automation still records its Commands here.
                env = managym.Env(seed=seed)
                obs, _ = env.reset(config.to_rust())
                observe_policies((policy,), env, obs)
            except BaseException as exc:
                # PyO3's PanicException derives directly from BaseException and has
                # no importable Python class. A failed disposable replacement must
                # not close the existing authority; process interrupts still escape.
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                raise ValueError(
                    "Could not prepare the requested game. The current game is unchanged."
                ) from exc
            manifest = env.content_pack_manifest()
            selected_pack = curated_pack_for_matchup(
                config.hero_deck_name,
                config.villain_deck_name,
            )
            player_config = raw_config if isinstance(raw_config, dict) else {}
            players = self.attempt_store.players_for_game(
                self.attempt_owner,
                asdict(config),
                player_config.get("player_token"),
                player_config.get("player_name"),
            )
            if self.watch_mode and self.record_players is not None:
                players = self.record_players
            if (
                not self.watch_mode
                and not player_config.get("player_token")
                and self.record_players
                and self._record_player_owner == self.attempt_owner
            ):
                players[0] = replace(
                    next(
                        player
                        for player in self.record_players
                        if player.kind == "human"
                    ),
                    deck=config.hero_deck_name,
                    seat=config.human_seat,
                )
            if self.trace is not None:
                self.close(end_reason="new_game")
            self.record_players = players
            self._record_player_owner = self.attempt_owner
            self._close_policies()
            self.env, self.obs, self.villain_policy = env, obs, policy
            self.mirror_retry = None
            prepared.pop_all()
        self._demo_name = demo_name
        # A pack identifies both decks, so closed games use the generic identity.
        # Cards still resolve their treatments by name from their public zones.
        self.interrupted_config = None
        self.human_seat = config.human_seat
        self.decklists_closed = config.decklists == "closed"
        if selected_pack is not None and not self.decklists_closed:
            self.asset_manifest_hash = selected_pack.manifest_sha256
            self.content_hash = (
                UNVERSIONED_CONTENT_HASH
                if selected_pack is CURATED_PACK
                else str(manifest["content_digest"])
            )
        else:
            self.content_hash = UNVERSIONED_CONTENT_HASH
            self.asset_manifest_hash = ASSET_MANIFEST_HASH

        self.deck_names = {
            "hero": DECK_DISPLAY_NAMES.get(config.hero_deck_name, "Custom"),
            "villain": DECK_DISPLAY_NAMES.get(config.villain_deck_name, "Custom"),
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
            timestamp=self._clock(),
        )
        self._trace_saved = False
        self.trace_id = None
        self._pending_action_log = []
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
        self._declaration_receipts = []
        self.canonical_presentation = {0: [], 1: []}
        self._authority_command_seq = 0
        self._study_roots = {}
        self._pending_decision = None
        self._decision_source_sha256 = {}
        self._decision_rules_revision = {}
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
        self.trace_id = self.match_id
        if not self.watch_mode:
            session_id = player_config.get("play_session_id")
            session_revision = player_config.get("setup_revision", 0)
            self._session_start = SessionStart(
                session_id=session_id or secrets.token_urlsafe(16),
                owner=self.attempt_owner,
                request_id=request_id,
                fingerprint=fingerprint,
                revision=session_revision,
                draft=setup_draft(config),
            )
        self._persist_attempt()
        # A checked demo prefix is setup, not a sequence of surfaced human
        # decisions. Retain every Command and Study root in memory, then persist
        # its complete final snapshot once instead of rewriting the growing
        # replay at each internal setup step.
        self._preparing_demo = demo is not None
        try:
            self._advance()
            self._last_presentation_cursor = self.presentation.next_seq
            update = self._wire_message(reason="initial_connect")
            if demo is not None:
                replay_checked_prefix(self, update, demo, demo_name)
        finally:
            self._preparing_demo = False
        if demo is not None:
            update = self._wire_message(reason="initial_connect")
        if launch_id is not None:
            update["launch"] = launch_id
        return update

    def continue_game(self, request: RematchMessage, owner: str) -> dict[str, Any]:
        """Resolve one revision-bound next game from the owner-private draft."""
        if self.trace is None and self.interrupted_config is None:
            raise ValueError("No game to continue.")
        membership = self.attempt_store.session_for_attempt(self.match_id)
        if membership is None:
            raise ValueError("This recording has no play session.")
        session = self.attempt_store.session(membership["session_id"], {owner})
        # Legacy callers may omit continuation bindings. New clients always pin
        # all three; generate a single-use binding for the legacy operation.
        request = request.model_copy(
            update={
                "request_id": request.request_id or secrets.token_urlsafe(16),
                "attempt_id": request.attempt_id or self.match_id,
                "setup_revision": request.setup_revision
                if request.setup_revision is not None
                else session["revision"],
            }
        )
        assert request.request_id is not None
        receipt_config = {
            "request_id": request.request_id,
            "intent": request.intent,
            "attempt_id": request.attempt_id,
            "setup_revision": request.setup_revision,
            "overrides": request.config,
        }
        fingerprint = request_fingerprint(receipt_config)
        receipt = self.attempt_store.continuation_receipt(
            owner, request.request_id, fingerprint
        )
        if receipt is not None:
            if receipt == self.match_id:
                return self.current_message()
            raise ValueError(
                f"Continuation already started recording {receipt}; recover that recording."
            )
        if (
            request.attempt_id != self.match_id
            or request.setup_revision != session["revision"]
        ):
            raise ValueError("The game or setup changed. Recover before continuing.")
        overrides = dict(request.config)
        forbidden = {
            "play_session_id",
            "setup_revision",
            "request_id",
            "launch",
            "demo",
        }
        if forbidden.intersection(overrides):
            raise ValueError("Continuation setup contains reserved fields.")
        config = continuation_config(session["draft"], overrides, request.intent)
        config.update(
            play_session_id=membership["session_id"],
            setup_revision=session["revision"],
            request_id=request.request_id,
        )
        # new_game checks the exact receipt again. Retain the request envelope's
        # fingerprint so a retry can be checked before resolving against new setup.
        self.attempt_owner = owner
        return self.new_game(config, continuation_fingerprint=fingerprint)

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
        batch_cursor = self.presentation.next_seq
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
        batch_cursor = self.presentation.next_seq
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
            batch_cursor = self.presentation.next_seq
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
        return {
            "type": "command_outcome",
            "status": "accepted",
            "update": {
                "base_revision": base_revision,
                "frame": self._experience_frame(),
                "presentation": [
                    dict(event)
                    for event in self.presentation_events
                    if int(event["seq"]) >= cursor
                ],
                "receipt": deepcopy(receipt),
                **self._drain_surface_metadata(),
            },
        }

    def _declaration_precondition(self, command: DeclarationCommand) -> str | None:
        if not command.command_id:
            return "invalid_command"
        if command.command_id.startswith(("authority.", "compound-")):
            return "reserved_command_id"
        if command.match_id != self.match_id:
            return "wrong_match"
        context = self.published_prompt
        if (
            context is None
            or self.env is None
            or self.obs is None
            or self.trace is None
        ):
            return "authority_busy"
        if command.expected_revision != context.revision:
            return "stale_revision"
        if command.prompt_id != context.prompt_id:
            return "stale_prompt"
        if (
            self.watch_mode
            or self.obs.game_over
            or not _is_hero_turn(self.obs, self.human_seat)
        ):
            return "not_actor"
        raw = self.env.declaration_projection_json(self.human_seat)
        if json.loads(raw) is None:
            return "not_declaration"
        projection = DeclarationProjection.model_validate_json(raw)
        if projection.binding != command.plan.binding:
            return "stale_declaration"
        return None

    def validate_declaration(
        self, command: DeclarationCommand, request_id: str
    ) -> dict[str, Any]:
        """Read-only draft validation; it never publishes a prompt or saves a trace."""
        status: Literal["valid", "invalid", "stale", "capacity", "unavailable"] = (
            "valid"
        )
        message = "Declaration is legal."
        capacity: DeclarationCapacity | None = None
        try:
            reason = self._declaration_precondition(command)
            if reason is not None:
                status = (
                    "stale"
                    if reason
                    in {
                        "wrong_match",
                        "stale_revision",
                        "stale_prompt",
                        "stale_declaration",
                    }
                    else "unavailable"
                )
                message = reason.replace("_", " ").capitalize() + "."
            else:
                assert self.env is not None
                self.env.validate_declaration_plan_json(command.plan.model_dump_json())
        except managym.CapacityError as error:
            status = "capacity"
            message = (
                "Rules validation reached its capacity limit. Nothing was committed."
            )
            capacity = DeclarationCapacity(
                resource=error.resource, limit=error.limit, visited=error.visited
            )
        except managym.AgentError as error:
            status, message = "invalid", str(error)
        return DeclarationValidationEvent(
            type="declaration_validation",
            request_id=request_id,
            match_id=self.match_id,
            revision=self.revision,
            prompt_id=None
            if self.published_prompt is None
            else self.published_prompt.prompt_id,
            status=status,
            message=message,
            capacity=capacity,
        ).model_dump(mode="json")

    def _declaration_recording_candidate(
        self, prepared: managym.PreparedDeclaration
    ) -> GameSession:
        """Share immutable history, copy append owners, and retain actual native roots."""
        assert self.trace is not None
        staged = copy(self)
        staged.env = prepared.root(0)
        staged.trace = replace(self.trace, events=list(self.trace.events))
        staged.presentation = deepcopy(self.presentation)
        staged.presentation_events = list(self.presentation_events)
        staged.canonical_decisions = list(self.canonical_decisions)
        staged.canonical_presentation = {
            viewer: list(events)
            for viewer, events in self.canonical_presentation.items()
        }
        staged.authority_transitions = list(self.authority_transitions)
        staged._study_roots = dict(self._study_roots)
        staged._decision_source_sha256 = dict(self._decision_source_sha256)
        staged._decision_rules_revision = dict(self._decision_rules_revision)
        staged._pending_action_log = list(self._pending_action_log)
        staged.accepted_commands = dict(self.accepted_commands)
        staged._declaration_receipts = list(self._declaration_receipts)
        staged.authority_fallback_counters = dict(self.authority_fallback_counters)
        return staged

    def hero_declaration(
        self,
        raw: dict[str, Any],
        *,
        committed: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Commit a full plan under the table lock, then run opponent policies.

        The lock owns the source through candidate execution, recording and the
        durable save. Only private candidates are touched before that save; no
        policy is observed or called. The source therefore cannot change between
        persistence and native publication. Parent receipts name that commit's
        frame; later policy/automation steps can advance the response frame.
        """
        command_id = raw.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a non-empty string.")
        if command_id in self.accepted_commands:
            return {
                "type": "command_outcome",
                "status": "duplicate",
                "receipt": deepcopy(self.accepted_commands[command_id]),
                "recovery": self.current_recovery("duplicate_command"),
            }
        try:
            command = DeclarationCommand.model_validate(raw)
        except ValidationError:
            return self._reject_command(raw, "invalid_command", recover=False)
        try:
            reason = self._declaration_precondition(command)
            if reason is not None:
                return self._reject_command(
                    raw,
                    reason,
                    recover=reason
                    in {
                        "wrong_match",
                        "stale_revision",
                        "stale_prompt",
                        "stale_declaration",
                    },
                )
            assert self.env is not None and self.published_prompt is not None
            live_env = self.env
            prepared = live_env.prepare_declaration_plan_json(
                command.plan.model_dump_json()
            )
            evidence = json.loads(prepared.to_json())
            children = [
                SemanticCommand.from_payload(child) for child in evidence["commands"]
            ]
            completion = DeclarationCommitment.model_validate(evidence["completion"])
            staged = self._declaration_recording_candidate(prepared)
            batch_cursor = self.presentation.next_seq
            for index, child in enumerate(children):
                context = staged._publish_current_prompt()
                if context is None:
                    raise RuntimeError("Prepared declaration child has no prompt.")
                action_index = context.action_by_offer.get(child.offer_id)
                if action_index is None or child.answers or child.object_preconditions:
                    raise RuntimeError(
                        "Prepared declaration child is not an ordinary bound choice."
                    )
                staged._step_and_record(
                    actor="hero",
                    action_index=action_index,
                    actions=context.actions,
                    context=context,
                    source="client",
                    command=Command(
                        command_id=child.command_id,
                        match_id=self.match_id,
                        expected_revision=context.revision,
                        prompt_id=context.prompt_id,
                        offer_id=child.offer_id,
                        answers=[],
                    ).model_dump(mode="json"),
                    prepared_child=prepared.result(index),
                )
            if not children or staged.obs is None:
                raise RuntimeError("Prepared declaration has no committed children.")
            if (
                _is_hero_turn(staged.obs, staged.human_seat)
                and not staged.obs.game_over
            ):
                context = staged._publish_current_prompt()
                assert context is not None
                committed_frame = context.frame
            else:
                committed_frame = staged._build_frame(
                    staged.obs,
                    viewer=self.human_seat,
                    prompt=None,
                    offers=[],
                    action_space="",
                )
            receipt = CommandReceipt(
                command_id=command.command_id,
                actor=self.human_seat,
                accepted_at=self.revision,
                resulting_revision=staged.revision,
                resulting_frame_hash=committed_frame["frame_hash"],
                declaration=DeclarationReceipt(
                    command=command,
                    child_command_ids=[child.command_id for child in children],
                    completion=completion,
                ),
            )
            staged._declaration_receipts.append(receipt)
            staged.accepted_commands[command.command_id] = receipt.model_dump(
                mode="json"
            )
            while len(staged.accepted_commands) > MAX_ACCEPTED_COMMANDS:
                staged.accepted_commands.pop(next(iter(staged.accepted_commands)))
        except managym.CapacityError:
            rejected = self._reject_command(raw, "capacity", recover=False)
            rejected["rejection"]["message"] = (
                "Combat confirmation reached its capacity limit. Nothing was committed."
            )
            return rejected
        except managym.AgentError as error:
            rejected = self._reject_command(raw, "invalid_declaration", recover=False)
            rejected["rejection"]["message"] = str(error)
            return rejected
        # JSON/schema, retained roots, parent linkage and the durable row all
        # succeed before publication. SQLite failures leave the live game alone.
        # The table lock prevents a source change between this save and publish;
        # an ownership violation must not be mislabeled a correctable rules error.
        staged._persist_attempt(
            end_reason="game_over" if staged.obs.game_over else None
        )
        prepared.publish()
        # Publish append owners together. The original Env object and its capture
        # origin survive; only its successfully prepared native state was replaced.
        for name in (
            "obs",
            "trace",
            "revision",
            "_next_prompt_id",
            "published_prompt",
            "_pending_decision",
            "presentation",
            "presentation_events",
            "canonical_decisions",
            "canonical_presentation",
            "authority_transitions",
            "authority_fallback_counters",
            "_study_roots",
            "_decision_source_sha256",
            "_decision_rules_revision",
            "_pending_action_log",
            "_bot_decision",
            "accepted_commands",
            "_declaration_receipts",
            "_trace_saved",
            "_study_provider",
        ):
            setattr(self, name, getattr(staged, name))
        self.env = live_env
        if committed is not None:
            committed(
                self._accepted_outcome(
                    command.expected_revision,
                    batch_cursor,
                    receipt.model_dump(mode="json"),
                )
            )
            batch_cursor = self.presentation.next_seq
        response_base = (
            self.revision if committed is not None else command.expected_revision
        )
        try:
            self._observe_policies()
            self._advance()
        finally:
            if self.obs is not None and self.obs.game_over:
                self._close_policies()
        self._last_presentation_cursor = batch_cursor
        return self._accepted_outcome(
            response_base, batch_cursor, receipt.model_dump(mode="json")
        )

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

        membership = self.attempt_store.session_for_attempt(self.match_id)
        if membership is not None:
            session = self.attempt_store.session(
                membership["session_id"], {self.attempt_owner}
            )
            draft = {
                **session["draft"],
                "stops": self.stops,
                "stop_on_stack": self.stop_on_stack,
                "auto_pass": self.auto_pass,
            }
            self.attempt_store.update_session(
                membership["session_id"], self.attempt_owner, session["revision"], draft
            )

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

        if _is_hero_turn(self.obs, self.human_seat) and _is_priority_space(self.obs):
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
        actor = "hero" if context.viewer == self.human_seat else "villain"
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
        prepared_child: managym.DeclarationChild | None = None,
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
        event_subjects = json.loads(self.env.presentation_subjects_json())
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
                }
            )
            retained_root = (
                self.env if prepared_child is not None else self.env.clone_env()
            )
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
                if prepared_child is None:
                    retained_root = pending.root.clone_env()
                self._decision_source_sha256[committed_address.serialize()] = (
                    pending.source_sha256
                )
                self._decision_rules_revision[committed_address.serialize()] = (
                    pending.rules_revision
                )
            self._study_roots[int(row.ordinal)] = retained_root
            self.canonical_decisions.append(row)
        action_description = actions[action_index]["description"]
        self.presentation.note_action(InteractionOffer.model_validate(selected_offer))
        from_revision = self.revision
        semantic_frame = json.loads(self.env.semantic_decision_frame_json())
        if (
            int(semantic_frame["actor"]) != actor_index
            or len(semantic_frame["offers"]) != len(actions)
            or int(semantic_frame["offers"][action_index]["id"])
            != int(command["offer_id"])
        ):
            raise RuntimeError("Academy decision drifted from semantic authority.")
        semantic_command = SemanticCommand(
            command_id=str(command["command_id"]),
            expected_revision=int(semantic_frame["revision"]),
            offer_id=int(command["offer_id"]),
        )
        if prepared_child is None:
            transition_json, next_obs, reward, _, _, _ = self.env.step_semantic_command(
                semantic_command.to_json()
            )
        else:
            transition_json = prepared_child.transition_json
            next_obs = prepared_child.observation
            reward = prepared_child.reward
            self.env = prepared_child.env
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
        event_subjects.update(json.loads(self.env.presentation_subjects_json()))
        self.presentation.observe(next_obs, event_subjects=event_subjects)
        if auto:
            # Priority-stop automation stays identifiable in the hero's log.
            self._pending_action_log.append(f"Auto: {action_description}")
        elif actor == "villain":
            self._pending_action_log.append(f"Villain: {action_description}")
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
            )
        )
        self.obs = next_obs
        self._bot_decision = None
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
                    presentation_events=deepcopy(presentation_events),
                    encountered_definition_ids=encountered_definition_ids,
                )
            )

        if prepared_child is None:
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

            actions = describe_actions(self.obs, hero_seat=self.human_seat)
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
        if self.stop_on_stack and len(obs.stack_objects) > 0:
            return True

        step_name = enum_name(StepEnum, obs.turn.step)
        stop_step = ENGINE_STEP_TO_STOP_STEP.get(step_name)
        if stop_step is None:
            # Untap/cleanup/end-of-combat: no stop exists for these steps.
            return False
        side = "my" if _hero_is_active_player(obs, self.human_seat) else "opponent"
        return stop_step in self.stops.get(side, [])

    def _stops_payload(self) -> dict[str, Any]:
        return {
            "my": list(self.stops.get("my", [])),
            "opponent": list(self.stops.get("opponent", [])),
            "stop_on_stack": self.stop_on_stack,
            "auto_pass": self.auto_pass,
        }

    def _drain_pending_action_log(self) -> list[str]:
        pending = list(self._pending_action_log)
        self._pending_action_log = []
        return pending

    def _drain_surface_metadata(self) -> dict[str, Any]:
        """Drain compatibility narration outside the canonical frame."""
        payload: dict[str, Any] = {}
        log = self._drain_pending_action_log()
        auto_passed = self._auto_passed_since_surface
        self._auto_passed_since_surface = 0
        if log:
            payload["log"] = log
        if auto_passed:
            payload["auto_passed"] = auto_passed
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
    ) -> list[dict[str, Any]]:
        """Commit one engine step into both authorized semantic tracks."""
        authority_events = self.presentation.drain(
            from_revision=from_revision,
            to_revision=self.revision,
            caused_by=caused_by,
        )
        viewer_events: dict[int, list[dict[str, Any]]] = {}
        for viewer in (0, 1):
            projected: list[dict[str, Any]] = []
            for event in authority_events:
                copy = deepcopy(event)
                if viewer != actor:
                    copy["caused_by"] = None
                projected.append(copy)
            self.canonical_presentation[viewer].extend(projected)
            viewer_events[viewer] = projected

        live_events = viewer_events[self.human_seat]
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
            source_sha256=decision_source_sha256(address, frame),
            rules_revision=int(
                json.loads(self.env.semantic_observation_json(self.human_seat))[
                    "identity"
                ]["revision"]
            ),
        )
        return self.published_prompt

    def _build_decision_context(
        self,
        obs: managym.Observation,
        *,
        viewer: int,
    ) -> DecisionContext:
        """Build one exact actor-safe frame and raw-action lowering map."""
        actions = describe_actions(obs, hero_seat=self.human_seat)
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
        declaration = json.loads(self.env.declaration_projection_json(viewer))
        if declaration is not None:
            frame_without_hash["declaration"] = declaration
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

    def _persist_attempt(self, *, end_reason: str | None = None) -> None:
        if self._preparing_demo and end_reason is None:
            return
        if self.trace is None or self.obs is None or self.env is None:
            return
        finished = end_reason is not None
        if finished and self._trace_saved:
            return
        if end_reason == "game_over" and not self.obs.game_over:
            raise ValueError("Only an authoritative terminal game can be completed.")

        canonical = self._canonical_replay_snapshot()
        study_provider = (
            StudyForkProvider(canonical, self._study_roots) if finished else None
        )
        retained = replace(
            self.trace,
            final_observation=decision_surface_view(self.env, self.human_seat),
            winner=self.env.winner_index() if self.obs.game_over else None,
            end_reason=end_reason if finished else self.trace.end_reason,
            canonical_replay=canonical.model_dump(mode="json"),
        )
        if finished:
            self.trace = retained
        self.attempt_store.save(
            self.match_id,
            self.attempt_owner,
            self.revision,
            {
                **self.env.content_pack_manifest(),
                "rules_binary_sha256": RULES_BINARY_SHA256,
                "experience_source_sha256": PLAY_SOURCE_SHA256,
            },
            retained.to_record(),
            finished=finished,
            players=self.record_players,
            session_start=self._session_start,
            origin=(
                (os.getenv("ACADEMY_PLAY_RECORD_ORIGIN") or "bot_evaluation")
                if self.watch_mode
                else "automated_validation"
                if self._demo_name
                else None
            ),
        )
        self._session_start = None
        if finished:
            self._study_provider = study_provider
            self._trace_saved = True

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
            command_receipts=list(self._declaration_receipts),
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

        event_subjects = branch.presentation_subjects()
        projector = PresentationProjector()
        projector.note_action(canonical_offer)
        try:
            observation = branch.submit_semantic(command)
        except (managym.AgentError, StudyBranchUnavailableError) as exc:
            raise StudyCommandUnavailableError("study_command_rejected") from exc
        event_subjects.update(branch.presentation_subjects())
        projector.observe(observation, event_subjects=event_subjects)
        presentation = projector.drain(
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
        close_policies((self.villain_policy, *(self.watch_policies or ())))

    def close(self, end_reason: str) -> None:
        try:
            self._close_study_attempts()
            self._persist_attempt(end_reason=end_reason)
        finally:
            try:
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
        if record.expires_at <= now and not record.lock.locked()
    ]
    for session_id in expired_ids:
        _drop_session(session_id, end_reason=SESSION_EXPIRED_END_REASON)


def _participant_from_resume(
    raw_session_id: Any,
    raw_resume_token: Any,
    play_session_id: str | None = None,
) -> tuple[SessionRecord, ParticipantLease]:
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise ValueError("resume messages require a non-empty 'session_id'.")
    if not isinstance(raw_resume_token, str) or not raw_resume_token:
        raise ValueError("resume messages require a non-empty 'resume_token'.")

    record = SESSION_REGISTRY.get(raw_session_id)
    if record is None:
        if play_session_id is None:
            raise ValueError("Session not found or expired. Start a new game.")
        game = _new_game_session()
        owner = participant_id(raw_resume_token)
        game.attempt_store.session(play_session_id, {owner})
        attempt = game.attempt_store.latest_session_attempt(play_session_id)
        retained = game.attempt_store.load(attempt)
        if retained is None or retained["end_reason"] == "active":
            raise ValueError(
                "Original table is unavailable; an active attempt cannot be recreated."
            )
        # Recover only private setup and recorded identity. No Env, prompt or
        # live Command authority can survive a process restart.
        game.attempt_owner = owner
        game.match_id = game.trace_id = attempt
        game.record_players = [
            RecordedPlayer(**player)
            for player in game.attempt_store.metadata(attempt)["players"]
        ]
        game._record_player_owner = owner
        game.interrupted_config = retained["config"]
        game.human_seat = retained["config"].get("human_seat", 0)
        record = SessionRecord(
            session_id=raw_session_id, resume_token=raw_resume_token, game=game
        )
        SESSION_REGISTRY[raw_session_id] = record
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
    committed = [
        {
            "address": DecisionAddressV2.from_row(
                replay_id=f"replay.{record.game.match_id}",
                match_id=record.game.match_id,
                row=decision,
            ).serialize(),
            "ordinal": decision.ordinal,
            "revision": decision.revision,
            "prompt_id": decision.prompt_id,
            "offer_id": decision.offer_id,
        }
        for decision in record.game.canonical_decisions
        if decision.viewer == record.game.human_seat
    ]
    pending = record.game._pending_decision
    if pending is None:
        return committed
    address = pending.address
    return [
        *committed,
        {
            "address": address.serialize(),
            "ordinal": address.ordinal,
            "revision": address.revision,
            "prompt_id": address.prompt_id,
            "offer_id": None,
        },
    ]


def _table_opponent_label(record: SessionRecord) -> str | None:
    trace = record.game.trace
    if trace is None:
        return None
    config = trace.config
    if config.villain_type == "search":
        return f"Search {config.villain_sims}"
    if config.opponent is not None:
        return f"{config.opponent['name']} · {config.opponent['sha256'][:10]}"
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
                rules_viewer=record.game.human_seat,
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
        attempt_id=record.game.trace_id,
        opponent=record.game.trace.config.opponent if record.game.trace else None,
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
    payload["work"] = record.game.work.payload()
    payload["mirror_retry"] = record.game.mirror_retry
    payload["mirror"] = (
        asdict(record.game.trace.config).get("mirror") if record.game.trace else None
    )
    # Launch metadata belongs only to the browser that submitted the setup.
    payload.pop("launch", None)
    payload.pop("play_session", None)
    membership = record.game.attempt_store.session_for_attempt(record.game.match_id)
    if membership is not None:
        try:
            session = record.game.attempt_store.session(
                membership["session_id"], {participant_id(participant.resume_token)}
            )
            payload["play_session"] = {
                **session,
                **membership,
                "config": asdict(record.game.trace.config)
                if record.game.trace
                else dict(record.game.interrupted_config or {}),
            }
            payload["play_session"]["config"]["seed"] = str(
                payload["play_session"]["config"].get("seed")
            )
        except ValueError:
            pass
    payload["session_interrupted"] = record.game.interrupted_config is not None
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
    "declaration": TableCapability.SUBMIT_LIVE_COMMAND,
    "validate_declaration": TableCapability.SUBMIT_LIVE_COMMAND,
    "action": TableCapability.SUBMIT_LIVE_COMMAND,
    "pass_turn": TableCapability.SUBMIT_LIVE_COMMAND,
    "retry_mirror": TableCapability.SUBMIT_LIVE_COMMAND,
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
    raise RuntimeError("shared-table request vocabulary and dispatch drifted")


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
    try:
        meta = advice_meta().model_dump(mode="json")
    except AdvisorUnavailable as exc:
        raise TableDispatchError(
            "not_found", "Advice fixtures are unavailable."
        ) from exc
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
    committed: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Authorize and execute one closed operation under the table lock."""
    if isinstance(request, (CommandMessage, DeclarationMessage)):
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
        owner = participant_id(participant.resume_token)
        previous_match = record.game.match_id
        if isinstance(request, RematchMessage):
            response = record.game.continue_game(request, owner)
        else:
            if {"play_session_id", "setup_revision"}.intersection(request.config):
                raise ValueError("Use continuation to join a session.")
            record.game.attempt_owner = owner
            response = record.game.new_game(dict(request.config))
        if record.game.match_id == previous_match:
            return response, False
        record.accepted_command_submitters.clear()
        record.shared_beliefs.clear()
        for candidate in record.participants.values():
            candidate.personal_beliefs.clear()
            candidate.study_attempts.clear()
        record.table_revision += 1
        return response, True
    if isinstance(request, (CommandMessage, DeclarationMessage)):
        command_id = request.command.command_id
        record.accepted_command_submitters[command_id] = participant.viewer_id
        try:
            apply = (
                record.game.hero_declaration
                if isinstance(request, DeclarationMessage)
                else record.game.hero_command
            )
            response = apply(
                request.command.model_dump(mode="json"), committed=committed
            )
        finally:
            # Keep retry ownership if policy/transport work failed after commit.
            if command_id not in record.game.accepted_commands:
                record.accepted_command_submitters.pop(command_id, None)
        return response, True
    if isinstance(request, ValidateDeclarationMessage):
        return record.game.validate_declaration(
            request.command, request.request_id
        ), False
    if isinstance(request, ActionMessage):
        return record.game.hero_action(request.index), True
    if isinstance(request, PassTurnMessage):
        return record.game.pass_turn(), True
    if isinstance(request, RetryMirrorMessage):
        return record.game.retry_mirror(
            request.match_id, request.expected_revision
        ), True
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
            restored = restore_decision(replay, request.address, record.game.human_seat)
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


def _launch_metadata(launch_id: object) -> dict[str, Any]:
    """Return setup to the requesting tab even when a known launch is refused."""
    if not isinstance(launch_id, str) or launch_id not in LAUNCH_CONFIGS:
        return {}
    return {
        "launch": launch_id,
        "launch_setup": {
            key: value
            for key, value in LAUNCH_CONFIGS[launch_id].items()
            if key != "seed"
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


def _cache_table_recovery(record: SessionRecord) -> None:
    # Project once at a safe worker boundary, then apply each viewer's access.
    response: dict[str, Any]
    if record.game.obs is None:
        response = {"type": "table_snapshot"}
    else:
        recovery = record.game.current_recovery("reconnect")
        frame = recovery["frame"]
        response = {
            "type": "game_over" if frame["projection"]["game_over"] else "observation",
            "data": frame["projection"],
            "actions": [],
            "winner": frame["winner"],
            "frame": frame,
            "recovery": recovery,
        }
    for participant in list(record.participants.values()):
        record.published[participant.viewer_id] = _response_with_participant(
            response, record, participant
        )


async def _publish_work(record: SessionRecord, status: WorkStatus) -> None:
    # Status has no projection path. Resume uses the last permitted publication.
    for participant in list(record.participants.values()):
        if participant.websocket is not None:
            with suppress(Exception):
                await participant.websocket.send_json(
                    {
                        "type": "work_status",
                        "session_id": record.session_id,
                        "work": asdict(status),
                    }
                )


async def _send_table_response(
    record: SessionRecord,
    initiator: ParticipantLease,
    response: dict[str, Any],
    *,
    broadcast: bool,
    watcher_invite: str | None = None,
) -> None:
    _cache_table_recovery(record)
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
        if recipient is initiator and response.get("launch") in LAUNCH_CONFIGS:
            payload.update(_launch_metadata(response["launch"]))
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
                if isinstance(parsed, (JoinTableMessage, ResumeMessage)):
                    presentation_cursor = _parse_presentation_cursor(
                        parsed.presentation_cursor
                    )
                    if isinstance(parsed, JoinTableMessage):
                        record, participant = _participant_from_invite(
                            parsed.table_id, parsed.invite_token
                        )
                    else:
                        record, participant = _participant_from_resume(
                            parsed.session_id,
                            parsed.resume_token,
                            parsed.play_session_id,
                        )
                    await _attach_participant_websocket(record, participant, websocket)
                    attached_table_id = record.session_id
                    attached_viewer_id = participant.viewer_id
                    if record.lock.locked():
                        cached = record.published.get(participant.viewer_id)
                        if cached is not None:
                            await websocket.send_json(
                                {
                                    **cached,
                                    "work": record.game.work.payload(),
                                }
                            )
                            continue
                    async with record.lock:
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
                    loop = asyncio.get_running_loop()

                    def committed(outcome: dict[str, Any]) -> None:
                        # Projection and delivery finish at the safe committed
                        # boundary before the worker may mutate the Env again.
                        asyncio.run_coroutine_threadsafe(
                            _send_table_response(
                                record, participant, outcome, broadcast=True
                            ),
                            loop,
                        ).result()

                    def publish_work(status: WorkStatus) -> None:
                        # Called while the worker is paused at a safe boundary.
                        asyncio.run_coroutine_threadsafe(
                            _publish_work(record, status), loop
                        ).result()

                    # Draft validation reads the same authority under this lock.
                    # Marking it as advancement disables its caller and restarts
                    # reactive validation before the reply can be applied.
                    reports_work = not isinstance(parsed, ValidateDeclarationMessage)
                    if reports_work:
                        record.game.work.stage(
                            "advance",
                            record.game.match_id
                            if record.game.obs is not None
                            else None,
                            record.game.revision
                            if record.game.obs is not None
                            else None,
                        )
                        status = record.game.work.read()
                        assert status is not None
                        _cache_table_recovery(record)
                        await _publish_work(record, status)
                        record.game.work.publish = publish_work
                    failed = True
                    try:
                        response, broadcast = await asyncio.to_thread(
                            _dispatch_table_request,
                            record,
                            participant,
                            parsed,
                            committed,
                        )
                        failed = record.game.mirror_retry is not None
                    finally:
                        if reports_work:
                            record.game.work.publish = None
                            record.game.work.finish(failed=failed)
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
                await websocket.send_json(_request_error(request, str(exc)))
                continue
    except WebSocketDisconnect:
        return
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, sqlite3.Error):
            message = "Game recording failed; this game may not have been saved."
        error = _request_error(request, message)
        if attached_table_id is not None:
            # The client must neither resume nor relaunch an ended game.
            error["session_ended"] = True
            try:
                _drop_session(attached_table_id, end_reason="error")
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


@app.get("/api/card-art/back.jpg")
def card_back() -> Response:
    return _art_response(card_art().back())


@app.get("/api/card-art/{kind}/{slug}.jpg")
def card_image(kind: Literal["art", "full"], slug: str) -> Response:
    return _art_response(card_art().image(slug, kind))


@app.get("/api/opponents")
async def get_opponents() -> dict[str, Any]:
    return await asyncio.to_thread(opponent_catalog)


@app.post("/api/policies/check")
def check_policy_setup(config: dict[str, Any]) -> dict[str, str | bool | None]:
    """Preview admission without loading a session; Start repeats the same checks."""
    try:
        _parse_game_config(config)
    except (ValueError, OSError, ImportError) as exc:
        return {"compatible": False, "reason": str(exc)}
    return {"compatible": True, "reason": None}


def _record_owners(tokens: str | None) -> set[str]:
    return {participant_id(token) for token in (tokens or "").split(",") if token}


@app.middleware("http")
async def authorize_record(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    parts = request.url.path.split("/")
    if len(parts) >= 4 and parts[1:3] in (["api", "traces"], ["api", "watch"]):
        store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
        access = store.access(
            parts[3],
            _record_owners(
                ",".join(
                    filter(
                        None,
                        [
                            request.headers.get("x-etude-participant-tokens"),
                            request.headers.get("x-etude-participant-token"),
                        ],
                    )
                )
            ),
            request.headers.get("x-etude-player-token"),
        )
        request.state.record_access = access
        # Each reader applies the recording's permitted information projection.
        # Feedback and every mutation remain participant-only.
        public_read = request.method == "GET" and (
            len(parts) == 4 or parts[4] in {"decisions", "recording", "positions"}
        )
        if access is not None and not (
            access["mine"] or (public_read and access["replay"])
        ):
            return JSONResponse(
                {"detail": "participant_credential_required"}, status_code=403
            )
    return await call_next(request)


@app.get("/api/player")
async def get_player(x_etude_player_token: str = Header()) -> dict:
    try:
        return AttemptStore(trace_store.TRACES_DIR / "play.sqlite").profile(
            x_etude_player_token
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/player")
async def name_player(
    payload: dict[str, Any], x_etude_player_token: str = Header()
) -> dict:
    try:
        return AttemptStore(trace_store.TRACES_DIR / "play.sqlite").profile(
            x_etude_player_token, payload.get("name")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/traces/{trace_id}/feedback")
async def record_feedback(
    trace_id: str,
    payload: dict[str, Any],
    x_etude_player_token: str | None = Header(default=None),
) -> dict[str, bool]:
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    if store.owner(trace_id) is None:
        raise HTTPException(status_code=404, detail="attempt_not_found")
    try:
        request_id, note = payload.get("request_id"), payload.get("note")
        if not isinstance(request_id, str) or not isinstance(note, str):
            raise ValueError("Feedback requires a request id and note.")
        address = payload.get("decision_address")
        if address is not None:
            if not isinstance(address, str):
                raise ValueError("Invalid decision address.")
            restore_decision(
                _load_trace_replay(trace_id),
                address,
                trace_store.load_trace(trace_id)["config"].get("human_seat", 0),
            )
        author = (
            store.profile(x_etude_player_token)["id"] if x_etude_player_token else None
        )
        store.feedback(
            trace_id, request_id, note, author_id=author, decision_address=address
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True}


@app.put("/api/play-sessions/{session_id}")
async def update_play_session(
    session_id: str,
    update: SessionSetupUpdate,
    x_etude_participant_token: str = Header(),
) -> dict[str, Any]:
    try:
        record, participant = _participant_from_resume(
            update.table_id, x_etude_participant_token
        )
        async with record.lock:
            _authorize_table_request(
                record,
                participant,
                NewGameMessage(type="new_game", grant_revision=update.grant_revision),
            )
            owner = participant_id(participant.resume_token)
            session = record.game.attempt_store.session(session_id, {owner})
            membership = record.game.attempt_store.session_for_attempt(
                record.game.match_id
            )
            if membership is None or membership["session_id"] != session_id:
                raise ValueError("Setup belongs to another table.")
            if update.revision != session["revision"]:
                raise ValueError("Session setup changed; recover before editing.")
            parsed = _parse_game_config(
                continuation_config(session["draft"], update.config, "rematch")
            )
            record.game.attempt_store.update_session(
                session_id, owner, update.revision, setup_draft(parsed)
            )
            response = _response_with_participant(
                {"type": "table_snapshot"}, record, participant
            )
            return response["play_session"]
    except (ValueError, TableDispatchError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/play-sessions/{session_id}")
def get_play_session(
    session_id: str,
    result_offset: int = Query(default=0, ge=0),
    x_etude_participant_tokens: str | None = Header(default=None),
) -> dict[str, Any]:
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    owners = _record_owners(x_etude_participant_tokens)
    try:
        rows = store.session_results(session_id, owners, offset=result_offset)
        return {
            **store.session(session_id, owners),
            "results": rows[:100],
            "next_result_offset": result_offset + 100 if len(rows) > 100 else None,
            "totals": store.session_totals(session_id, owners),
        }
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


@app.get("/api/traces")
async def list_traces(
    response: Response,
    x_etude_participant_tokens: str | None = Header(default=None),
    x_etude_player_token: str | None = Header(default=None),
    session: str | None = None,
    scope: str = "all",
    q: str = "",
    player: str | None = None,
    other: str | None = None,
    pairing: str | None = None,
    version: str | None = None,
    deck: str | None = None,
    status: str | None = None,
    result: str | None = None,
    after: str | None = None,
    before: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict[str, Any]]:
    store = AttemptStore(trace_store.TRACES_DIR / "play.sqlite")
    try:
        # One extra row tells the client whether another page exists.
        rows = store.summaries(
            _record_owners(x_etude_participant_tokens),
            token=x_etude_player_token,
            scope=scope,
            session=session,
            q=q,
            player=player,
            other=other,
            pairing=pairing,
            version=version,
            deck=deck,
            status=status,
            result=result,
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
    if (
        scope == "all"
        and not session
        and not any((player, other, pairing, version, deck, result))
    ):
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
                    "mine": False,
                    "replay_available": True,
                    "feedback": [],
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
    """Authorize one exact participant, including single-player tables."""
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


@app.get("/api/traces/{trace_id}/recording")
def get_recording_index(
    trace_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 200,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 200:
        raise HTTPException(400, "invalid_page")
    _cleanup_expired_sessions()
    recording, metadata = _recording_read(trace_id)
    payload = recording.payload
    producer = next(
        (
            record
            for record in SESSION_REGISTRY.values()
            if record.game.trace_id == trace_id
        ),
        None,
    )
    finished = payload.get("end_reason") == "game_over" and bool(
        payload["final_observation"].get("game_over")
    )
    study_available = False
    if producer:
        try:
            _rest_study_owner(producer, x_etude_participant_token)
            study_available = True
        except HTTPException:
            pass
    available = bool(
        study_available
        and producer
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
        "status": "finished"
        if finished
        else "waiting"
        if producer and payload.get("end_reason") == "active"
        else "interrupted",
        "can_produce": available,
        "mirror": _recorded_mirrors(players),
        "mirror_retry": producer.game.mirror_retry if available and producer else None,
        "can_study": study_available,
        "winner": payload.get("winner") if finished else None,
        "players": players,
        "source_kind": "play"
        if recording.completed_mirror
        else opponent.get("source_kind"),
        "opponent": payload["config"].get("opponent"),
        "opponent_checkpoint": provenance,
        "watch": recording.watch,
        "private_setup": {
            **setup_draft(GameConfig(**payload["config"])),
            "watch_players": payload["config"].get("watch_players"),
            "watch_mirrors": _recorded_mirrors(players),
        }
        if recording.watch
        and (getattr(request.state, "record_access", None) or {}).get("mine")
        else None,
        "can_feedback": bool(
            (getattr(request.state, "record_access", None) or {}).get("mine")
        ),
    }


@app.get("/api/traces/{trace_id}/positions/{key}")
def get_recording_position(
    trace_id: str,
    key: str,
    reveal: bool = False,
    study: bool = False,
    before: int | None = None,
    x_etude_participant_token: str | None = Header(default=None),
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


@app.get("/api/traces/{trace_id}/decisions")
async def get_trace_decisions(trace_id: str) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    return projection_with_addresses(
        project_replay(
            replay, trace_store.load_trace(trace_id)["config"].get("human_seat", 0)
        )
    )


@app.get("/api/traces/{trace_id}/decisions/{address}")
async def get_trace_decision(
    trace_id: str,
    address: str,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    replay = _load_trace_replay(trace_id)
    payload = trace_store.load_trace(trace_id)
    watch_mode = bool(payload["config"].get("watch_players"))
    if watch_mode and not (
        payload["end_reason"] == "game_over"
        and payload["final_observation"]["game_over"]
    ):
        _rest_study_owner(_live_record_for_trace(trace_id), x_etude_participant_token)
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
async def get_trace(
    trace_id: str, request: Request, reveal_hidden: bool = False
) -> dict[str, Any]:
    try:
        payload = trace_store.load_trace(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = trace_store.prepare_trace_payload(payload, reveal_hidden=reveal_hidden)
    payload["id"] = trace_id
    access = getattr(request.state, "record_access", None)
    payload["can_feedback"] = bool(access and access["mine"])
    payload["read_only"] = bool(access and not access["mine"])
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
def start_watch_setup(
    handle: str,
    payload: dict[str, Any],
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    from .watch import WatchRequest, start_setup

    try:
        return start_setup(
            handle, WatchRequest.model_validate(payload), x_etude_participant_token
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/watch/setups/{handle}")
def read_watch_setup(
    handle: str,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    from .watch import setup_attempt

    return setup_attempt(handle, x_etude_participant_token).payload()


@app.post("/api/watch/{trace_id}/step")
async def advance_bot_watch(
    trace_id: str,
    payload: dict[str, Any],
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    from .watch import record_for_watch, step_watch

    revision = payload.get("revision")
    if type(revision) is not int:
        raise HTTPException(400, "revision_required")
    record = record_for_watch(trace_id, x_etude_participant_token)
    async with record.lock:
        try:
            return await asyncio.to_thread(
                step_watch,
                trace_id,
                x_etude_participant_token,
                revision,
                payload.get("retry") is True,
            )
        except SearchError as exc:
            raise HTTPException(409, {"code": exc.reason, "message": str(exc)}) from exc
        finally:
            record.game.work.finish(failed=record.game.mirror_retry is not None)


@app.get("/api/watch/{trace_id}/work")
def watch_work(
    trace_id: str,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    from .watch import record_for_watch

    record = record_for_watch(trace_id, x_etude_participant_token)
    return {"work": record.game.work.payload()}


@app.get("/api/watch/{trace_id}/decisions/{ordinal}")
def inspect_bot_watch(
    trace_id: str,
    ordinal: int,
    reveal: bool = False,
    x_etude_participant_token: str | None = Header(default=None),
) -> dict[str, Any]:
    from .watch import inspect_watch

    if ordinal < -1:
        raise HTTPException(400, "invalid_ordinal")
    try:
        return inspect_watch(trace_id, ordinal, x_etude_participant_token, reveal)
    except FileNotFoundError as exc:
        raise HTTPException(404, "watch_not_found") from exc
    except ValueError as exc:
        raise HTTPException(400, "watch_record_invalid") from exc


@app.post("/api/watch/import")
def import_arena_watch(payload: dict[str, Any]) -> dict[str, Any]:
    from .watch import import_arena

    try:
        return import_arena(payload)
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
        "world_version": world["world_version"],
        "card_digest": world["card_digest"],
    }
