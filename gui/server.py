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
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

# Local imports
from manabot.env.observation import ActionEnum, PhaseEnum, StepEnum, ZoneEnum
import managym

from . import trace as trace_store
from .trace import GameConfig, Trace, TraceEvent
from .villain import build_villain_policy

DEFAULT_DECK = {
    "Mountain": 12,
    "Forest": 12,
    "Llanowar Elves": 18,
    "Grey Ogre": 18,
}

MAX_AUTOPLAY_STEPS = 1024
HERO_PLAYER_INDEX = 0
ACTION_LABELS = {
    "PRIORITY_PLAY_LAND": "Play land",
    "PRIORITY_CAST_SPELL": "Cast spell",
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
    last_seen_at: datetime = field(default_factory=_now_utc)
    expires_at: datetime = field(default_factory=lambda: _now_utc() + SESSION_TTL)

    def touch(self) -> None:
        now = _now_utc()
        self.last_seen_at = now
        self.expires_at = now + SESSION_TTL


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
    cards_by_id = {int(card.id): card for card in cards}

    grouped_cards: dict[str, list[dict[str, Any]]] = {
        "HAND": [],
        "GRAVEYARD": [],
        "EXILE": [],
        "STACK": [],
    }
    for card in cards:
        zone_name = _enum_name(ZoneEnum, card.zone)
        if zone_name in grouped_cards:
            grouped_cards[zone_name].append(_serialize_card(card))

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
        "battlefield": [
            _serialize_permanent(permanent, cards_by_id.get(int(permanent.id)))
            for permanent in permanents
        ],
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


def _build_id_to_name(obs: managym.Observation) -> dict[int, str]:
    names: dict[int, str] = {
        int(obs.agent.id): "agent",
        int(obs.opponent.id): "opponent",
    }
    names.update(
        {int(card.id): card.name for card in [*obs.agent_cards, *obs.opponent_cards]}
    )
    return names


def _format_action(
    action: managym.Action, card_name: str | None, names: dict[int, str]
) -> str:
    action_name = _enum_name(ActionEnum, action.action_type)

    if action_name == "PRIORITY_PASS_PRIORITY":
        return "Pass priority"
    label = ACTION_LABELS.get(action_name, action_name)

    subject = card_name
    if subject is None:
        subject = next(
            (
                name
                for value in action.focus
                if (name := names.get(int(value))) is not None
            ),
            None,
        )

    if subject is None:
        return label
    return f"{label}: {subject}"


def describe_actions(obs: managym.Observation) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    names = _build_id_to_name(obs)
    for index, action in enumerate(obs.action_space.actions):
        focus = [int(value) for value in action.focus]
        card_name = names.get(focus[0]) if focus else None
        results.append(
            {
                "index": index,
                "type": _enum_name(ActionEnum, action.action_type),
                "card": card_name,
                "focus": focus,
                "description": _format_action(action, card_name, names),
            }
        )
    return results


def _winner_for_hero(obs: managym.Observation) -> int | None:
    if not obs.game_over:
        return None

    agent_is_hero = int(obs.agent.player_index) == HERO_PLAYER_INDEX
    if bool(obs.won):
        return 0 if agent_is_hero else 1
    return 1 if agent_is_hero else 0


def _is_hero_turn(obs: managym.Observation) -> bool:
    return int(obs.agent.player_index) == HERO_PLAYER_INDEX


def _normalize_deck(value: Any, fallback: dict[str, int]) -> dict[str, int]:
    if value is None:
        return dict(fallback)
    if not isinstance(value, dict):
        raise ValueError("Deck config must be an object of {card_name: count}.")

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

    return deck


def _parse_game_config(config: Any) -> GameConfig:
    data = config or {}
    if not isinstance(data, dict):
        raise ValueError("new_game.config must be an object.")

    villain_type = data.get("villain_type", "passive")
    if villain_type not in {"passive", "random"}:
        raise ValueError("villain_type must be 'passive' or 'random'.")

    seed_value = data.get("seed")
    if seed_value is None:
        seed: int | None = None
    else:
        try:
            seed = int(seed_value)
        except Exception as exc:
            raise ValueError("seed must be an integer.") from exc

    return GameConfig(
        hero_deck=_normalize_deck(data.get("hero_deck"), DEFAULT_DECK),
        villain_deck=_normalize_deck(data.get("villain_deck"), DEFAULT_DECK),
        villain_type=villain_type,
        seed=seed,
    )


class GameSession:
    def __init__(self, trace_dir: Path | None = None):
        self.trace_dir = trace_dir or trace_store.TRACES_DIR
        self.env: managym.Env | None = None
        self.obs: managym.Observation | None = None
        self.villain_policy: Callable[[managym.Observation], int] | None = None
        self.trace: Trace | None = None
        self.trace_id: str | None = None
        self._trace_saved = False

    def new_game(self, raw_config: Any) -> dict[str, Any]:
        if self.trace is not None:
            self.close(end_reason="new_game")

        config = _parse_game_config(raw_config)
        self.villain_policy = build_villain_policy(config.villain_type)

        seed = config.seed if config.seed is not None else 0
        self.env = managym.Env(seed=seed)

        player_configs = [
            managym.PlayerConfig("Hero", config.hero_deck),
            managym.PlayerConfig("Villain", config.villain_deck),
        ]
        self.obs, _ = self.env.reset(player_configs)

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

        self._auto_play_villain()
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
        self._auto_play_villain()
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
    ) -> None:
        if self.env is None or self.obs is None or self.trace is None:
            raise RuntimeError("Cannot step without an active game.")

        observation = serialize_observation(self.obs)
        action_description = actions[action_index]["description"]
        next_obs, reward, _, _, _ = self.env.step(action_index)
        self.trace.events.append(
            TraceEvent(
                actor=actor,
                observation=observation,
                actions=actions,
                action=action_index,
                action_description=action_description,
                reward=float(reward),
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
            action_index = int(self.villain_policy(self.obs))
            if action_index < 0 or action_index >= len(actions):
                raise RuntimeError(
                    f"Villain policy selected invalid action index: {action_index}"
                )
            self._step_and_record(
                actor="villain", action_index=action_index, actions=actions
            )

    def _wire_message(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("No observation available.")

        data = serialize_observation(self.obs)
        if self.obs.game_over:
            self._finalize_trace(end_reason="game_over")
            return {
                "type": "game_over",
                "data": data,
                "winner": _winner_for_hero(self.obs),
            }

        return {
            "type": "observation",
            "data": data,
            "actions": describe_actions(self.obs),
        }

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


def _new_session_id() -> str:
    return secrets.token_urlsafe(12)


def _new_resume_token() -> str:
    return secrets.token_urlsafe(24)


def _create_session_record() -> SessionRecord:
    while True:
        session_id = _new_session_id()
        if session_id not in SESSION_REGISTRY:
            break

    record = SessionRecord(
        session_id=session_id,
        resume_token=_new_resume_token(),
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


async def _attach_session_websocket(record: SessionRecord, websocket: WebSocket) -> None:
    previous_websocket = record.websocket
    record.websocket = websocket
    record.touch()
    if previous_websocket is not None and previous_websocket is not websocket:
        with suppress(Exception):
            await previous_websocket.close(code=4000)


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
                    if (
                        attached_session_id is None
                        or attached_session_id not in SESSION_REGISTRY
                    ):
                        record = _create_session_record()
                        await _attach_session_websocket(record, websocket)
                        attached_session_id = record.session_id
                    else:
                        record = SESSION_REGISTRY[attached_session_id]
                        await _attach_session_websocket(record, websocket)

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

    if not reveal_hidden:
        payload = trace_store.redact_trace_payload(payload)

    payload["id"] = trace_id
    return payload
