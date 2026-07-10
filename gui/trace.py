"""
trace.py
Trace dataclasses and JSON persistence helpers for GUI games.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
TRACES_DIR = Path(os.getenv("MANABOT_GUI_TRACES_DIR", "gui/traces"))


@dataclass
class GameConfig:
    hero_deck: dict[str, int]
    villain_deck: dict[str, int]
    villain_type: str
    seed: int | None = None
    # Named-deck identifiers ("interactive" / "ur_lessons" / "gw_allies" /
    # "custom") — recorded so every trace is attributable to an exact matchup.
    hero_deck_name: str = "custom"
    villain_deck_name: str = "custom"
    # Opponent parameters (recorded in traces so every game is attributable
    # to an exact opponent configuration).
    villain_sims: int | None = None  # search: simulations per legal action
    villain_checkpoint: str | None = None  # checkpoint: path to .pt file
    villain_deterministic: bool = False  # checkpoint: argmax instead of sampling
    # Priority-stop configuration at game start (MTGO-style auto-pass).
    # ``stops`` maps "my"/"opponent" to the stop step names that surface;
    # None means the server defaults. set_stops updates the live session,
    # not this record.
    stops: dict[str, list[str]] | None = None
    stop_on_stack: bool = True  # always surface when the stack is non-empty
    auto_pass: bool = True  # master switch; False = surface every window


@dataclass
class TraceEvent:
    actor: str
    observation: dict[str, Any]
    actions: list[dict[str, Any]]
    action: int
    action_description: str
    reward: float
    # True when the server auto-passed this priority window (stops system /
    # F6), False for decisions the human actually clicked. Competency metrics
    # must not credit auto-passes as deliberate passes.
    auto: bool = False


@dataclass
class Trace:
    config: GameConfig
    events: list[TraceEvent]
    final_observation: dict[str, Any]
    winner: int | None
    end_reason: str
    timestamp: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp_for_filename(timestamp: str) -> str:
    # Replace the UTC offset before stripping ':' so the trace id contains no
    # '+' (TRACE_ID_PATTERN would reject it and the trace could not be loaded).
    normalized = (
        timestamp.replace("+00:00", "Z")
        .replace("-", "")
        .replace(":", "")
        .replace(".", "_")
    )
    return normalized


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    return asdict(trace)


def _trace_path(trace_id: str, trace_dir: Path) -> Path:
    if not TRACE_ID_PATTERN.fullmatch(trace_id):
        raise ValueError("Invalid trace id")
    return trace_dir / f"{trace_id}.json"


def save_trace(trace: Trace, trace_dir: Path | None = None) -> Path:
    target_dir = trace_dir or TRACES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    base_stem = f"{_normalize_timestamp_for_filename(trace.timestamp)}_hero_vs_villain"
    path = target_dir / f"{base_stem}.json"
    suffix = 1
    while path.exists():
        path = target_dir / f"{base_stem}_{suffix}.json"
        suffix += 1

    payload = trace_to_dict(trace)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def load_trace(trace_id: str, trace_dir: Path | None = None) -> dict[str, Any]:
    target_dir = trace_dir or TRACES_DIR
    path = _trace_path(trace_id, target_dir)
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_id}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def list_trace_summaries(trace_dir: Path | None = None) -> list[dict[str, Any]]:
    target_dir = trace_dir or TRACES_DIR
    if not target_dir.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for path in target_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        events = payload.get("events", [])
        summaries.append(
            {
                "id": path.stem,
                "timestamp": payload.get("timestamp"),
                "winner": payload.get("winner"),
                "end_reason": payload.get("end_reason"),
                "num_events": len(events),
            }
        )

    summaries.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    return summaries


def _redact_hand(player_state: dict[str, Any]) -> None:
    hand = player_state.get("hand")
    if not isinstance(hand, list):
        return
    zone_counts = player_state.get("zone_counts")
    if isinstance(zone_counts, dict) and "HAND" in zone_counts:
        player_state["hand_hidden_count"] = int(zone_counts["HAND"])
    else:
        player_state["hand_hidden_count"] = len(hand)
    player_state["hand"] = []


def redact_observation(observation: dict[str, Any]) -> None:
    """Strip the opponent's hidden information from a serialized observation.

    Mutates in place. Libraries are never serialized (only counts), so the
    opponent's hand is the only hidden zone to remove.
    """
    if not isinstance(observation, dict):
        return

    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        _redact_hand(opponent)


# Backwards-compatible alias used by redact_trace_payload.
_redact_observation = redact_observation


def normalize_observation_to_hero(observation: dict[str, Any]) -> None:
    """Present an observation from the hero's (player 0) perspective.

    Trace events record the raw engine observation, whose perspective is the
    player to act — villain events are villain-perspective. The replay viewer
    (and any human-facing payload) renders ``agent`` as the hero, so swap the
    sides back when needed. Mutates in place; no-op when player_index is
    missing (e.g. minimal test fixtures).
    """
    if not isinstance(observation, dict):
        return
    agent = observation.get("agent")
    opponent = observation.get("opponent")
    if not isinstance(agent, dict) or not isinstance(opponent, dict):
        return
    if agent.get("player_index") == 1 and opponent.get("player_index") == 0:
        observation["agent"], observation["opponent"] = opponent, agent
        if observation.get("game_over") and "won" in observation:
            observation["won"] = not bool(observation["won"])


def prepare_trace_payload(
    payload: dict[str, Any], reveal_hidden: bool = False
) -> dict[str, Any]:
    """Normalize a stored trace for the viewer: hero-perspective throughout,
    with the villain's hand redacted unless ``reveal_hidden``."""
    prepared = deepcopy(payload)

    observations = [
        event.get("observation", {}) for event in prepared.get("events", [])
    ]
    observations.append(prepared.get("final_observation", {}))

    for observation in observations:
        normalize_observation_to_hero(observation)
        if not reveal_hidden:
            redact_observation(observation)

    return prepared


def redact_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return prepare_trace_payload(payload, reveal_hidden=False)
