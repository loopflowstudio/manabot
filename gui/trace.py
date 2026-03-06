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


@dataclass
class TraceEvent:
    actor: str
    observation: dict[str, Any]
    actions: list[dict[str, Any]]
    action: int
    action_description: str
    reward: float


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
    normalized = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
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
    player_state["hand_hidden_count"] = len(hand)
    player_state["hand"] = []


def _redact_observation(observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        return

    opponent = observation.get("opponent")
    if isinstance(opponent, dict):
        _redact_hand(opponent)


def redact_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = deepcopy(payload)

    for event in redacted.get("events", []):
        _redact_observation(event.get("observation", {}))

    _redact_observation(redacted.get("final_observation", {}))

    return redacted
