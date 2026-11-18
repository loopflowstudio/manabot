"""Private repeated-play setup and continuation receipts in the attempt database.

Attempts remain the outcome authority. Session rows retain only the next setup;
request receipts and membership are inserted with the attempt's first save.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import Field

from managym.experience import ProtocolModel, UInt64

if TYPE_CHECKING:
    from managym.academy.trace import GameConfig


class PlaySessionRecord(TypedDict):
    id: str
    created_at: str
    title: str
    revision: int
    # The existing wire setup is validated through _parse_game_config before
    # saving and re-admitted before play; sources never leave owner reads.
    draft: dict[str, Any]


class SessionMembership(TypedDict):
    session_id: str
    game_number: int
    revision: int


@dataclass(frozen=True)
class SessionStart:
    session_id: str
    owner: str
    request_id: str
    fingerprint: str
    revision: int
    draft: dict[str, Any]


def request_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def setup_draft(config: GameConfig) -> dict[str, Any]:
    """Re-admittable setup, without a resolved random seed or player credentials."""
    draft: dict[str, Any] = {
        "hero_deck": config.hero_deck_name
        if config.hero_deck_name != "custom"
        else config.hero_deck,
        "villain_deck": config.villain_deck_name
        if config.villain_deck_name != "custom"
        else config.villain_deck,
        "villain_type": config.villain_type,
        "villain_sims": config.villain_sims,
        "human_seat": config.human_seat,
        "mana_floats": config.mana_floats,
        "mana_payment": config.mana_payment,
        "decklists": config.decklists,
        "stops": config.stops,
        "auto_pass": config.auto_pass,
        "stop_on_stack": config.stop_on_stack,
        "seed": str(config.seed) if config.seed_was_requested else None,
        "mirror": asdict(config.mirror) if config.mirror else None,
    }
    if config.opponent and "sha256" in config.opponent:
        draft["opponent_sha256"] = config.opponent.get(
            "base_sha256", config.opponent["sha256"]
        )
    if config.hero_source is not None and config.villain_source is not None:
        draft["deck_import"] = {
            side: {"source": source["source"], "filename": source["filename"]}
            for side, source in (
                ("hero", config.hero_source),
                ("villain", config.villain_source),
            )
        }
    return draft


def continuation_config(
    draft: dict[str, Any],
    overrides: dict[str, Any],
    intent: Literal["rematch", "swap_sides", "change_opponent"],
) -> dict[str, Any]:
    """Apply explicit edits once, after deriving seat changes from retained setup."""
    resolved = dict(draft)
    if intent == "swap_sides":
        resolved["human_seat"] = 1 - draft.get("human_seat", 0)
    if "hero_deck" in overrides or "villain_deck" in overrides:
        resolved.pop("deck_import", None)
    resolved.update(overrides)
    return resolved


class SessionSetupUpdate(ProtocolModel):
    table_id: str
    grant_revision: UInt64
    revision: int = Field(ge=1)
    config: dict[str, Any]
