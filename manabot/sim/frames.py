"""Viewer-safe protocol frames and prompt-bound Commands for native games.

Arena games, replay and checkpoint scenarios record the acting player's
redacted protocol-v1 frame and the Command chosen from its offers, so a trace
replays without exposing the opponent's private hand. The frame shape,
including its historical "Teacher decision" prompt title, is hashed into
retained evidence and must not change. Torch-free: game children import it.
"""

from __future__ import annotations

from typing import Any

from managym.experience import PROTOCOL_VERSION, Command, ExperienceFrame
from managym.experience_projection import (
    describe_actions,
    offer_verb,
    redact_observation,
    serialize_observation,
)

from manabot.env.observation import ActionSpaceEnum
from manabot.infra.identity import canonical_sha256


def _action_space_name(raw: Any) -> str:
    return ActionSpaceEnum(int(raw.action_space.action_space_type)).name


def build_viewer_frame(
    raw: Any,
    *,
    match_id: str,
    revision: int,
    content_hash: str,
    asset_manifest_hash: str,
) -> dict[str, Any]:
    """Build the protocol frame for exactly the acting player's observation."""

    projection = serialize_observation(raw)
    redact_observation(projection)
    actor = int(raw.agent.player_index)
    actions = describe_actions(raw)
    action_space = _action_space_name(raw)
    offers = [
        {
            "id": int(action["index"]),
            "actor": actor,
            "verb": offer_verb(str(action["type"])),
            "source": None,
            "label": str(action["description"]),
            "help": None,
            "choices": [],
            "confirm_label": str(action["description"]),
            "action_type": str(action["type"]),
            "focus": [int(value) for value in action["focus"]],
        }
        for action in actions
    ]
    prompt = {
        "id": revision,
        "actor": actor,
        "kind": action_space.lower(),
        "title": "Teacher decision",
        "instruction": "Choose an action",
    }
    core = {
        "protocol": PROTOCOL_VERSION,
        "match_id": match_id,
        "revision": revision,
        "content_hash": content_hash,
        "asset_manifest_hash": asset_manifest_hash,
        "status": "ready",
        "prompt": prompt,
        "projection": projection,
        "offers": offers,
    }
    frame = {
        **core,
        "frame_hash": canonical_sha256(core),
        "winner": None,
        "action_space": action_space,
        "stops": {
            "my": [],
            "opponent": [],
            "stop_on_stack": False,
            "auto_pass": False,
        },
    }
    validated = ExperienceFrame.model_validate(frame)
    return validated.model_dump(mode="json", exclude_unset=True)


def build_command(frame: dict[str, Any], action: int) -> dict[str, Any]:
    offer_ids = {int(offer["id"]) for offer in frame["offers"]}
    if action not in offer_ids:
        raise RuntimeError(f"teacher action {action} is absent from legal offers")
    command = Command(
        command_id=f"{frame['match_id']}:{frame['revision']}",
        match_id=str(frame["match_id"]),
        expected_revision=int(frame["revision"]),
        prompt_id=int(frame["prompt"]["id"]),
        offer_id=action,
        answers=[],
    )
    return command.model_dump(mode="json")
