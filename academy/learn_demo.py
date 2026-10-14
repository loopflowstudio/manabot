"""A checked legal Command prefix into the ordinary Learn interaction."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import GameSession


DEMO_PATH = Path(__file__).parent / "fixtures" / "learn-demo.json"


def load_learn_demo() -> dict[str, Any]:
    return json.loads(DEMO_PATH.read_text())


def replay_learn_demo(
    session: "GameSession", update: dict[str, Any], demo: dict[str, Any]
) -> dict[str, Any]:
    if session.env.content_pack_manifest() != demo["manifest"]:
        raise ValueError("Learn demo content/setup identity differs from this runtime")
    for ordinal, step in enumerate(demo["prefix"]):
        if session.env.state_digest() != step["before_digest"]:
            raise ValueError(f"Learn demo state diverged before Command {ordinal}")
        frame = update["frame"]
        offer = next(o for o in frame["offers"] if o["id"] == step["offer_id"])
        if offer["label"] != step["label"]:
            raise ValueError(f"Learn demo offer diverged before Command {ordinal}")
        result = session.hero_command(
            {
                "command_id": f"learn-demo.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        if result["status"] != "accepted":
            raise ValueError(f"Learn demo Command {ordinal} rejected")
        update = result["update"]
    if (
        session.env.state_digest() != demo["learn_digest"]
        or update["frame"]["revision"] != demo["learn_revision"]
        or update["frame"]["action_space"] != "LEARN"
    ):
        raise ValueError("Learn demo did not reach its checked decision")
    return update
