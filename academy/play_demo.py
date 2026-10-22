"""Checked legal prefixes into ordinary GameSession decisions, without injections.

The fixture pins content and state digests. Every prefix step uses the public
Command path; final recovery is built by GameSession at the resulting revision.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from .server import GameSession


class DemoStep(BaseModel):
    before_digest: str
    offer_id: int
    label: str


class PlayDemo(BaseModel):
    config: dict[str, Any]
    manifest: dict[str, Any]
    prefix: list[DemoStep]
    decision_digest: str
    decision_revision: int
    decision_kind: str = "LEARN"


def load_play_demo(name: str) -> PlayDemo:
    filenames = {
        "learn": "learn-demo-explicit-w10.json",
        "precision": "precision-demo-w10.json",
        "precision-mana": "precision-mana-demo-w10.json",
    }
    if name not in filenames:
        raise ValueError(f"Unknown play demo: {name}")
    return PlayDemo.model_validate_json(
        (Path(__file__).parent / "fixtures" / filenames[name]).read_text()
    )


def replay_checked_prefix(
    session: "GameSession", update: dict[str, Any], demo: PlayDemo, name: str
) -> dict[str, Any]:
    if session.env.content_pack_manifest() != demo.manifest:
        raise ValueError(
            f"{name} demo content/setup identity differs from this runtime"
        )
    for ordinal, step in enumerate(demo.prefix):
        if session.env.state_digest() != step.before_digest:
            raise ValueError(f"{name} demo state diverged before Command {ordinal}")
        frame = update["frame"]
        offer = next(o for o in frame["offers"] if o["id"] == step.offer_id)
        if offer["label"] != step.label:
            raise ValueError(f"{name} demo offer diverged before Command {ordinal}")
        result = session.hero_command(
            {
                "command_id": f"{name}-demo.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        if result["status"] != "accepted":
            raise ValueError(f"{name} demo Command {ordinal} rejected")
        update = result["update"]
    if (
        session.env.state_digest() != demo.decision_digest
        or update["frame"]["revision"] != demo.decision_revision
        or update["frame"]["action_space"] != demo.decision_kind
    ):
        raise ValueError(f"{name} demo did not reach its checked decision")
    return update
