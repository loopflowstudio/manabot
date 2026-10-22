"""An interrupted optional mana activation retains its committed costs on resume."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from academy import server, trace as trace_store
from managym.experience import ExperienceFrame, InteractionOffer


def _command(
    frame: ExperienceFrame, offer: InteractionOffer, identity: str
) -> dict[str, object]:
    assert frame.prompt is not None
    return {
        "command_id": identity,
        "match_id": frame.match_id,
        "expected_revision": frame.revision,
        "prompt_id": frame.prompt.id,
        "offer_id": offer.id,
        "answers": [],
    }


def test_optional_mana_cost_survives_lost_response_reconnect_and_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Island": 20, "Waterbending Lesson": 8, "Prophetic Prism": 8},
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": 165,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            initial = websocket.receive_json()
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            prism_cast = False
            # A fixed legal setup prefix through the real server; no state injection.
            for ordinal in range(80):
                if frame.action_space == "WATERBEND":
                    break
                labels = [
                    "Play Island",
                    "Cast Waterbending Lesson"
                    if prism_cast
                    else "Cast Prophetic Prism",
                    "Pass priority",
                ]
                offer = next(
                    (
                        offer
                        for label in labels
                        for offer in frame.offers
                        if offer.label == label
                    ),
                    next(
                        (
                            offer
                            for offer in frame.offers
                            if offer.action_type == "PAY_COST"
                        ),
                        frame.offers[0],
                    ),
                )
                prism_cast |= offer.label == "Cast Prophetic Prism"
                websocket.send_json(
                    {
                        "type": "command",
                        "command": _command(frame, offer, f"optional-prefix.{ordinal}"),
                    }
                )
                outcome = websocket.receive_json()
                assert outcome["status"] == "accepted"
                frame = ExperienceFrame.model_validate(outcome["update"]["frame"])
            assert frame.action_space == "WATERBEND"
            activation = next(
                offer
                for offer in frame.offers
                if offer.label == "Activate ability: Prophetic Prism"
            )
            command = _command(frame, activation, "optional-lost-activation")

        record = server.SESSION_REGISTRY[initial["session_id"]]
        accepted = record.game.hero_command(command)
        assert accepted["status"] == "accepted"
        assert accepted["update"]["frame"]["action_space"] == "MODAL"
        assert sum(record.game.obs.agent.mana) == 0
        digest = record.game.env.state_digest()
        events = len(record.game.trace.events)
        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovery = resumed.receive_json()["recovery"]
            assert (
                recovery["frame"]["frame_hash"]
                == accepted["update"]["frame"]["frame_hash"]
            )
            assert record.game.env.state_digest() == digest
            resumed.send_json({"type": "command", "command": command})
            assert resumed.receive_json()["status"] == "duplicate"
            assert record.game.env.state_digest() == digest
            assert len(record.game.trace.events) == events
            frame = ExperienceFrame.model_validate(recovery["frame"])
            resumed.send_json(
                {
                    "type": "command",
                    "command": _command(frame, frame.offers[0], "optional-color"),
                }
            )
            outcome = resumed.receive_json()
            assert outcome["status"] == "accepted"
            frame = ExperienceFrame.model_validate(outcome["update"]["frame"])
            # Prism's only remaining Island paid the activation. Its output is
            # retained, while the existing forced-action path declines {2} and
            # reaches mandatory discard without offering a second activation.
            assert frame.action_space == "DISCARD"
            assert sum(record.game.obs.agent.mana) == 1
