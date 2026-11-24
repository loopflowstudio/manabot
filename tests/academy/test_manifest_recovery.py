"""Recover a real manifest offer, then replay its private choice and public reveal."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession

import managym
from manabot.semantic.text_observation import capture_text_decision
from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame, InteractionOffer
from tests.academy.socket_messages import settled_message


def submit(
    socket: WebSocketTestSession, frame: ExperienceFrame, offer: InteractionOffer
) -> ExperienceFrame:
    assert frame.prompt is not None
    socket.send_json(
        {
            "type": "command",
            "command": {
                "command_id": f"manifest.{frame.revision}",
                "match_id": frame.match_id,
                "expected_revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "offer_id": offer.id,
                "answers": [],
            },
        }
    )
    result = settled_message(socket)
    assert result["status"] == "accepted"
    return ExperienceFrame.model_validate(result["update"]["frame"])


def test_manifest_turn_up_reconnect_text_and_native_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Forest": 32, "Manifest Dread": 4, "Llanowar Elves": 4},
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": 4,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = settled_message(socket)
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            manifested = False
            for _ in range(60):
                by_label = {offer.label: offer for offer in frame.offers}
                if any(
                    label.startswith("Turn Llanowar Elves face up")
                    for label in by_label
                ):
                    break
                labels = ["Manifest Llanowar Elves", "Play Forest"]
                if not manifested:
                    labels.append("Cast Manifest Dread")
                labels += ["Pay", "Pass priority"]
                offer = next(
                    (by_label[label] for label in labels if label in by_label),
                    frame.offers[0],
                )
                manifested |= offer.label == "Manifest Llanowar Elves"
                frame = submit(socket, frame, offer)
            else:
                pytest.fail("turn-up offer was never reached")
            assert manifested and game.env is not None
            assert any(
                "Turn Llanowar Elves face up" in a.description
                for a in capture_text_decision(game.env).actions
            )
            digest = game.env.state_digest()
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovered = ExperienceFrame.model_validate(
                settled_message(socket)["recovery"]["frame"]
            )
            assert recovered == frame
            assert game.env.state_digest() == digest
            reveal = next(
                o
                for o in recovered.offers
                if o.label.startswith("Turn Llanowar Elves face up")
            )
            frame = submit(socket, recovered, reveal)
            assert game.obs is not None
            if not any(
                p.characteristics.name == "Llanowar Elves"
                for p in game.obs.agent_permanents
            ):
                pay = next(o for o in frame.offers if o.label.startswith("Pay"))
                submit(socket, frame, pay)
            assert any(
                p.characteristics.name == "Llanowar Elves"
                for p in game.obs.agent_permanents
            )
            archive = game.env.snapshot().to_replay_json()
            replay = managym.GameSnapshot.from_replay_json(archive).restore()
            assert replay.state_digest() == game.env.state_digest()
            assert (
                replay.semantic_decision_frame_json()
                == game.env.semantic_decision_frame_json()
            )
