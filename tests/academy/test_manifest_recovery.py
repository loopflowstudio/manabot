"""Recover a real manifest offer, then replay its private choice and public reveal.

A manifest's public references keep incarnation 0 after it turns face up, so
Academy's offer subjects, presentation events, reconnect frames and recording
labels must still join them to the face-up permanent on the board.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession

from manabot.semantic.text_observation import capture_text_decision
import managym
from managym.academy import server, trace as trace_store
from managym.academy.recording import Recording
from managym.experience import ExperienceFrame, InteractionOffer, ObjectRenderId
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


def cited_objects(event: object) -> list[dict[str, int]]:
    """Every `{entity, incarnation}` reference inside a presentation event."""
    if isinstance(event, dict):
        own = [event] if set(event) == {"entity", "incarnation"} else []
        return own + [ref for child in event.values() for ref in cited_objects(child)]
    if isinstance(event, list):
        return [ref for child in event for ref in cited_objects(child)]
    return []


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

            # The face-up manifest is still cited at incarnation 0, and every
            # consumer joins that reference to the named board permanent.
            for _ in range(4):
                if any(o.label == "Attack with Llanowar Elves" for o in frame.offers):
                    break
                pass_priority = next(
                    o for o in frame.offers if o.label == "Pass priority"
                )
                frame = submit(socket, frame, pass_priority)
            attack = next(
                o for o in frame.offers if o.label == "Attack with Llanowar Elves"
            )
            (subject,) = attack.subjects
            board = {
                card["id"]: card["name"]
                for card in frame.projection.model_dump(mode="json")["agent"][
                    "battlefield"
                ]
            }
            elves = subject.object.display_id
            assert board[elves] == subject.object.label == "Llanowar Elves"
            assert subject.object.subject.id == ObjectRenderId(
                entity=elves, incarnation=0
            )
            assert attack.focus == [elves]
            frame = submit(socket, frame, attack)
            for _ in range(6):
                kinds = {e["kind"]["kind"] for e in game.presentation_events}
                if "damage" in kinds:
                    break
                frame = submit(socket, frame, frame.offers[0])
            combat = [
                e
                for e in game.presentation_events
                if e["kind"]["kind"] in ("attack_group", "damage")
            ]
            assert {e["kind"]["kind"] for e in combat} == {"attack_group", "damage"}
            for event in combat:
                assert {"entity": elves, "incarnation": 0} in cited_objects(event)

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
            payload = game.attempt_store.load(game.match_id)
            assert payload is not None
            latest = Recording(game.match_id, payload).read("latest")
            labelled = [
                labels["objects"].get(f"{elves}:0")
                for event, labels in zip(
                    latest["history"], latest["history_labels"], strict=True
                )
                if event["kind"]["kind"] in ("attack_group", "damage")
            ]
            assert labelled == ["Llanowar Elves", "Llanowar Elves"]
            assert "expired object" not in json.dumps(latest)

            archive = game.env.snapshot().to_replay_json()
            replay = managym.GameSnapshot.from_replay_json(archive).restore()
            assert replay.state_digest() == game.env.state_digest()
            assert (
                replay.semantic_decision_frame_json()
                == game.env.semantic_decision_frame_json()
            )
