"""A private named-card search retains its choices and permissions on reconnect."""

from pathlib import Path

from fastapi.testclient import TestClient
from managym.experience import ExperienceFrame, InteractionOffer
from managym.experience_projection import decision_surface_view
import pytest

from academy import server, trace as trace_store


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


def test_sozin_search_reconnect_retains_selection_visibility_and_stale_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Swamp": 24, "The Rise of Sozin": 12},
        "villain_deck": {"Plains": 20, "Gray Ogre": 20},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": 165,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            cast = False
            # Reach chapter II using ordinary Commands, without injecting state.
            for ordinal in range(200):
                if any(o.label == "Name Gray Ogre" for o in frame.offers):
                    break
                labels = ["Name Gray Ogre", "Target Villain", "Play Swamp"]
                if not cast:
                    labels.append("Cast The Rise of Sozin")
                labels.extend(["Pass priority", "Decline"])
                offer = next(
                    (
                        offer
                        for label in labels
                        for offer in frame.offers
                        if offer.label == label
                    ),
                    frame.offers[0],
                )
                cast |= offer.label == "Cast The Rise of Sozin"
                socket.send_json(
                    {
                        "type": "command",
                        "command": _command(frame, offer, f"search-prefix.{ordinal}"),
                    }
                )
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail("Sozin's naming decision was not reached")

        # Disconnect before naming: recovery must retain the complete vocabulary
        # and its exact Command binding, without granting search inspection early.
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovered_name = ExperienceFrame.model_validate(
                socket.receive_json()["recovery"]["frame"]
            )
            assert recovered_name == frame
            name_offer = next(
                offer for offer in frame.offers if offer.label == "Name Gray Ogre"
            )
            name_command = _command(frame, name_offer, "name-gray-ogre")
            socket.send_json({"type": "command", "command": name_command})
            named = socket.receive_json()
            assert named["status"] == "accepted"
            frame = ExperienceFrame.model_validate(named["update"]["frame"])
            socket.send_json(
                {
                    "type": "command",
                    "command": name_command | {"command_id": "name-stale"},
                }
            )
            stale_name = socket.receive_json()
            assert stale_name["status"] == "rejected"
            assert stale_name["rejection"]["code"] == "stale_revision"
            assert frame.action_space == "LOOK_AND_SELECT"
            choices = [o for o in frame.offers if o.action_type == "SELECT_CARD"]
            assert len(choices) > 4
            chosen = choices[0]
            command = _command(frame, chosen, "search-first")
            socket.send_json({"type": "command", "command": command})
            result = socket.receive_json()
            assert result["status"] == "accepted"
            pending = ExperienceFrame.model_validate(result["update"]["frame"])
            assert pending.action_space == "LOOK_AND_SELECT"
            assert [
                o.focus for o in pending.offers if o.action_type == "SELECT_CARD"
            ] == [o.focus for o in choices[1:]]
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            assert game.env is not None
            assert game.obs is not None and game.trace is not None
            assert any(f.decision == 25 and f.mode == 3 for f in game.obs.continuation)
            digest = game.env.state_digest()
            events = len(game.trace.events)
            searched = decision_surface_view(game.env, 0)
            assert searched["agent"]["inspected"]
            assert chosen.focus[0] in {c["id"] for c in searched["opponent"]["exile"]}
            assert not searched["opponent"]["hand"]
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")

        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovery = socket.receive_json()["recovery"]
            recovered = ExperienceFrame.model_validate(recovery["frame"])
            assert recovered == pending
            assert game.env.state_digest() == digest
            assert decision_surface_view(game.env, 0) == searched
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")

            socket.send_json({"type": "command", "command": command})
            assert socket.receive_json()["status"] == "duplicate"
            socket.send_json(
                {
                    "type": "command",
                    "command": command | {"command_id": "search-stale"},
                }
            )
            rejected = socket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert game.env.state_digest() == digest
            assert len(game.trace.events) == events

            stop = next(
                o for o in recovered.offers if o.label == "Stop searching and shuffle"
            )
            socket.send_json(
                {"type": "command", "command": _command(recovered, stop, "search-stop")}
            )
            assert socket.receive_json()["status"] == "accepted"
            finished = decision_surface_view(game.env, 0)
            assert not finished["agent"].get("inspected")
            assert chosen.focus[0] in {c["id"] for c in finished["opponent"]["exile"]}
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")
