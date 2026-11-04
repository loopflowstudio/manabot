"""Public committed combat scheduling survives retained-session recovery."""

import json
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


def test_bumi_schedule_reconnect_keeps_current_restriction_and_rejects_stale_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    # Bolt leaves a real priority choice after the trigger, so native trivial
    # skipping does not jump straight past the pending schedule.
    config = {
        "hero_deck": {
            "Mountain": 16,
            "Forest": 12,
            "Bumi, Unleashed": 8,
            "Lightning Bolt": 4,
        },
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "decklists": "closed",
        "seed": 165,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            cast = False
            saw_pending = False
            stale: dict[str, object] | None = None
            for ordinal in range(250):
                schedule = frame.projection.schedule
                assert schedule is not None
                if schedule.additional_combats:
                    saw_pending = True
                    assert schedule.additional_combats[0].attackers == "land creatures"
                if schedule.current_combat and schedule.current_combat.attackers:
                    assert schedule.current_combat.attackers == "land creatures"
                    assert not schedule.additional_combats
                    break
                labels = ["Play Mountain", "Play Forest"]
                if not cast:
                    labels.append("Cast Bumi, Unleashed")
                labels.extend(
                    [
                        "Target Mountain",
                        "Target Forest",
                        "Attack with Bumi, Unleashed",
                        "Pass priority",
                    ]
                )
                offer = next(
                    (o for label in labels for o in frame.offers if o.label == label),
                    frame.offers[0],
                )
                cast |= offer.label == "Cast Bumi, Unleashed"
                stale = _command(frame, offer, f"schedule-prefix.{ordinal}")
                socket.send_json({"type": "command", "command": stale})
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail(
                    f"Restricted combat not reached: {frame.action_space}, {[o.label for o in frame.offers]}"
                )
            assert saw_pending and stale is not None
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            assert game.env is not None and game.trace is not None
            digest = game.env.state_digest()
            events = len(game.trace.events)
            for viewer in (0, 1):
                view = decision_surface_view(game.env, viewer)
                assert view["schedule"] == schedule.model_dump(mode="json")
                assert not view["opponent"]["hand"]
                assert "main_counts" not in view["opponent"]
                assert not view["agent"].get("inspected")

        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovered = ExperienceFrame.model_validate(
                socket.receive_json()["recovery"]["frame"]
            )
            assert recovered == frame
            socket.send_json(
                {"type": "command", "command": stale | {"command_id": "schedule-stale"}}
            )
            rejected = socket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert game.env.state_digest() == digest
            assert len(game.trace.events) == events
            assert decision_surface_view(game.env, 0)[
                "schedule"
            ] == schedule.model_dump(mode="json")


def test_delayed_lifetime_reconnect_preserves_waiter_and_expired_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Swamp": 12, "Mountain": 8, "Gray Ogre": 12, "Fatal Fissure": 8},
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "decklists": "closed",
        "seed": 165,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            cast_ogre = False
            played_mountain = False
            for ordinal in range(250):
                assert game.obs is not None
                waiting = [f for f in game.obs.continuation if f.kind == 10]
                if waiting:
                    break
                labels = (
                    ["Play Swamp", "Play Mountain"]
                    if played_mountain
                    else ["Play Mountain", "Play Swamp"]
                )
                if not cast_ogre:
                    labels.append("Cast Gray Ogre")
                else:
                    labels.append("Cast Fatal Fissure")
                labels.extend(["Target Gray Ogre", "Pass priority"])
                offer = next(
                    (o for label in labels for o in frame.offers if o.label == label),
                    frame.offers[0],
                )
                cast_ogre |= offer.label == "Cast Gray Ogre"
                played_mountain |= offer.label == "Play Mountain"
                command = _command(frame, offer, f"delayed-prefix.{ordinal}")
                socket.send_json({"type": "command", "command": command})
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail("Fatal Fissure's watcher was not reached")
            assert len(waiting) == 1
            waiter = waiting[0]
            assert waiter.delayed_frequency == 1 and waiter.this_turn
            assert waiter.controller == game.obs.agent.player_index
            assert len(waiter.remembered) == 1
            assert game.env is not None
            digest = game.env.state_digest()
            captured = json.loads(
                game.env.capture_decision(waiter.controller).to_json()
            )["digest"]
            clone = game.env.clone_env()
            assert clone.state_digest() == digest
            assert (
                json.loads(clone.capture_decision(waiter.controller).to_json())[
                    "digest"
                ]
                == captured
            )

        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovered = ExperienceFrame.model_validate(
                socket.receive_json()["recovery"]["frame"]
            )
            assert recovered == frame
            assert game.env.state_digest() == digest
            assert (
                json.loads(game.env.capture_decision(waiter.controller).to_json())[
                    "digest"
                ]
                == captured
            )
            socket.send_json(
                {
                    "type": "command",
                    "command": command | {"command_id": "delayed-stale"},
                }
            )
            assert socket.receive_json()["status"] == "rejected"
            assert game.env.state_digest() == digest
            # No match before cleanup: let the real turn advance, without
            # recasting or recreating the watcher during recovery.
            for ordinal in range(100):
                assert game.obs is not None
                if not any(f.kind == 10 for f in game.obs.continuation):
                    break
                offer = next(
                    (o for o in frame.offers if o.label == "Pass priority"),
                    frame.offers[0],
                )
                socket.send_json(
                    {
                        "type": "command",
                        "command": _command(frame, offer, f"delayed-expire.{ordinal}"),
                    }
                )
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail("This-turn watcher did not expire")
            expired_digest = game.env.state_digest()
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            assert (
                ExperienceFrame.model_validate(
                    socket.receive_json()["recovery"]["frame"]
                )
                == frame
            )
            assert game.env.state_digest() == expired_digest
            assert game.obs is not None
            assert not any(f.kind == 10 for f in game.obs.continuation)
