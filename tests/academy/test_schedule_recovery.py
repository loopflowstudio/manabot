"""Public committed combat scheduling survives retained-session recovery."""

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
