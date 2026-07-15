"""
test_server.py
WebSocket integration tests for the GUI backend server.
"""

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

# Local imports
from gui import server, trace as trace_store
from gui.server import app

PROTOCOL_DIR = Path(__file__).parents[2] / "protocol"
PROTOCOL_V1_BOLT_FIXTURE = json.loads(
    (PROTOCOL_DIR / "fixtures" / "bolt-target.json").read_text(encoding="utf-8")
)
PROTOCOL_V1_SCHEMA = json.loads(
    (PROTOCOL_DIR / "experience-v1.schema.json").read_text(encoding="utf-8")
)
PROTOCOL_V1_VALIDATOR = Draft202012Validator(PROTOCOL_V1_SCHEMA)


def _pick_action(actions: list[dict]) -> int:
    preferred_action_types = {*server.ACTION_LABELS, "PRIORITY_PASS_PRIORITY"}
    for action in actions:
        if action["type"] in preferred_action_types:
            return int(action["index"])

    return int(actions[0]["index"])


def _command(frame: dict, offer: dict, command_id: str) -> dict:
    return {
        "command_id": command_id,
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": offer["id"],
        "answers": [],
    }


def _offer(frame: dict, *, action_type: str, label: str | None = None) -> dict:
    return next(
        offer
        for offer in frame["offers"]
        if offer["action_type"] == action_type
        and (label is None or offer["label"] == label)
    )


def test_protocol_v1_bolt_and_pass_offers_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Mountain": 12, "Lightning Bolt": 12},
        "villain_deck": {"Island": 20},
        "villain_type": "passive",
        "seed": 3,
        "auto_pass": False,
    }

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            initial = websocket.receive_json()
            frame = initial["recovery"]["frame"]
            assert frame["protocol"] == 1
            assert _offer(frame, action_type="PRIORITY_PASS_PRIORITY")["verb"] == (
                "pass_priority"
            )

            for action_type, label in (
                ("PRIORITY_PLAY_LAND", None),
                ("PRIORITY_CAST_SPELL", "Cast Lightning Bolt"),
                ("CHOOSE_TARGET", "Target Villain"),
                ("PRIORITY_PASS_PRIORITY", "Pass priority"),
            ):
                offer = _offer(frame, action_type=action_type, label=label)
                command_id = f"bolt-{frame['revision']}"
                if action_type == "CHOOSE_TARGET":
                    fixture_frame = PROTOCOL_V1_BOLT_FIXTURE["recovery"]["frame"]
                    for key in (
                        "protocol",
                        "revision",
                        "content_hash",
                        "asset_manifest_hash",
                        "status",
                        "prompt",
                        "offers",
                        "action_space",
                        "stops",
                    ):
                        assert frame[key] == fixture_frame[key]
                    command_id = PROTOCOL_V1_BOLT_FIXTURE["command"]["command_id"]

                command = _command(frame, offer, command_id)
                if action_type == "CHOOSE_TARGET":
                    fixture_command = PROTOCOL_V1_BOLT_FIXTURE["command"]
                    for key in (
                        "command_id",
                        "expected_revision",
                        "prompt_id",
                        "offer_id",
                        "answers",
                    ):
                        assert command[key] == fixture_command[key]
                    record = next(iter(server.SESSION_REGISTRY.values()))
                    live_bundle = {
                        "recovery": record.game.current_recovery("explicit_resync"),
                        "command": command,
                    }
                    PROTOCOL_V1_VALIDATOR.validate(live_bundle)
                websocket.send_json({"type": "command", "command": command})
                outcome = websocket.receive_json()
                assert outcome["status"] == "accepted"
                assert outcome["update"]["base_revision"] == frame["revision"]
                frame = outcome["update"]["frame"]

    assert frame["revision"] == 4
    assert frame["projection"]["opponent"]["life"] == 17
    record = next(iter(server.SESSION_REGISTRY.values()))
    assert [event.action_description for event in record.game.trace.events[:3]] == [
        "Play Mountain",
        "Cast Lightning Bolt",
        "Target Villain",
    ]
    assert any(
        event.actor == "hero"
        and event.action_description == "Pass priority"
        and not event.auto
        for event in record.game.trace.events
    )


def test_protocol_v1_shared_fixture_matches_rust_generated_schema():
    Draft202012Validator.check_schema(PROTOCOL_V1_SCHEMA)
    PROTOCOL_V1_VALIDATOR.validate(PROTOCOL_V1_BOLT_FIXTURE)

    invalid = json.loads(json.dumps(PROTOCOL_V1_BOLT_FIXTURE))
    invalid["recovery"]["protocol"] = 2
    assert list(PROTOCOL_V1_VALIDATOR.iter_errors(invalid))


def test_protocol_v1_rejects_stale_and_dedupes_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Mountain": 12, "Lightning Bolt": 12},
        "villain_deck": {"Island": 20},
        "villain_type": "passive",
        "seed": 3,
        "auto_pass": False,
    }

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            old_frame = websocket.receive_json()["frame"]
            land_command = _command(
                old_frame,
                _offer(old_frame, action_type="PRIORITY_PLAY_LAND"),
                "same-command",
            )
            stale_command = _command(
                old_frame,
                _offer(old_frame, action_type="PRIORITY_PASS_PRIORITY"),
                "stale-command",
            )

            websocket.send_json({"type": "command", "command": land_command})
            accepted = websocket.receive_json()
            assert accepted["status"] == "accepted"
            receipt = accepted["update"]["receipt"]

            websocket.send_json({"type": "command", "command": land_command})
            duplicate = websocket.receive_json()
            assert duplicate["status"] == "duplicate"
            assert duplicate["receipt"] == receipt
            assert duplicate["recovery"]["reason"] == "duplicate_command"
            assert duplicate["recovery"]["frame"]["revision"] == 1

            websocket.send_json({"type": "command", "command": stale_command})
            rejected = websocket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert rejected["recovery"]["reason"] == "stale_command"
            assert rejected["recovery"]["frame"]["revision"] == 1

    record = next(iter(server.SESSION_REGISTRY.values()))
    assert len(record.game.trace.events) == 1


def test_websocket_new_game_action_loop_and_trace_output(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    config = {
        "hero_deck": {"Mountain": 1, "Gray Ogre": 1},
        "villain_deck": {"Forest": 1, "Llanowar Elves": 1},
        "villain_type": "passive",
        "seed": 7,
    }

    seen_observation_messages = 0
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            payload = websocket.receive_json()
            assert isinstance(payload.get("session_id"), str)
            assert isinstance(payload.get("resume_token"), str)

            max_steps = 300
            while payload["type"] != "game_over":
                assert payload["type"] == "observation"
                seen_observation_messages += 1
                assert payload["data"]["agent"]["player_index"] == 0
                assert payload["actions"], (
                    "Expected hero actions when observation is emitted"
                )

                action_index = _pick_action(payload["actions"])
                websocket.send_json({"type": "action", "index": action_index})
                payload = websocket.receive_json()

                max_steps -= 1
                assert max_steps > 0, (
                    "Game did not complete within expected step budget"
                )

            assert payload["winner"] in {0, 1, None}

    assert seen_observation_messages > 0

    trace_files = sorted(tmp_path.glob("*.json"))
    assert len(trace_files) == 1

    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["end_reason"] == "game_over"
    assert isinstance(trace_payload["events"], list)
    assert trace_payload["events"], "Trace should record hero and villain events"
    assert trace_payload["final_observation"]["game_over"] is True


def test_websocket_rejects_action_without_active_game(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "action", "index": 0})
            payload = websocket.receive_json()
            assert payload["type"] == "error"
            assert "No active game session" in payload["message"]


def test_websocket_can_resume_existing_session(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": {"seed": 5}})
            payload = websocket.receive_json()
            assert payload["type"] == "observation"
            session_id = payload["session_id"]
            resume_token = payload["resume_token"]

            first_action = _pick_action(payload["actions"])
            websocket.send_json({"type": "action", "index": first_action})
            payload = websocket.receive_json()
            assert payload["type"] == "observation"
            frame_before_reconnect = payload["recovery"]["frame"]

        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": session_id,
                    "resume_token": resume_token,
                }
            )
            resumed_payload = resumed.receive_json()
            assert resumed_payload["type"] == "observation"
            assert resumed_payload["session_id"] == session_id
            assert resumed_payload["resume_token"] == resume_token
            assert resumed_payload["recovery"]["reason"] == "reconnect"
            resumed_frame = resumed_payload["recovery"]["frame"]
            for key in ("match_id", "revision", "prompt"):
                assert resumed_frame[key] == frame_before_reconnect[key]


def test_websocket_rejects_invalid_resume_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game"})
            payload = websocket.receive_json()
            session_id = payload["session_id"]

        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": session_id,
                    "resume_token": "not-valid",
                }
            )
            error_payload = resumed.receive_json()
            assert error_payload["type"] == "error"
            assert "Invalid resume credentials" in error_payload["message"]


def test_websocket_expired_session_requires_new_game(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    monkeypatch.setattr(server, "SESSION_TTL", timedelta(seconds=0))
    server.SESSION_REGISTRY.clear()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game"})
            payload = websocket.receive_json()
            session_id = payload["session_id"]
            resume_token = payload["resume_token"]

        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": session_id,
                    "resume_token": resume_token,
                }
            )
            error_payload = resumed.receive_json()
            assert error_payload["type"] == "error"
            assert "expired" in error_payload["message"].lower()

    trace_files = sorted(tmp_path.glob("*.json"))
    assert len(trace_files) == 1
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["end_reason"] == server.SESSION_EXPIRED_END_REASON


def test_wire_message_includes_pending_villain_log_on_observation(monkeypatch):
    session = server.GameSession()
    session.obs = SimpleNamespace(
        game_over=False,
        action_space=SimpleNamespace(action_space_type=1),
    )
    session.trace = trace_store.Trace(
        config=trace_store.GameConfig(
            hero_deck={}, villain_deck={}, villain_type="passive"
        ),
        events=[],
        final_observation={},
        winner=None,
        end_reason="disconnect",
        timestamp="2026-03-06T00:00:00+00:00",
    )
    session._pending_villain_log = ["Villain: Pass priority"]

    monkeypatch.setattr(server, "hero_view", lambda obs: {"game_over": False})
    monkeypatch.setattr(
        server,
        "describe_actions",
        lambda obs: [{"index": 0, "description": "Pass priority"}],
    )

    payload = session._wire_message()

    assert payload["type"] == "observation"
    assert payload["log"] == ["Villain: Pass priority"]
    assert session._pending_villain_log == []


def test_wire_message_includes_pending_villain_log_on_game_over(monkeypatch):
    session = server.GameSession()
    session.obs = SimpleNamespace(game_over=True)
    session.trace = trace_store.Trace(
        config=trace_store.GameConfig(
            hero_deck={}, villain_deck={}, villain_type="passive"
        ),
        events=[],
        final_observation={},
        winner=None,
        end_reason="disconnect",
        timestamp="2026-03-06T00:00:00+00:00",
    )
    session._pending_villain_log = ["Villain: Attack with Gray Ogre"]

    monkeypatch.setattr(server, "hero_view", lambda obs: {"game_over": True})
    monkeypatch.setattr(server, "_winner_for_hero", lambda obs: 1)
    monkeypatch.setattr(session, "_finalize_trace", lambda end_reason: None)

    payload = session._wire_message()

    assert payload["type"] == "game_over"
    assert payload["winner"] == 1
    assert payload["log"] == ["Villain: Attack with Gray Ogre"]
    assert session._pending_villain_log == []
