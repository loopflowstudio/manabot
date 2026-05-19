"""
test_server.py
WebSocket integration tests for the GUI backend server.
"""

from datetime import timedelta
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

# Local imports
from gui import server, trace as trace_store
from gui.server import app


def _pick_action(actions: list[dict]) -> int:
    preferred_action_types = {*server.ACTION_LABELS, "PRIORITY_PASS_PRIORITY"}
    for action in actions:
        if action["type"] in preferred_action_types:
            return int(action["index"])

    return int(actions[0]["index"])


def test_websocket_new_game_action_loop_and_trace_output(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    config = {
        "hero_deck": {"Mountain": 1, "Grey Ogre": 1},
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
            assert payload["type"] in {"observation", "game_over"}

        with client.websocket_connect("/ws/play") as resumed:
            resumed.send_json(
                {
                    "type": "resume",
                    "session_id": session_id,
                    "resume_token": resume_token,
                }
            )
            resumed_payload = resumed.receive_json()
            assert resumed_payload["type"] in {"observation", "game_over"}
            if resumed_payload["type"] == "observation":
                assert resumed_payload["session_id"] == session_id
                assert resumed_payload["resume_token"] == resume_token


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
    session.obs = SimpleNamespace(game_over=False)
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

    monkeypatch.setattr(
        server, "serialize_observation", lambda obs: {"game_over": False}
    )
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
    session._pending_villain_log = ["Villain: Attack with Grey Ogre"]

    monkeypatch.setattr(
        server, "serialize_observation", lambda obs: {"game_over": True}
    )
    monkeypatch.setattr(server, "_winner_for_hero", lambda obs: 1)
    monkeypatch.setattr(session, "_finalize_trace", lambda end_reason: None)

    payload = session._wire_message()

    assert payload["type"] == "game_over"
    assert payload["winner"] == 1
    assert payload["log"] == ["Villain: Attack with Grey Ogre"]
    assert session._pending_villain_log == []
