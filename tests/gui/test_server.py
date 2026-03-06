"""
test_server.py
WebSocket integration tests for the GUI backend server.
"""

import json

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - fallback when fastapi is unavailable
    from gui._mini_fastapi import TestClient

# Local imports
from gui import trace as trace_store
from gui.server import app


def _pick_action(actions: list[dict]) -> int:
    for action in actions:
        if action["type"] in {
            "PRIORITY_PLAY_LAND",
            "PRIORITY_CAST_SPELL",
            "DECLARE_ATTACKER",
            "DECLARE_BLOCKER",
            "CHOOSE_TARGET",
        }:
            return int(action["index"])

    for action in actions:
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return int(action["index"])

    return int(actions[0]["index"])


def test_websocket_new_game_action_loop_and_trace_output(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)

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

            max_steps = 300
            while payload["type"] != "game_over":
                assert payload["type"] == "observation"
                seen_observation_messages += 1
                assert payload["data"]["agent"]["player_index"] == 0
                assert payload["actions"], "Expected hero actions when observation is emitted"

                action_index = _pick_action(payload["actions"])
                websocket.send_json({"type": "action", "index": action_index})
                payload = websocket.receive_json()

                max_steps -= 1
                assert max_steps > 0, "Game did not complete within expected step budget"

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

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "action", "index": 0})
            payload = websocket.receive_json()
            assert payload["type"] == "error"
            assert "No active game session" in payload["message"]
