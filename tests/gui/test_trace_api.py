"""
test_trace_api.py
Integration tests for trace listing/loading replay endpoints.
"""

import json

from fastapi.testclient import TestClient

# Local imports
from gui import trace as trace_store
from gui.server import app


def _write_trace(path, trace_id: str, timestamp: str) -> None:
    payload = {
        "config": {
            "hero_deck": {"Mountain": 12},
            "villain_deck": {"Forest": 12},
            "villain_type": "passive",
            "seed": 1,
        },
        "events": [
            {
                "actor": "hero",
                "observation": {
                    "agent": {"hand": [{"name": "Mountain"}]},
                    "opponent": {"hand": [{"name": "Forest"}]},
                },
                "actions": [
                    {
                        "index": 0,
                        "type": "PRIORITY_PASS_PRIORITY",
                        "description": "Pass priority",
                    }
                ],
                "action": 0,
                "action_description": "Pass priority",
                "reward": 0.0,
            },
            {
                "actor": "villain",
                "observation": {
                    "agent": {"hand": [{"name": "Forest"}]},
                    "opponent": {"hand": [{"name": "Mountain"}]},
                },
                "actions": [
                    {
                        "index": 0,
                        "type": "PRIORITY_PASS_PRIORITY",
                        "description": "Pass priority",
                    }
                ],
                "action": 0,
                "action_description": "Pass priority",
                "reward": 0.0,
            },
        ],
        "final_observation": {"game_over": True, "agent": {"hand": []}},
        "winner": 0,
        "end_reason": "game_over",
        "timestamp": timestamp,
    }
    (path / f"{trace_id}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_trace_api_list_and_get_with_redaction(monkeypatch, tmp_path):
    _write_trace(tmp_path, "trace_b", "2026-03-05T12:00:00+00:00")
    _write_trace(tmp_path, "trace_a", "2026-03-05T13:00:00+00:00")

    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)

    with TestClient(app) as client:
        list_response = client.get("/api/traces")
        assert list_response.status_code == 200
        summaries = list_response.json()
        assert [entry["id"] for entry in summaries] == ["trace_a", "trace_b"]
        assert summaries[0]["num_events"] == 2

        redacted = client.get("/api/traces/trace_a")
        assert redacted.status_code == 200
        redacted_payload = redacted.json()

        hero_event = redacted_payload["events"][0]
        assert hero_event["observation"]["opponent"]["hand"] == []
        assert hero_event["observation"]["opponent"]["hand_hidden_count"] == 1

        villain_event = redacted_payload["events"][1]
        assert villain_event["observation"]["opponent"]["hand"] == []
        assert villain_event["observation"]["opponent"]["hand_hidden_count"] == 1

        revealed = client.get("/api/traces/trace_a?reveal_hidden=true")
        assert revealed.status_code == 200
        revealed_payload = revealed.json()
        assert revealed_payload["events"][0]["observation"]["opponent"]["hand"] == [
            {"name": "Forest"}
        ]


def test_trace_api_rejects_invalid_trace_id(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/traces/bad$id")
        assert response.status_code == 400
