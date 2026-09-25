"""Human-play attempt durability, safe replacement, and record authorization."""

import json
import sqlite3

from fastapi.testclient import TestClient
import pytest

from etude import server, trace
from etude.attempts import AttemptStore, participant_id


def rows(path):
    with sqlite3.connect(path / "play.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute("SELECT * FROM attempts ORDER BY started_at")
        ]


def test_replacement_preserves_authority_and_records_prefix(tmp_path):
    game = server.GameSession(trace_dir=tmp_path)
    initial = game.new_game(
        {"villain_type": "random", "seed": 79, "request_id": "first"}
    )
    match = game.match_id
    seed = game.trace.config.seed
    with pytest.raises(ValueError):
        game.new_game({"villain_type": "checkpoint"})
    for invalid in ({"seed": -1}, {"hero_deck": {"Nonexistent card": 60}}):
        with pytest.raises(ValueError):
            game.new_game({"villain_type": "random", **invalid})
    assert game.match_id == match
    assert game.current_recovery("explicit_resync")["frame"] == initial["frame"]
    game.new_game({"villain_type": "random", "request_id": "first"})
    assert game.match_id == match
    assert len(rows(tmp_path)) == 1
    game.new_game({"villain_type": "random"})
    assert game.match_id != match
    assert game.trace.config.seed != seed
    saved = rows(tmp_path)
    assert saved[0]["status"] == "stopped"
    assert saved[0]["ending_reason"] == "new_game"
    assert saved[0]["winner"] is None
    assert saved[1]["status"] == "active"
    assert not list(tmp_path.glob("*.json"))
    game.close("expired")
    assert rows(tmp_path)[1]["ending_reason"] == "expired"


def test_private_replay_and_idempotent_feedback_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    game = server.GameSession(trace_dir=tmp_path)
    game.attempt_owner = participant_id("pilot-secret")
    game.new_game({"villain_type": "random", "seed": 79})
    attempt = game.match_id
    headers = {"x-etude-participant-tokens": "pilot-secret"}
    client = TestClient(server.app)
    assert client.get("/api/traces?scope=mine").json() == []
    assert client.get(f"/api/traces/{attempt}").status_code == 403
    assert client.get(f"/api/traces/{attempt}/decisions").status_code == 403
    payload = client.get(
        f"/api/traces/{attempt}?reveal_hidden=true", headers=headers
    ).json()
    assert "seed" not in payload["config"]
    assert "villain_checkpoint" not in payload["config"]
    assert "canonical_replay" not in payload
    assert payload["final_observation"]["opponent"]["hand"] == []
    body = {"request_id": "note-1", "note": "The block choice felt awkward."}
    assert client.post(f"/api/traces/{attempt}/feedback", json=body).status_code == 403
    for _ in range(2):
        assert client.post(
            f"/api/traces/{attempt}/feedback", json=body, headers=headers
        ).json() == {"saved": True}
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        assert db.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1
        db.execute("UPDATE attempts SET process_id='old-process'")
    restarted = AttemptStore(tmp_path / "play.sqlite")
    assert restarted.load(attempt)["end_reason"] == "server_restart"
    saved = rows(tmp_path)[0]
    assert saved["status"] == "interrupted"
    assert saved["winner"] is None and saved["ended_at"] is None
    assert json.loads(saved["trace_json"])["canonical_replay"]


def test_explicit_deal_reproduces_and_recording_failure_is_visible(
    tmp_path, monkeypatch
):
    game = server.GameSession(trace_dir=tmp_path)
    seed = 2**64 - 1
    first = game.new_game({"villain_type": "passive", "seed": seed})
    second = game.new_game({"villain_type": "passive", "seed": seed})
    assert first["data"] == second["data"]
    saved = rows(tmp_path)[0]
    assert int(saved["deal_seed"], 16) == seed
    assert json.loads(saved["trace_json"])["config"]["seed"] == seed

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(game.attempt_store, "save", fail)
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        game.current_recovery("explicit_resync")


def test_changed_candidate_cannot_replace_playable_match(tmp_path, monkeypatch):
    import hashlib

    from etude.opponent import Opponent

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"pinned bytes")
    opponent = Opponent(
        "Pinned opponent",
        "producer",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint,
        False,
        {},
        {},
    )
    monkeypatch.setattr(server, "configured_opponent", lambda: opponent)
    game = server.GameSession(trace_dir=tmp_path)
    game.new_game({"villain_type": "passive", "seed": 79})
    before = game.current_recovery("explicit_resync")["frame"]
    checkpoint.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="changed"):
        game.new_game(
            {"villain_type": "checkpoint", "opponent_sha256": opponent.sha256}
        )
    assert game.current_recovery("explicit_resync")["frame"] == before
    assert rows(tmp_path)[0]["status"] == "active"


def test_generated_deal_seed_reproduces_control_opponent_decisions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server.secrets, "randbits", lambda _bits: 79)
    generated = server.GameSession(trace_dir=tmp_path / "generated")
    generated.new_game({"villain_type": "random"})
    replayed = server.GameSession(trace_dir=tmp_path / "replayed")
    replayed.new_game({"villain_type": "random", "seed": generated.trace.config.seed})

    for _ in range(20):
        assert server.serialize_observation(
            generated.obs
        ) == server.serialize_observation(replayed.obs)
        assert [(event.actor, event.action) for event in generated.trace.events] == [
            (event.actor, event.action) for event in replayed.trace.events
        ]
        if generated.obs.game_over:
            break
        generated.hero_action(0)
        replayed.hero_action(0)
    generated.close("test")
    replayed.close("test")


def test_websocket_reports_recording_failure_even_when_close_also_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    monkeypatch.setattr(server, "SESSION_REGISTRY", {})

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {"type": "new_game", "config": {"villain_type": "passive"}}
            )
            assert socket.receive_json()["type"] == "observation"
            monkeypatch.setattr(AttemptStore, "save", fail)
            socket.send_json(
                {"type": "new_game", "config": {"villain_type": "passive"}}
            )
            message = socket.receive_json()
            assert message["type"] == "error"
            assert "recording failed" in message["message"].lower()
            assert "not have been saved" in message["message"]
    assert not server.SESSION_REGISTRY


def test_single_player_study_still_requires_participant_credential(tmp_path):
    record = server.SessionRecord(
        session_id="private-study",
        resume_token="pilot-secret",
        game=server.GameSession(trace_dir=tmp_path),
    )
    with pytest.raises(server.HTTPException) as error:
        server._rest_study_owner(record, None)
    assert error.value.status_code == 403
    assert server._rest_study_owner(record, "pilot-secret") is not None
