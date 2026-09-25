"""Shared game history is searchable by either seat without exposing private play."""

from dataclasses import replace
import json
import sqlite3

from fastapi.testclient import TestClient
import pytest

from etude import server, trace
from etude.attempts import AttemptStore, RecordedPlayer, participant_id

TOKEN_A = "a" * 36
TOKEN_B = "b" * 36


def retained(identifier, *, ended=True, winner=0):
    return {
        "config": {
            "seed": 79,
            "villain_type": "random",
            "hero_deck_name": "ur_lessons",
            "villain_deck_name": "gw_allies",
            "villain_checkpoint": "/private/model.pt",
        },
        "timestamp": "2026-09-25T02:00:00+00:00",
        "events": [],
        "end_reason": "game_over" if ended else "active",
        "winner": winner if ended else None,
        "final_observation": {
            "game_over": ended,
            "agent": {"player_index": 0, "hand": []},
            "opponent": {"player_index": 1, "hand": [{"name": "secret"}]},
        },
    }


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    store = AttemptStore(tmp_path / "play.sqlite")
    alice = store.profile(TOKEN_A, "Same name")
    bob = store.profile(TOKEN_B, "Same name")
    human_a = RecordedPlayer(0, alice["id"], "human", alice["name"], "ur_lessons")
    human_b = RecordedPlayer(0, bob["id"], "human", bob["name"], "ur_lessons")
    bot = RecordedPlayer(
        1,
        "bot:challenger",
        "bot",
        "Challenger",
        "gw_allies",
        "a" * 64,
        {"policy": "checkpoint"},
    )
    for identifier, players, ended in [
        ("game-a", [human_a, bot], True),
        (
            "game-b",
            [replace(bot, seat=0, version="b" * 64), replace(human_b, seat=1)],
            True,
        ),
        ("game-c", [human_b, bot], False),
        ("game-d", [replace(bot, seat=0), bot], True),
    ]:
        store.save(
            identifier,
            participant_id(identifier),
            5,
            {"content_digest": "world"},
            retained(identifier, ended=ended),
            finished=ended,
            players=players,
            origin="automated_validation",
        )
    return store, TestClient(server.app), human_a, human_b, bot


def test_shared_search_filters_either_seat_versions_pairs_and_pagination(history):
    _, client, alice, bob, bot = history

    def ids(**params):
        response = client.get("/api/traces", params={"scope": "all", **params})
        assert response.status_code == 200, response.text
        return [row["id"] for row in response.json()]

    assert ids() == ["game-d", "game-c", "game-b", "game-a"]
    assert ids(player=bot.player_id) == ids()
    assert ids(player=bot.player_id, other=bob.player_id) == ["game-c", "game-b"]
    assert ids(player=bob.player_id, other=bot.player_id) == ["game-c", "game-b"]
    assert ids(player=bot.player_id, other=bot.player_id) == ["game-d"]
    assert ids(q="same NAME") == ["game-c", "game-b", "game-a"]
    assert ids(player=alice.player_id) == ["game-a"]
    assert ids(version="b" * 64) == ["game-b"]
    assert ids(q="bbbbbbbb") == ["game-b"]
    assert ids(pairing="bot-bot") == ["game-d"]
    assert ids(pairing="human-human") == []
    assert ids(
        pairing="human-bot",
        deck="ur_lessons",
        status="completed",
        after="2026-09-25",
        before="2026-09-25",
    ) == ["game-b", "game-a"]
    assert ids(player=bob.player_id, result="loss") == ["game-b"]
    assert ids(result="draw") == []
    assert ids(q="%") == []
    assert ids(q="' OR 1=1 --") == []
    assert client.get("/api/traces?result=win").status_code == 400
    assert client.get("/api/traces?cursor=bad!").status_code == 400
    assert client.get("/api/traces?after=yesterday").status_code == 400
    seen, cursor = [], None
    for _ in range(3):
        response = client.get(
            "/api/traces",
            params={
                "scope": "all",
                "limit": 2,
                **({"cursor": cursor} if cursor else {}),
            },
        )
        seen.extend(row["id"] for row in response.json())
        cursor = response.headers.get("x-etude-next-cursor")
        if not cursor:
            break
    assert seen == ids() and len(seen) == len(set(seen))


def test_shared_listing_is_not_private_replay_or_feedback_authority(history):
    store, client, alice, _, _ = history
    headers = {"x-etude-player-token": TOKEN_A}
    body = {"request_id": "note", "note": "Awkward choice"}
    assert client.post(
        "/api/traces/game-a/feedback", json=body, headers=headers
    ).json() == {"saved": True}
    assert client.post(
        "/api/traces/game-a/feedback", json=body, headers=headers
    ).json() == {"saved": True}
    assert client.post("/api/traces/game-a/feedback", json=body).status_code == 403
    public = client.get("/api/traces?scope=all").json()
    assert all(not row["feedback"] for row in public)
    assert all(
        "participant_id" not in row and "world_json" not in row for row in public
    )
    mine = client.get("/api/traces?scope=mine", headers=headers).json()
    assert [r["id"] for r in mine] == ["game-a"]
    assert mine[0]["feedback"][0]["author_id"] == alice.player_id
    assert len(mine[0]["feedback"]) == 1
    replay = client.get("/api/traces/game-a").json()
    assert replay["can_feedback"] is False
    assert (
        "seed" not in replay["config"] and "villain_checkpoint" not in replay["config"]
    )
    assert replay["final_observation"]["opponent"]["hand"] == []
    assert client.get("/api/traces/game-c?reveal_hidden=true").status_code == 403
    assert client.get("/api/traces/game-c/decisions").status_code == 403
    assert (
        client.get(
            "/api/traces/game-c", headers={"x-etude-player-token": TOKEN_B}
        ).status_code
        == 200
    )
    with store.connect() as db:
        db.execute("UPDATE attempts SET visibility='private' WHERE id='game-a'")
    assert "game-a" not in [r["id"] for r in client.get("/api/traces?scope=all").json()]
    assert client.get("/api/traces/game-a").status_code == 403
    assert client.get("/api/traces/game-a", headers=headers).status_code == 200


def test_migration_preserves_old_private_trace_and_unknown_provenance(tmp_path):
    path = tmp_path / "play.sqlite"
    payload = json.dumps(retained("old", ended=False))
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE attempts (
            id TEXT PRIMARY KEY,participant_id TEXT NOT NULL,process_id TEXT NOT NULL,
            started_at TEXT NOT NULL,ended_at TEXT,status TEXT NOT NULL,ending_reason TEXT,
            bot_sha256 TEXT,hero_deck TEXT,villain_deck TEXT,deal_seed INTEGER,winner INTEGER,
            revision INTEGER NOT NULL,world_json TEXT NOT NULL,trace_json TEXT NOT NULL);
            CREATE TABLE feedback(attempt_id TEXT,request_id TEXT,created_at TEXT,note TEXT,
                PRIMARY KEY(attempt_id,request_id));""")
        db.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                participant_id("old-secret"),
                "old-process",
                "2026-09-24",
                None,
                "active",
                None,
                None,
                "ur_lessons",
                "gw_allies",
                79,
                None,
                3,
                "{}",
                payload,
            ),
        )
    store = AttemptStore(path)
    assert store.summaries(set(), scope="all") == []
    rows = store.summaries({participant_id("old-secret")})
    assert rows[0]["status"] == "interrupted"
    assert rows[0]["record_version"] == 1 and rows[0]["origin"] == "unknown"
    assert len(rows[0]["players"]) == 2
    with store.connect() as db:
        assert db.execute("SELECT trace_json FROM attempts").fetchone()[0] == payload
    again = AttemptStore(path)
    assert again.summaries({participant_id("old-secret")}) == rows


def test_browser_identity_survives_tables_and_rename_without_changing_game(history):
    store, client, alice, _, _ = history
    headers = {"x-etude-player-token": TOKEN_A}
    renamed = client.put("/api/player", json={"name": "Jack"}, headers=headers).json()
    assert renamed["id"] == alice.player_id
    assert client.get("/api/player", headers=headers).json() == renamed
    assert store.summaries(set(), token=TOKEN_A)[0]["players"][0]["name"] == "Same name"
    sessions = []
    for _ in range(2):
        game = server.GameSession(trace_dir=store.path.parent)
        game.new_game({"villain_type": "passive", "seed": 79, "player_token": TOKEN_A})
        sessions.append(game.match_id)
        game.new_game({"villain_type": "passive", "seed": 80})
        sessions.append(game.match_id)
        game.close("new_game")
    records = store.summaries(set(), token=TOKEN_A)
    for identifier in sessions:
        row = next(r for r in records if r["id"] == identifier)
        assert row["players"][0]["player_id"] == alice.player_id
        assert row["players"][0]["name"] == "Jack"
        assert row["status"] == "stopped" and row["winner"] is None
    assert (
        client.put("/api/player", json={"name": " "}, headers=headers).status_code
        == 400
    )
