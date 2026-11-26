"""Game records: durability, safe replacement, History's listing and older databases."""

from dataclasses import replace
import json
import sqlite3

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace
from managym.academy.attempts import AttemptStore, RecordedPlayer
from tests.academy.socket_messages import settled_message


def rows(path):
    with sqlite3.connect(path / "play.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute("SELECT * FROM attempts ORDER BY started_at")
        ]


def test_replacement_preserves_authority_and_records_prefix(tmp_path):
    game = server.GameSession(trace_dir=tmp_path)
    initial = game.new_game({"villain_type": "random", "seed": 79})
    match = game.match_id
    seed = game.trace.config.seed
    with pytest.raises(ValueError):
        game.new_game({"villain_type": "checkpoint"})
    for invalid in ({"seed": -1}, {"hero_deck": {"Nonexistent card": 60}}):
        with pytest.raises(ValueError):
            game.new_game({"villain_type": "random", **invalid})
    assert game.match_id == match
    assert game.current_recovery("explicit_resync")["frame"] == initial["frame"]
    assert len(rows(tmp_path)) == 1
    game.new_game({"villain_type": "random"})
    assert game.match_id != match
    assert game.trace.config.seed != seed
    saved = rows(tmp_path)
    # Starting another game ends nothing: the first stays unfinished.
    assert saved[0]["status"] == "active"
    assert saved[0]["ending_reason"] is None
    assert saved[0]["winner"] is None
    assert saved[1]["status"] == "active"
    assert not list(tmp_path.glob("*.json"))
    game.close("expired")
    assert rows(tmp_path)[1]["ending_reason"] == "expired"


def test_an_unfinished_game_reads_without_credentials_and_survives_restart(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    game = server.GameSession(trace_dir=tmp_path)
    game.new_game({"villain_type": "random", "seed": 79})
    attempt = game.match_id
    client = TestClient(server.app)
    (listed,) = client.get("/api/traces").json()
    assert listed["id"] == attempt and listed["resumable"]
    assert [
        (p["player_id"], p["name"]) for p in listed["players"] if p["kind"] == "human"
    ] == [("human:local", "You")]
    assert not {"mine", "replay_available", "feedback", "play_session_id"} & set(listed)
    assert client.get(f"/api/traces/{attempt}/decisions").status_code == 200
    index = client.get(f"/api/traces/{attempt}/recording").json()
    assert index["status"] == "waiting" and index["resumable"]
    assert "can_feedback" not in index
    payload = client.get(f"/api/traces/{attempt}?reveal_hidden=true").json()
    assert "seed" not in payload["config"]
    assert "villain_checkpoint" not in payload["config"]
    assert "canonical_replay" not in payload
    assert payload["final_observation"]["opponent"]["hand"] == []
    assert payload["resumable"] and "can_feedback" not in payload
    assert client.post(
        f"/api/traces/{attempt}/feedback", json={"request_id": "n", "note": "x"}
    ).status_code in {404, 405}
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        db.execute("UPDATE attempts SET process_id='old-process'")
    restarted = AttemptStore(tmp_path / "play.sqlite")
    # Another process opening the store ends nothing either.
    assert restarted.load(attempt)["end_reason"] == "active"
    assert restarted.unfinished(attempt)
    saved = rows(tmp_path)[0]
    assert saved["status"] == "active"
    assert saved["winner"] is None and saved["ended_at"] is None
    assert json.loads(saved["trace_json"])["canonical_replay"]
    # A game dropped from memory is still in progress, not interrupted.
    game.release()
    index = client.get(f"/api/traces/{attempt}/recording").json()
    assert index["status"] == "waiting" and not index["can_study"]


def retained(*, ended=True, winner=0):
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


def test_history_lists_every_game_and_filters_by_bot_pairing_deck_and_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    store = AttemptStore(tmp_path / "play.sqlite")
    human = RecordedPlayer(0, "human:local", "human", "You", "ur_lessons")
    bot = RecordedPlayer(
        1, "bot:challenger", "bot", "Challenger", "gw_allies", "a" * 64, {}
    )
    other = replace(bot, player_id="bot:other", name="Other", version="b" * 64)
    for identifier, players, ended in [
        ("game-a", [human, bot], True),
        ("game-b", [replace(other, seat=0), replace(human, seat=1)], True),
        ("game-c", [human, bot], False),
        ("game-d", [replace(bot, seat=0), other], True),
    ]:
        store.save(
            identifier,
            5,
            {"content_digest": "world"},
            retained(ended=ended),
            trace_json=json.dumps(retained(ended=ended)),
            finished=ended,
            players=players,
            origin="automated_validation",
        )
    client = TestClient(server.app)

    def ids(**params):
        response = client.get("/api/traces", params=params)
        assert response.status_code == 200, response.text
        return [row["id"] for row in response.json()]

    assert ids() == ["game-d", "game-c", "game-b", "game-a"]
    assert {row["origin"] for row in client.get("/api/traces").json()} == {
        "automated_validation"
    }
    assert ids(bot="bot:challenger") == ["game-d", "game-c", "game-a"]
    assert ids(bot="bot:other") == ["game-d", "game-b"]
    # The person is not a bot to filter by.
    assert ids(bot="human:local") == []
    assert ids(q="challenger") == ["game-d", "game-c", "game-a"]
    assert ids(q="bbbbbbbb") == ["game-d", "game-b"]
    assert ids(pairing="bot-bot") == ["game-d"]
    assert ids(status="active") == ["game-c"]
    assert ids(status="stopped") == []
    assert ids(
        pairing="human-bot",
        deck="ur_lessons",
        status="completed",
        after="2026-09-25",
        before="2026-09-25",
    ) == ["game-b", "game-a"]
    assert ids(q="%") == []
    assert ids(q="' OR 1=1 --") == []
    assert [
        row["id"] for row in client.get("/api/traces").json() if row["resumable"]
    ] == ["game-c"]
    for refused in (
        "pairing=human-human",
        "status=interrupted",
        "cursor=bad!",
        "after=yesterday",
    ):
        assert client.get(f"/api/traces?{refused}").status_code == 400
    seen, cursor = [], None
    for _ in range(3):
        response = client.get(
            "/api/traces", params={"limit": 2, **({"cursor": cursor} if cursor else {})}
        )
        seen.extend(row["id"] for row in response.json())
        cursor = response.headers.get("x-etude-next-cursor")
        if not cursor:
            break
    assert seen == ids() and len(seen) == len(set(seen))
    # A game an earlier build closed at a restart is still in progress.
    with store.connect() as db:
        db.execute(
            "UPDATE attempts SET status='interrupted', ending_reason='server_restart' WHERE id='game-c'"
        )
    assert ids(status="active") == ["game-c"] and ids(status="stopped") == []


EARLIER_SCHEMA = """
    CREATE TABLE attempts (
        id TEXT PRIMARY KEY, participant_id TEXT NOT NULL,
        process_id TEXT NOT NULL, started_at TEXT NOT NULL,
        ended_at TEXT, status TEXT NOT NULL, ending_reason TEXT,
        deal_seed TEXT, winner INTEGER, revision INTEGER NOT NULL,
        world_json TEXT NOT NULL, trace_json TEXT NOT NULL,
        record_version INTEGER NOT NULL DEFAULT 2,
        visibility TEXT NOT NULL DEFAULT 'shared',
        origin TEXT NOT NULL DEFAULT 'unknown',
        num_events INTEGER NOT NULL DEFAULT 0, numbers_json TEXT);
    CREATE TABLE players (id TEXT PRIMARY KEY, credential_hash TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL);
    CREATE TABLE feedback (attempt_id TEXT NOT NULL, request_id TEXT NOT NULL,
        created_at TEXT NOT NULL, note TEXT NOT NULL, author_id TEXT,
        decision_address TEXT, PRIMARY KEY (attempt_id, request_id));
"""


def test_a_database_from_an_earlier_build_keeps_its_rows_and_takes_new_games(tmp_path):
    """The identity and feedback an earlier build stored stay, unread; a new
    game is saved beside them, and that build can still open the file."""
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        db.executescript(EARLIER_SCHEMA)
        db.execute("INSERT INTO players VALUES('human:abc','hash','Ada')")
        db.execute(
            "INSERT INTO feedback VALUES('old','note-1','2026-09-25','Kept.',NULL,NULL)"
        )
    game = server.GameSession(trace_dir=tmp_path)
    game.new_game({"villain_type": "random", "seed": 79})
    (saved,) = rows(tmp_path)
    assert saved["id"] == game.match_id and saved["visibility"] == "shared"
    assert saved["participant_id"] == ""
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        assert db.execute("SELECT name FROM players").fetchall() == [("Ada",)]
        assert db.execute("SELECT note FROM feedback").fetchall() == [("Kept.",)]
    game.close("expired")


def test_a_new_database_has_no_identity_feedback_or_session_tables(tmp_path):
    AttemptStore(tmp_path / "play.sqlite")
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
    assert tables == {"attempts", "attempt_players", "analyses", "analysed_decisions"}
    assert not {"participant_id", "visibility"} & columns


def test_migration_preserves_the_oldest_trace_and_unknown_provenance(tmp_path):
    path = tmp_path / "play.sqlite"
    payload = json.dumps(retained(ended=False))
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE attempts (
            id TEXT PRIMARY KEY,participant_id TEXT NOT NULL,process_id TEXT NOT NULL,
            started_at TEXT NOT NULL,ended_at TEXT,status TEXT NOT NULL,ending_reason TEXT,
            bot_sha256 TEXT,hero_deck TEXT,villain_deck TEXT,deal_seed INTEGER,winner INTEGER,
            revision INTEGER NOT NULL,world_json TEXT NOT NULL,trace_json TEXT NOT NULL);""")
        db.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                "owner",
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
    listed = store.summaries()
    assert listed[0]["status"] == "active" and listed[0]["resumable"]
    assert listed[0]["record_version"] == 1 and listed[0]["origin"] == "unknown"
    assert [p["name"] for p in listed[0]["players"]] == ["You", "Random"]
    with store.connect() as db:
        assert db.execute("SELECT trace_json FROM attempts").fetchone()[0] == payload
    assert AttemptStore(path).summaries() == listed


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
    # Projecting the saved state again writes nothing; the next move must save.
    game.current_recovery("explicit_resync")
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        game.hero_action(0)


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


def test_websocket_reports_a_recording_failure_and_keeps_the_game_in_play(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    monkeypatch.setattr(server, "LIVE_GAMES", {})

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {"type": "new_game", "config": {"villain_type": "passive"}}
            )
            first = settled_message(socket)
            assert first["type"] == "observation"
            monkeypatch.setattr(AttemptStore, "save", fail)
            socket.send_json(
                {"type": "new_game", "config": {"villain_type": "passive"}}
            )
            message = settled_message(socket)
            assert message["type"] == "error"
            assert "recording failed" in message["message"].lower()
            assert "not have been saved" in message["message"]
            # The game that failed to start is not live; the one in play is.
            assert list(server.LIVE_GAMES) == [first["game"]]


def test_live_saves_keep_the_whole_trace_as_one_encoding_would(tmp_path):
    """Each save re-encodes only new events; the stored record stays exact."""
    game = server.GameSession(trace_dir=tmp_path)
    game.new_game({"villain_type": "passive", "seed": 79, "auto_pass": False})
    for _ in range(12):
        frame = game.current_recovery("explicit_resync")["frame"]
        if frame["status"] != "ready":
            break
        offer = next(
            (o for o in frame["offers"] if o["verb"] in ("play", "cast")),
            next(
                (o for o in frame["offers"] if o["verb"] == "pass_priority"),
                frame["offers"][0],
            ),
        )
        outcome = game.hero_command(
            {
                "command_id": f"save-{frame['prompt']['id']}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted", outcome
        stored = json.loads(rows(tmp_path)[0]["trace_json"])
        events = json.loads(json.dumps(game.trace.to_record()["events"]))
        assert stored["events"] == events
        replay = game._canonical_replay_snapshot().model_dump(mode="json")
        assert stored["canonical_replay"] == json.loads(json.dumps(replay))
    assert any(event["presentation"] for event in stored["events"])
    assert len(stored["events"]) >= 12
    assert rows(tmp_path)[0]["num_events"] == len(stored["events"])


def test_trace_encoding_follows_appended_and_replaced_events():
    config = trace.GameConfig(
        hero_deck={"Island": 1}, villain_deck={"Island": 1}, villain_type="passive"
    )
    first = trace.TraceEvent("hero", {"turn": 1}, [], 0, "Pass", 0.0)
    record = trace.Trace(
        config, [first], {}, None, "active", "2026-11-18T00:00:00+00:00"
    )
    encoded: list[tuple[trace.TraceEvent, str]] = []
    assert record.to_json(encoded) == json.dumps(record.to_record())
    second = trace.TraceEvent("villain", {"turn": 1}, [], 1, "Cast", 0.0)
    record.events.append(second)
    record.events[0] = replace(first, presentation=[{"seq": 0}])
    assert record.to_json(encoded) == json.dumps(record.to_record())
    record.events.pop()
    assert record.to_json(encoded) == json.dumps(record.to_record())
