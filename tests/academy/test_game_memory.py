"""Games held in memory: what leaves it, what a read never rebuilds, and what
a closed decklist shows before and after the game is over."""

from pathlib import Path
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession

from managym.academy import server, trace as trace_store
from tests.academy.socket_messages import settled_message
from tests.academy.test_play_modes import finish

ISLANDS = {"source": "8 Island", "filename": "blue.txt", "name": "Blue"}
MOUNTAINS = {"source": "8 Mountain", "filename": "red.txt", "name": "Red"}
SHORT_GAME: dict[str, Any] = {
    "villain_type": "passive",
    "seed": 3,
    "deck_import": {"hero": ISLANDS, "villain": MOUNTAINS},
}


@pytest.fixture()
def traces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    return tmp_path


def new_game(ws: WebSocketTestSession, **config: Any) -> dict[str, Any]:
    ws.send_json({"type": "new_game", "config": {**SHORT_GAME, **config}})
    return settled_message(ws)


def stored_revision(traces: Path, game: str) -> int:
    db = sqlite3.connect(traces / "play.sqlite")
    try:
        return db.execute(
            "SELECT revision FROM attempts WHERE id=?", (game,)
        ).fetchone()[0]
    finally:
        db.close()


def test_a_closed_deck_is_named_nowhere_until_the_game_is_over(traces: Path) -> None:
    def bot_deck(players: list[dict[str, Any]]) -> dict[str, Any]:
        return next(player for player in players if player["kind"] == "bot")

    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        start = new_game(ws, decklists="closed")
        game = start["game"]
        live = server.LIVE_GAMES[game]

        assert start["deck_names"] == {"hero": "Blue", "villain": "Closed decklist"}
        assert "villain_deck" not in start["setup"]
        (row,) = client.get("/api/traces").json()
        assert bot_deck(row["players"])["deck"] == "closed"
        assert client.get("/api/traces", params={"deck": "Red"}).json() == []
        assert len(client.get("/api/traces", params={"deck": "Blue"}).json()) == 1
        recording = client.get(f"/api/traces/{game}/recording").json()
        assert bot_deck(recording["players"])["deck"] == "closed"
        assert bot_deck(recording["players"])["deck_name"] == "Closed decklist"
        config = client.get(f"/api/traces/{game}").json()["config"]
        assert config["villain_deck_name"] == "closed"

        finish(live.game)

        over = live.game.current_message()
        assert over["deck_names"] == {"hero": "Blue", "villain": "Red"}
        assert server._reply(live, over)["setup"]["deck_import"]["villain"] == MOUNTAINS
        (row,) = client.get("/api/traces", params={"deck": "Red"}).json()
        assert row["status"] == "completed"
        assert bot_deck(row["players"])["deck"] == "Red"
        recording = client.get(f"/api/traces/{game}/recording").json()
        assert bot_deck(recording["players"])["deck_name"] == "Red"
        # The trace names the deck as its players do; the list stays hidden.
        trace = client.get(f"/api/traces/{game}").json()
        assert bot_deck(trace["players"])["deck"] == "Red"
        assert trace["config"]["villain_deck_name"] == "Red"
        assert trace["config"]["villain_deck"] == {}


def test_a_closed_game_taken_up_from_its_record_keeps_its_deck_names(
    traces: Path,
) -> None:
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as ws:
            game = new_game(ws, decklists="closed")["game"]
        server.evict_game(game, "session_expired")
        with client.websocket_connect("/ws/play") as ws:
            ws.send_json({"type": "resume", "game": game})
            assert settled_message(ws)["game"] == game
            session = server.LIVE_GAMES[game].game
            assert session.record_players is not None
            assert {player.deck for player in session.record_players} == {"Blue", "Red"}

            finish(session)

        (row,) = client.get("/api/traces", params={"deck": "Red"}).json()
        assert (row["id"], row["status"]) == (game, "completed")


def test_a_finished_game_refuses_a_stops_change(traces: Path) -> None:
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        game = new_game(ws)["game"]
        session = server.LIVE_GAMES[game].game
        finish(session)
        assert session.trace is not None
        stops = session.trace.config.stops
        revision = stored_revision(traces, game)

        ws.send_json({"type": "set_stops", "stops": {"my": [], "opponent": []}})
        refused = settled_message(ws)

        assert refused["type"] == "error"
        assert refused["message"] == "Game is already over. Start a new game."
        assert session.trace.config.stops == stops
        assert stored_revision(traces, game) == revision
        assert client.get(f"/api/traces/{game}").json()["config"]["stops"] == stops


def test_watch_starts_a_list_on_one_side_against_a_named_deck(traces: Path) -> None:
    with TestClient(server.app) as client:
        created = client.post(
            "/api/watch",
            json={
                "players": ["random", "passive"],
                "hero_deck": "kept:mine",
                "villain_deck": "gw_allies",
                "deck_import": {"hero": ISLANDS},
                "seed": 3,
            },
        )
        assert created.status_code == 200, created.text
        config = client.get(f"/api/traces/{created.json()['id']}").json()["config"]
        assert config["hero_deck_name"] == "Blue"
        assert client.post("/api/watch", json={"deck_import": {}}).status_code == 400


def test_a_database_without_the_disclosure_column_learns_it_from_its_traces(
    traces: Path,
) -> None:
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        closed = new_game(ws, decklists="closed")["game"]
        opened = new_game(ws)["game"]
        db = sqlite3.connect(traces / "play.sqlite")
        with db:
            db.execute("ALTER TABLE attempts DROP COLUMN decklists")
        db.close()

        decks = {
            row["id"]: next(p["deck"] for p in row["players"] if p["kind"] == "bot")
            for row in client.get("/api/traces").json()
        }
        assert decks == {closed: "closed", opened: "Red"}


def test_a_preview_never_rebuilds_a_game_that_left_memory(traces: Path) -> None:
    address = {"match_id": "m", "revision": 0, "prompt_id": 0, "frame_hash": "h"}
    requests = {
        "payment": {**address, "offer_id": 0},
        "combat": {**address, "plan": {"attackers": [], "blocks": []}},
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as ws:
            game = new_game(ws)["game"]
        server.evict_game(game, "session_expired")
        revision = stored_revision(traces, game)

        for kind, request in requests.items():
            stale = client.post(f"/api/games/{game}/{kind}-preview", json=request)
            assert (stale.status_code, stale.json()["detail"]) == (
                409,
                f"{kind}_preview_stale",
            )
            unknown = client.post(
                f"/api/games/no-such-game/{kind}-preview", json=request
            )
            assert unknown.status_code == 404
        assert game not in server.LIVE_GAMES
        assert stored_revision(traces, game) == revision


def test_idle_games_beyond_the_bound_leave_memory_and_stay_resumable(
    traces: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "IDLE_GAMES_KEPT", 2)
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        games = [new_game(ws, seed=seed)["game"] for seed in range(1, 5)]
        server._cleanup_expired_games()

        # The connection's game and the two most recently left stay.
        assert list(server.LIVE_GAMES) == games[1:]
        store = server.AttemptStore(traces / "play.sqlite")
        assert all(store.unfinished(game) for game in games)

        ws.send_json({"type": "resume", "game": games[0]})
        assert settled_message(ws)["game"] == games[0]
        assert games[0] in server.LIVE_GAMES
        assert server._REBUILDS == {}


def test_a_finished_game_is_never_taken_up_though_memory_holds_it(traces: Path) -> None:
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        game = new_game(ws)["game"]
        finish(server.LIVE_GAMES[game].game)
        with client.websocket_connect("/ws/play") as other:
            other.send_json({"type": "resume", "game": game})
            refused = settled_message(other)
        assert refused["type"] == "error"
        assert refused["message"].startswith("game_finished: ")
        # The game is still held: Study and Rematch read it.
        assert game in server.LIVE_GAMES
        ws.send_json(
            {"type": "rematch", "game": game, "intent": "rematch", "config": {}}
        )
        assert settled_message(ws)["game"] != game
