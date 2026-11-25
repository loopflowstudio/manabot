"""The game in numbers: a played game's summary, and the record of My games."""

from pathlib import Path
import random
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace
from managym.academy.attempts import AttemptStore, participant_id
from managym.academy.summary import wilson_interval
from tests.academy.test_played_replay import BOLT, play

TOKEN = "s" * 36
HEADERS = {"x-etude-participant-tokens": TOKEN}


@pytest.fixture(autouse=True)
def traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    return tmp_path


def finish(tmp_path: Path, seed: int, **config: Any) -> server.GameSession:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": seed,
            "auto_pass": False,
            **config,
        }
    )
    play(game, random.Random(seed))
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        db.execute("UPDATE attempts SET participant_id=?", (participant_id(TOKEN),))
    return game


def test_a_finished_game_saves_its_numbers(tmp_path: Path) -> None:
    game = finish(tmp_path, 11)
    client = TestClient(server.app)
    (row,) = client.get("/api/traces?scope=mine", headers=HEADERS).json()
    summary = row["summary"]
    final = game.attempt_store.load(game.match_id)["final_observation"]
    assert summary["complete"] and summary["turns"] == final["turn"]["turn_number"]
    life = {
        final["agent"]["player_index"]: final["agent"]["life"],
        final["opponent"]["player_index"]: final["opponent"]["life"],
    }
    for seat in summary["seats"]:
        # Bolt v Bolt: every life point lost is damage.
        assert seat["damage_taken"] == 20 - life[seat["seat"]]
        assert seat["lands_played"] > 0 and seat["cards_drawn"] > 0
    assert sum(s["spells_cast"] for s in summary["seats"]) > 0
    assert client.get(f"/api/traces/{game.match_id}/summary").json() == {
        "summary": summary
    }


def test_an_older_game_gets_its_numbers_on_first_read(tmp_path: Path) -> None:
    game = finish(tmp_path, 12)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (saved,) = db.execute("SELECT summary_json FROM attempts").fetchone()
        db.execute("UPDATE attempts SET summary_json=NULL")
    client = TestClient(server.app)
    assert client.get("/api/traces?scope=mine", headers=HEADERS).json()[0][
        "summary"
    ] is None
    read = client.get(f"/api/traces/{game.match_id}/summary").json()["summary"]
    assert read is not None
    assert store.summary(game.match_id) == ("completed", saved)


def test_a_game_from_another_card_set_has_no_numbers(tmp_path: Path) -> None:
    game = finish(tmp_path, 13)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        db.execute(
            "UPDATE attempts SET summary_json=NULL, world_json=json_set(world_json,'$.content_digest',?)",
            ("0" * 64,),
        )
    client = TestClient(server.app)
    assert client.get(f"/api/traces/{game.match_id}/summary").json() == {
        "summary": None
    }


def test_my_record_counts_completed_games_by_opponent_and_decks(
    tmp_path: Path,
) -> None:
    winners = [finish(tmp_path, seed) for seed in (14, 15, 16)]
    # An abandoned game is not a result.
    abandoned = server.GameSession(tmp_path)
    abandoned.new_game(
        {"hero_deck": BOLT, "villain_deck": BOLT, "villain_type": "random"}
    )
    abandoned.new_game(
        {"hero_deck": BOLT, "villain_deck": BOLT, "villain_type": "random"}
    )
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        db.execute("UPDATE attempts SET participant_id=?", (participant_id(TOKEN),))
    client = TestClient(server.app)
    (row,) = client.get("/api/traces/record", headers=HEADERS).json()["rows"]
    wins = sum(game.attempt_store.load(game.match_id)["winner"] == 0 for game in winners)
    assert row["games"] == 3 and row["wins"] == wins
    assert row["wins"] + row["losses"] + row["draws"] == 3
    assert row["opponent_name"] == "Random"
    assert row["interval"] == list(wilson_interval(wins, 3))
    assert client.get("/api/traces/record").json() == {"rows": []}


def test_wilson_interval() -> None:
    low, high = wilson_interval(7, 10)
    assert round(low, 3) == 0.397 and round(high, 3) == 0.892
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 5)[0] == 0.0
