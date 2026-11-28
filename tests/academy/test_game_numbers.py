"""The game in numbers: a played game's numbers."""

from pathlib import Path
import random
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace
from managym.academy.attempts import AttemptStore
from tests.academy.test_played_replay import BOLT, play


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
    return game


def test_a_finished_game_saves_its_numbers(tmp_path: Path) -> None:
    game = finish(tmp_path, 11)
    client = TestClient(server.app)
    (row,) = client.get("/api/traces").json()
    numbers = row["numbers"]
    final = game.attempt_store.load(game.match_id)["final_observation"]
    assert numbers["complete"] and numbers["turns"] == final["turn"]["turn_number"]
    life = {
        final["agent"]["player_index"]: final["agent"]["life"],
        final["opponent"]["player_index"]: final["opponent"]["life"],
    }
    for seat in numbers["seats"]:
        # Bolt v Bolt: every life point lost is damage.
        assert seat["damage_taken"] == 20 - life[seat["seat"]]
        assert seat["lands_played"] > 0 and seat["cards_drawn"] > 0
    assert sum(s["spells_cast"] for s in numbers["seats"]) > 0
    assert client.get(f"/api/traces/{game.match_id}/numbers").json() == {
        "numbers": numbers
    }
    # Review reads how long the game took from its recording index.
    index = client.get(f"/api/traces/{game.match_id}/recording").json()
    assert index["started_at"] == row["timestamp"]
    assert index["ended_at"] == row["ended_at"] is not None


def test_an_older_game_gets_its_numbers_on_first_read(tmp_path: Path) -> None:
    game = finish(tmp_path, 12)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (saved,) = db.execute("SELECT numbers_json FROM attempts").fetchone()
        db.execute("UPDATE attempts SET numbers_json=NULL")
    client = TestClient(server.app)
    assert client.get("/api/traces").json()[0]["numbers"] is None
    read = client.get(f"/api/traces/{game.match_id}/numbers").json()["numbers"]
    assert read is not None
    assert store.numbers(game.match_id) == ("completed", saved)


def test_a_game_from_another_card_set_has_no_numbers(tmp_path: Path) -> None:
    game = finish(tmp_path, 13)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        db.execute(
            "UPDATE attempts SET numbers_json=NULL, world_json=json_set(world_json,'$.content_digest',?)",
            ("0" * 64,),
        )
    client = TestClient(server.app)
    assert client.get(f"/api/traces/{game.match_id}/numbers").json() == {
        "numbers": None
    }
