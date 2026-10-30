"""Academy lists trained checkpoints, refuses mismatches by name, and plays them."""

import hashlib
from pathlib import Path
import random

from fastapi.testclient import TestClient
import pytest
import torch

from academy import server, trace as trace_store
from academy.server import app
from manabot.model.world import WORLD
from tests.academy.test_play_modes import _recorded_traces, _write_tiny_checkpoint


@pytest.fixture()
def runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.delenv("ACADEMY_PLAY_CHECKPOINTS", raising=False)
    monkeypatch.setenv("MANABOT_RUNS_DIR", str(tmp_path / ".runs"))
    server.SESSION_REGISTRY.clear()
    # Catalog checkpoints sample their actions; fix the game's length.
    torch.manual_seed(0)
    current = tmp_path / ".runs" / "self-play" / "step_0.pt"
    _write_tiny_checkpoint(current)
    checkpoint = torch.load(current, map_location="cpu", weights_only=False)
    checkpoint["world_binding"]["world"] = "w5"
    (tmp_path / ".runs" / "older").mkdir()
    torch.save(checkpoint, tmp_path / ".runs" / "older" / "step_0.pt")
    return tmp_path / ".runs"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_game_error(client: TestClient, config: dict) -> str:
    with client.websocket_connect("/ws/play") as websocket:
        websocket.send_json({"type": "new_game", "config": config})
        payload = websocket.receive_json()
    assert payload["type"] == "error", payload
    return payload["message"]


def test_catalog_lists_admissible_and_refused_checkpoints(runs: Path) -> None:
    with TestClient(app) as client:
        catalog = client.get("/api/opponents").json()

    assert catalog["catalog_error"] is None
    entries = {entry["name"]: entry for entry in catalog["opponents"]}
    current = entries["self-play/step_0"]
    assert current["available"] and current["reason"] is None
    assert current["sha256"] == _sha(runs / "self-play" / "step_0.pt")
    assert current["world"] == WORLD
    assert current["source"] == "runs"
    assert current["model"].startswith("width 64")
    assert sorted(current["decks"]) == ["GW Allies", "UR Lessons"]
    older = entries["older/step_0"]
    assert not older["available"]
    assert older["reason"] == f"trained in w5, engine is {WORLD}"


def test_incompatible_checkpoint_is_refused_with_the_mismatch(runs: Path) -> None:
    with TestClient(app) as client:
        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _sha(runs / "older" / "step_0.pt"),
            },
        )
        assert f"trained in w5, engine is {WORLD}" in message

        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _sha(runs / "self-play" / "step_0.pt"),
                "hero_deck": "interactive",
                "villain_deck": "interactive",
            },
        )
        assert "was trained on UR Lessons vs GW Allies" in message


@pytest.mark.parametrize(
    ("hero_deck", "villain_deck"),
    [("ur_lessons", "gw_allies"), ("gw_allies", "ur_lessons")],
)
def test_chosen_checkpoint_plays_either_deck_and_is_recorded(
    runs: Path, hero_deck: str, villain_deck: str
) -> None:
    sha256 = _sha(runs / "self-play" / "step_0.pt")
    rng = random.Random(4)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "checkpoint",
                        "opponent_sha256": sha256,
                        "hero_deck": hero_deck,
                        "villain_deck": villain_deck,
                        "seed": 7,
                    },
                }
            )
            payload = websocket.receive_json()
            # Enough hero decisions for the bot to act from its own seat.
            for _ in range(40):
                if payload["type"] == "game_over":
                    break
                assert payload["type"] == "observation", payload
                actions = payload["actions"]
                choice = actions[rng.randrange(len(actions))]
                websocket.send_json({"type": "action", "index": choice["index"]})
                payload = websocket.receive_json()
    assert payload["table"]["opponent"]["sha256"] == sha256

    [(_, trace)] = _recorded_traces(trace_store.TRACES_DIR)
    config = trace["config"]
    assert config["villain_type"] == "checkpoint"
    assert config["opponent"]["sha256"] == sha256
    assert config["opponent_checkpoint"] == {
        "name": "self-play/step_0",
        "producer": "unknown",
        "sha256": sha256,
        "source": "runs",
        "world": WORLD,
        "model": config["opponent_checkpoint"]["model"],
        "step": None,
        "checkpoint": str((runs / "self-play" / "step_0.pt").resolve()),
        "deterministic": False,
    }


def test_replaced_checkpoint_cannot_replace_playable_match(runs: Path) -> None:
    checkpoint = runs / "self-play" / "step_0.pt"
    sha256 = _sha(checkpoint)
    game = server.GameSession(trace_dir=trace_store.TRACES_DIR)
    game.new_game({"villain_type": "passive", "seed": 79})
    before = game.current_recovery("explicit_resync")["frame"]
    checkpoint.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="no longer available"):
        game.new_game({"villain_type": "checkpoint", "opponent_sha256": sha256})
    assert game.current_recovery("explicit_resync")["frame"] == before


def _deck_import() -> dict[str, object]:
    return {
        "hero": {"source": "10 Forest\n10 Mountain\n2 Avatar Enthusiasts"},
        "villain": {"source": "20 Island"},
    }


def test_imported_decks_play_world_checkpoints_and_refuse_setup_bindings(
    runs: Path,
) -> None:
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.checkpoint import save_checkpoint

    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(semantic_pack="world"))
    world = runs / "world" / "step_0.pt"
    save_checkpoint(agent, space, world, player_configs=None)
    rng = random.Random(0)
    with TestClient(app) as client:
        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _sha(runs / "self-play" / "step_0.pt"),
                "deck_import": _deck_import(),
            },
        )
        assert "was trained on" in message
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "checkpoint",
                        "opponent_sha256": _sha(world),
                        "deck_import": _deck_import(),
                        "seed": 7,
                    },
                }
            )
            payload = websocket.receive_json()
            for _ in range(40):
                if payload["type"] == "game_over":
                    break
                assert payload["type"] == "observation", payload
                actions = payload["actions"]
                choice = actions[rng.randrange(len(actions))]
                websocket.send_json({"type": "action", "index": choice["index"]})
                payload = websocket.receive_json()
    [(_, trace)] = _recorded_traces(trace_store.TRACES_DIR)
    assert trace["config"]["opponent"]["sha256"] == _sha(world)
    villain_turns = [
        event for event in trace["events"] if event.get("actor") == "villain"
    ]
    assert villain_turns
