"""The table's projection carries what the layered game says a permanent is."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame
from tests.academy.socket_messages import settled_message


def test_battlefield_permanents_carry_their_effective_keywords(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "hero_deck": {"Island": 36, "First-Time Flyer": 4},
                        "villain_deck": {"Plains": 40},
                        "villain_type": "passive",
                        "seed": 7,
                        "auto_pass": False,
                    },
                }
            )
            initial = settled_message(websocket)
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            assert game.env is not None
            game.env.scenario_force_battlefield(0, "First-Time Flyer", ready=True)
            game.env.scenario_force_battlefield(0, "Island", ready=True)
            game.obs = game.env.scenario_refresh()
            game.published_prompt = None
            frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    permanents = {
        permanent.name: permanent for permanent in frame.projection.agent.battlefield
    }
    assert "flying" in permanents["First-Time Flyer"].keywords
    assert permanents["Island"].keywords == []
