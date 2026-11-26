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
    server.LIVE_GAMES.clear()
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
            game = server.LIVE_GAMES[initial["game"]].game
            assert game.env is not None and game.obs is not None
            while game.obs.turn.turn_number == 0:
                game.obs, *_ = game.env.step(0)
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


def test_the_spell_being_paid_for_is_announced_with_its_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Between Cast and its payment the spell is on the stack by the rules
    (CR 601.2a); the acting viewer's projection says so, with its target."""
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    config = {
        "hero_deck": {"Mountain": 30, "Lightning Bolt": 10},
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "seed": 11,
        "auto_pass": False,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            initial = settled_message(websocket)
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            seen: list[ExperienceFrame] = []
            for ordinal in range(120):
                prefixes = ("Play Mountain", "Cast Lightning Bolt", "Target Villain")
                offer = next(
                    (
                        o
                        for prefix in prefixes
                        for o in frame.offers
                        if o.label.startswith(prefix)
                    ),
                    None,
                )
                if frame.action_space == "PAY_OR_NOT":
                    seen.append(frame)
                    break
                assert frame.prompt is not None
                chosen = offer or frame.offers[0]
                websocket.send_json(
                    {
                        "type": "command",
                        "command": {
                            "command_id": f"c{ordinal}",
                            "match_id": frame.match_id,
                            "expected_revision": frame.revision,
                            "prompt_id": frame.prompt.id,
                            "offer_id": chosen.id,
                            "answers": [],
                        },
                    }
                )
                result = settled_message(websocket)
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
    assert seen, "no payment reached"
    context = seen[0].projection.decision_context
    assert context is not None and context.source is not None
    assert context.announcing == "spell"
    assert context.source.label == "Lightning Bolt"
    assert [target.object.label for target in context.targets] == ["Seat 2"]
    hand = {card.id for card in seen[0].projection.agent.hand}
    assert context.source.display_id in hand
