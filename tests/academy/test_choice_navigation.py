"""Retained navigation positions come from ordinary selected-matchup Commands."""

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest

from managym.academy import server, trace as trace_store
from managym.academy.server import GameSession
from managym.experience import ExperienceFrame
from tests.academy.socket_messages import settled_message


class PrefixStep(BaseModel):
    offer_id: int
    label: str


class NavigationPosition(BaseModel):
    config: dict[str, str | int | bool | dict[str, int]]
    prefix: list[PrefixStep]
    digest: str
    frame: ExperienceFrame


class NavigationFixture(BaseModel):
    positions: dict[str, NavigationPosition]


FIXTURE = (
    Path(__file__).parents[2]
    / "managym/academy/fixtures/choice-navigation-opening-v1.json"
)
POSITIONS = NavigationFixture.model_validate_json(FIXTURE.read_text()).positions


@pytest.mark.parametrize("family", list(POSITIONS))
def test_retained_choice_position(tmp_path: Path, family: str) -> None:
    # JSON is checked against the live producer, including full offers and projection.
    position = POSITIONS[family]
    session = GameSession(tmp_path)
    frame = session.new_game(position.config)["frame"]
    for ordinal, step in enumerate(position.prefix):
        offer = next(o for o in frame["offers"] if o["id"] == step.offer_id)
        assert offer["label"] == step.label
        result = session.hero_command(
            {
                "command_id": f"navigation.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
        frame = result["update"]["frame"]
    assert session.env is not None
    assert session.env.state_digest() == position.digest
    expected = position.frame.model_dump(mode="json", exclude_unset=True)
    for field in (
        "revision",
        "action_space",
        "offers",
        "projection",
        "content_hash",
        "asset_manifest_hash",
    ):
        assert frame[field] == expected[field], field

    # A stale click cannot mutate this retained position, even with a legal ID.
    result = session.hero_command(
        {
            "command_id": "navigation.stale",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"] - 1,
            "prompt_id": frame["prompt"]["id"],
            "offer_id": frame["offers"][0]["id"],
            "answers": [],
        }
    )
    assert result["status"] == "rejected"
    assert session.env.state_digest() == position.digest


@pytest.mark.parametrize("family", ["CHOOSE_TARGET", "DISCARD"])
def test_filtered_positions_show_the_filter(family: str) -> None:
    # choice-navigation.spec.ts filters these; ActionPanel shows the filter at 8 offers.
    assert len(POSITIONS[family].frame.offers) >= 8


def test_accumulate_wisdom_order_survives_reconnect_and_commits_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    position = POSITIONS["LEARN"]
    prefix = [
        *position.prefix,
        PrefixStep(offer_id=5, label="Take Accumulate Wisdom"),
        PrefixStep(offer_id=0, label="Cast Accumulate Wisdom"),
        PrefixStep(offer_id=4, label="Pay the cost"),
        PrefixStep(offer_id=0, label="Put Dragonfly Swarm into your hand"),
    ]
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": position.config})
            initial = settled_message(socket)
            frame = initial["frame"]
            for ordinal, step in enumerate(prefix):
                assert any(
                    offer["id"] == step.offer_id and offer["label"] == step.label
                    for offer in frame["offers"]
                )
                socket.send_json(
                    {
                        "type": "command",
                        "command": {
                            "command_id": f"bottom-prefix.{ordinal}",
                            "match_id": frame["match_id"],
                            "expected_revision": frame["revision"],
                            "prompt_id": frame["prompt"]["id"],
                            "offer_id": step.offer_id,
                            "answers": [],
                        },
                    }
                )
                result = settled_message(socket)
                assert result["status"] == "accepted"
                frame = result["update"]["frame"]
            assert len(frame["offers"]) == 2
            assert all(
                offer["label"].endswith(" next on the bottom")
                for offer in frame["offers"]
            )
            game = server.LIVE_GAMES[initial["game"]].game
            assert game.env is not None
            pending = frame["projection"]["decision_context"]["pending"]
            assert pending
            pending_digest = game.env.state_digest()
            command = {
                "command_id": "bottom-order",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][-1]["id"],
                "answers": [],
            }
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "game": initial["game"],
                }
            )
            recovered = settled_message(socket)["recovery"]["frame"]
            assert recovered["offers"] == frame["offers"]
            assert recovered["projection"]["decision_context"]["pending"] == pending
            assert game.env.state_digest() == pending_digest
            socket.send_json({"type": "command", "command": command})
            assert settled_message(socket)["status"] == "accepted"
            committed_digest = game.env.state_digest()
            assert committed_digest != pending_digest
            socket.send_json({"type": "command", "command": command})
            assert settled_message(socket)["status"] == "duplicate"
            assert game.env.state_digest() == committed_digest
