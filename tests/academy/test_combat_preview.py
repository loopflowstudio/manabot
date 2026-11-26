"""A whole drafted attack is previewed read-only and declared one creature at a time."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame
from tests.academy.socket_messages import settled_message


def _command(
    game: server.GameSession, frame: ExperienceFrame, offer_id: int
) -> ExperienceFrame:
    assert frame.prompt is not None
    outcome = game.hero_command(
        {
            "command_id": f"combat-{frame.prompt.id}",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "offer_id": offer_id,
            "answers": [],
        }
    )
    assert outcome["status"] == "accepted", outcome
    return ExperienceFrame.model_validate(outcome["update"]["frame"])


def test_a_drafted_attack_is_previewed_without_commitment_and_declared_as_planned(
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
                        "hero_deck": {"Mountain": 30, "Gray Ogre": 10},
                        "villain_deck": {"Forest": 30, "Llanowar Elves": 10},
                        "villain_type": "passive",
                        "seed": 11,
                        "auto_pass": False,
                    },
                }
            )
            initial = settled_message(websocket)
            record = server.LIVE_GAMES[initial["game"]]
            game = record.game
            assert game.env is not None
            frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
            while keep := next(
                (o for o in frame.offers if o.label == "Keep hand"), None
            ):
                frame = _command(game, frame, keep.id)
            for _ in range(3):
                game.env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
            game.obs = game.env.scenario_refresh()
            game.published_prompt = None
            frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
            while frame.action_space != "DECLARE_ATTACKER":
                passing = next(o for o in frame.offers if o.verb == "pass_priority")
                frame = _command(game, frame, passing.id)
            assert frame.prompt is not None

            endpoint = f"/api/games/{initial['game']}/combat-preview"
            address = {
                "match_id": frame.match_id,
                "revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "frame_hash": frame.frame_hash,
            }
            empty = {**address, "plan": {"attackers": [], "blocks": []}}
            ask = client.post(endpoint, json=empty)
            assert ask.status_code == 200, ask.text
            declarable = ask.json()["preview"]["declarable"]
            ogres = [creature["creature"] for creature in declarable]
            assert [creature["label"] for creature in declarable] == ["Gray Ogre"] * 3

            before = game.env.search_witness_json()
            prompts = game._next_prompt_id
            decisions = len(game.canonical_decisions)
            plan = {**address, "plan": {"attackers": ogres[:2], "blocks": []}}
            response = client.post(endpoint, json=plan)
            assert response.status_code == 200, response.text
            preview = response.json()["preview"]
            assert preview["refused"] is None
            assert [step["attack"] for step in preview["steps"]] == [True, True, False]
            assert game.env.search_witness_json() == before
            assert game._next_prompt_id == prompts
            assert len(game.canonical_decisions) == decisions
            for change in ({"revision": frame.revision + 1}, {"frame_hash": "other"}):
                assert client.post(endpoint, json={**plan, **change}).status_code == 409

            # Each step's offer, sent at the frame it predicts, names its creature.
            for step in preview["steps"]:
                assert frame.action_space == "DECLARE_ATTACKER"
                offer = next(o for o in frame.offers if o.id == step["action_index"])
                assert offer.subjects is not None
                assert {s.object.display_id for s in offer.subjects} == {
                    step["creature"]["entity"]
                }
                frame = _command(game, frame, offer.id)
            # Two 2/2 attackers reached the unblocked passive opponent.
            assert frame.projection.opponent.life == 16, frame.projection.turn
            assert client.post(endpoint, json=plan).status_code == 409
