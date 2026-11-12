"""Counted Food payments survive transport recovery and native vector replay."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from managym.experience import ExperienceFrame, InteractionOffer
import numpy as np
import pytest

from academy import server, trace as trace_store
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import capture_text_decision
import managym


def _command(
    frame: ExperienceFrame, offer: InteractionOffer, identity: str
) -> dict[str, object]:
    assert frame.prompt is not None
    return {
        "command_id": identity,
        "match_id": frame.match_id,
        "expected_revision": frame.revision,
        "prompt_id": frame.prompt.id,
        "offer_id": offer.id,
        "answers": [],
    }


@pytest.mark.parametrize("version", [5, 6, 8, 9])
def test_merchant_payment_reconnect_vector_and_fork(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: int
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    seed = 165
    config = {
        "hero_deck": {
            "Forest": 16,
            "The Cabbage Merchant": 4,
            "Unlucky Cabbage Merchant": 20,
        },
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": seed,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            merchant_cast = False
            # Three Foods leave a real choice at both taps with forced skipping on.
            # Cast the real cards; no token deck entries or injected board state.
            for ordinal in range(180):
                assert game.obs is not None
                assert frame.offers, "game ended before the payment fixture"
                if sum(c.name == "Food" for c in game.obs.agent_cards) == 3:
                    activation = next(
                        (
                            o
                            for o in frame.offers
                            if o.label == "Activate ability: The Cabbage Merchant"
                        ),
                        None,
                    )
                    if activation is not None:
                        break
                labels = ["Play Forest"]
                if not merchant_cast:
                    labels.append("Cast The Cabbage Merchant")
                else:
                    labels.append("Cast Unlucky Cabbage Merchant")
                labels.append("Pass priority")
                offer = next(
                    (o for label in labels for o in frame.offers if o.label == label),
                    next(
                        (
                            o
                            for o in frame.offers
                            if o.action_type == "PAY_COST"
                            or o.label.startswith("Do not attack")
                        ),
                        frame.offers[0],
                    ),
                )
                merchant_cast |= offer.label == "Cast The Cabbage Merchant"
                socket.send_json(
                    {
                        "type": "command",
                        "command": _command(frame, offer, f"prefix.{ordinal}"),
                    }
                )
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail("Merchant and three Foods were not reached")
            socket.send_json(
                {
                    "type": "command",
                    "command": _command(frame, activation, "activate"),
                }
            )
            result = socket.receive_json()
            assert result["status"] == "accepted"
            frame = ExperienceFrame.model_validate(result["update"]["frame"])

        assert game.env is not None and game.trace is not None
        space = ObservationSpace(ObservationConfig(policy_input_version=version))
        vector = managym.VectorEnv(num_envs=1, seed=seed)
        buffers = space.encoder.allocate(1)
        buffers.update(
            rewards=np.zeros(1, dtype=np.float64),
            terminated=np.zeros(1, dtype=np.uint8),
            truncated=np.zeros(1, dtype=np.uint8),
        )
        vector.set_buffers(buffers)
        vector.reset_all_into_buffers(game.trace.config.to_rust())
        for event in game.trace.events:
            vector.step_into_buffers([event.action])
        fork = game.env.clone_env()
        foods: list[int] = []
        # Recover at each unpaid count, then at the output choice. A lost reply
        # must not repeat a tap or silently choose the output color.
        for remaining in (2, 1, 0):
            digest = game.env.state_digest()
            with client.websocket_connect("/ws/play") as socket:
                socket.send_json(
                    {
                        "type": "resume",
                        "session_id": initial["session_id"],
                        "resume_token": initial["resume_token"],
                    }
                )
                recovered = ExperienceFrame.model_validate(
                    socket.receive_json()["recovery"]["frame"]
                )
                assert recovered == frame
                assert game.env.state_digest() == digest == fork.state_digest()
                assert game.obs is not None
                assert json.loads(
                    vector.semantic_decision_frames_json()[0]
                ) == json.loads(game.env.semantic_decision_frame_json())
                for key, expected in space.encode(game.obs).items():
                    np.testing.assert_array_equal(
                        buffers[key][0], expected, err_msg=key
                    )
                if remaining:
                    assert game.obs.continuation[-1].mode == remaining
                    assert (
                        f"{remaining} tap selections still required"
                        in capture_text_decision(game.env).text
                    )
                    assert len(frame.offers) == remaining + 1
                    assert all("Tap Food to pay" in o.label for o in frame.offers)
                    foods.append(frame.offers[0].focus[0])
                else:
                    assert frame.action_space == "MODAL"
                    assert len(frame.offers) == 5
                    assert len(set(foods)) == 2
                before = len(game.trace.events)
                command = _command(frame, frame.offers[0], f"pay.{remaining}")
                contract = SemanticDecisionContract.from_env(fork)
                contract.apply(
                    fork, contract.frame.offers[0]["id"], command_id=f"fork.{remaining}"
                )
                assert game.env.state_digest() == digest
                socket.send_json({"type": "command", "command": command})
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
                assert game.env.state_digest() == fork.state_digest()
                for event in game.trace.events[before:]:
                    vector.step_into_buffers([event.action])
                committed = game.env.state_digest()
                events = len(game.trace.events)
                socket.send_json({"type": "command", "command": command})
                assert socket.receive_json()["status"] == "duplicate"
                socket.send_json(
                    {
                        "type": "command",
                        "command": command | {"command_id": f"stale.{remaining}"},
                    }
                )
                rejected = socket.receive_json()
                assert rejected["status"] == "rejected"
                assert rejected["rejection"]["code"] == "stale_revision"
                assert game.env.state_digest() == committed
                assert len(game.trace.events) == events
        assert game.obs is not None
        assert sum(game.obs.agent.mana) == 1
        food_keys = {c.registry_key for c in game.obs.agent_cards if c.name == "Food"}
        assert (
            sum(
                p.registry_key in food_keys and p.tapped
                for p in game.obs.agent_permanents
            )
            == 2
        )
        assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
            game.env.semantic_decision_frame_json()
        )
        for key, expected in space.encode(game.obs).items():
            np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
