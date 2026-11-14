"""Force of Negation's exile-from-hand alternative survives transport recovery.

The hero casts Lightning Bolt in the passive villain's upkeep and responds with
Force of Negation. With its mana cost payable the hero chooses the alternative,
then chooses which blue card to exile. Reconnect restores that payment prompt;
the native vector replay, a fork and the live session agree; a repeated
Command is a duplicate and an earlier revision's Command is stale.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym
from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame, InteractionOffer


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


def _offer(frame: ExperienceFrame, prefix: str) -> InteractionOffer | None:
    return next((o for o in frame.offers if o.label.startswith(prefix)), None)


def test_force_of_negation_exile_payment_reconnect_vector_and_fork(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    seed = 165
    config = {
        "hero_deck": {
            "Island": 14,
            "Mountain": 8,
            "Lightning Bolt": 6,
            "Force of Negation": 6,
            "Counterspell": 6,
        },
        "villain_deck": {"Plains": 40},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": seed,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:

            def send(offer: InteractionOffer, identity: str) -> ExperienceFrame:
                socket.send_json(
                    {"type": "command", "command": _command(frame, offer, identity)}
                )
                result = socket.receive_json()
                assert result["status"] == "accepted"
                return ExperienceFrame.model_validate(result["update"]["frame"])

            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            bolted = False
            # Reach the response with ordinary Commands; nothing is injected.
            for ordinal in range(200):
                assert game.obs is not None and frame.offers
                offer = _offer(frame, "Play Island") or _offer(frame, "Play Mountain")
                theirs = not game.obs.agent.is_active
                if offer is None and bolted:
                    offer = _offer(frame, "Cast Force of Negation")
                    if offer is not None:
                        frame = send(offer, "force")
                        break
                if (
                    offer is None
                    and theirs
                    and len(game.obs.agent_permanents) >= 4
                    and not bolted
                ):
                    offer = _offer(frame, "Cast Lightning Bolt")
                    bolted = offer is not None
                offer = (
                    offer
                    or _offer(frame, "Target Villain")
                    or _offer(frame, "Pass priority")
                    or _offer(frame, "Do not attack")
                    or frame.offers[0]
                )
                frame = send(offer, f"prefix.{ordinal}")
            else:
                pytest.fail("Force of Negation was not offered in response")

            # Both costs payable: the alternative is a deliberate choice.
            assert frame.action_space == "PAY_OR_NOT"
            assert {o.label for o in frame.offers} == {
                "Exile a card from your hand instead of paying the mana cost",
                "Pay the mana cost",
            }
            alternative = _offer(frame, "Exile a card")
            assert alternative is not None
            frame = send(alternative, "alternative")

        assert frame.action_space == "LOOK_AND_SELECT"
        assert frame.offers and all(
            o.label in {"Exile Counterspell to pay", "Exile Force of Negation to pay"}
            for o in frame.offers
        )
        assert game.env is not None and game.trace is not None and game.obs is not None
        hand = len([c for c in game.obs.agent_cards if c.zone == managym.ZoneEnum.HAND])
        space = ObservationSpace(ObservationConfig(policy_input_version=9))
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
            assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
                game.env.semantic_decision_frame_json()
            )
            for key, expected in space.encode(game.obs).items():
                np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)

            command = _command(frame, frame.offers[0], "exile")
            contract = SemanticDecisionContract.from_env(fork)
            contract.apply(fork, contract.frame.offers[0]["id"], command_id="fork")
            assert game.env.state_digest() == digest
            socket.send_json({"type": "command", "command": command})
            result = socket.receive_json()
            assert result["status"] == "accepted"
            assert game.env.state_digest() == fork.state_digest()
            committed = game.env.state_digest()
            events = len(game.trace.events)
            socket.send_json({"type": "command", "command": command})
            assert socket.receive_json()["status"] == "duplicate"
            socket.send_json(
                {"type": "command", "command": command | {"command_id": "stale"}}
            )
            rejected = socket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert game.env.state_digest() == committed
            assert len(game.trace.events) == events
        # The exiled card and Force of Negation both left the hand.
        assert game.obs is not None
        assert (
            len([c for c in game.obs.agent_cards if c.zone == managym.ZoneEnum.HAND])
            == hand - 2
        )
