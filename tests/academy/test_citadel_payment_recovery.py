"""A mana ability's damage suffix survives recovery at its mid-payment color choice.

Tarnished Citadel's any-color ability deals 3 damage to its controller as part
of the same mana ability. Activated while paying for Sol Ring, it suspends on a
five-color choice after the damage; reconnect, a fork and the native vector
replay must agree there, and neither a duplicate nor a stale Command may repeat
the damage or choose a color.
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

CITADEL = "Activate ability: Tarnished Citadel"


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


def test_citadel_color_choice_reconnect_vector_fork_and_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    seed = 286
    config = {
        "hero_deck": {"Forest": 32, "Tarnished Citadel": 4, "Sol Ring": 4},
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
            casting = False
            # Play real cards until Sol Ring's payment offers the colored ability.
            for ordinal in range(240):
                assert frame.offers, "game ended before the payment fixture"
                activation = next((o for o in frame.offers if o.label == CITADEL), None)
                if casting and activation is not None:
                    break
                labels = ["Play Tarnished Citadel", "Cast Sol Ring", "Play Forest"]
                offer = next(
                    (o for label in labels for o in frame.offers if o.label == label),
                    next(
                        (
                            o
                            for o in frame.offers
                            if o.label == "Pass priority"
                            or o.label.startswith("Do not attack")
                        ),
                        frame.offers[0],
                    ),
                )
                casting = offer.label == "Cast Sol Ring"
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
                pytest.fail(
                    "Sol Ring's payment with an untapped Citadel was not reached"
                )
            assert game.obs is not None
            life = game.obs.agent.life
            permanents = len(game.obs.agent_permanents)
            socket.send_json(
                {
                    "type": "command",
                    "command": _command(frame, activation, "activate"),
                }
            )
            result = socket.receive_json()
            assert result["status"] == "accepted"
            frame = ExperienceFrame.model_validate(result["update"]["frame"])

        assert frame.action_space == "MODAL"
        assert len(frame.offers) == 5
        assert game.obs is not None
        assert game.obs.agent.life == life - 3, (
            "the damage resolves once, before the color"
        )
        assert game.env is not None and game.trace is not None
        space = ObservationSpace(ObservationConfig())
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
            assert game.obs.agent.life == life - 3, "reconnect does not repeat damage"
            assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
                game.env.semantic_decision_frame_json()
            )
            for key, expected in space.encode(game.obs).items():
                np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)

            command = _command(frame, frame.offers[0], "color")
            contract = SemanticDecisionContract.from_env(fork)
            contract.apply(
                fork, contract.frame.offers[0]["id"], command_id="fork.color"
            )
            assert game.env.state_digest() == digest, "the fork is independent"
            before = len(game.trace.events)
            socket.send_json({"type": "command", "command": command})
            result = socket.receive_json()
            assert result["status"] == "accepted"
            # The server then plays the passive villain's decisions; the fork
            # takes the same recorded steps.
            for event in game.trace.events[before + 1 :]:
                fork.step(event.action)
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
                    "command": command | {"command_id": "stale.color"},
                }
            )
            rejected = socket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert game.env.state_digest() == committed
            assert len(game.trace.events) == events

        assert game.obs is not None
        assert game.obs.agent.life == life - 3
        assert len(game.obs.agent_permanents) == permanents + 1, "Sol Ring resolved"
        assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
            game.env.semantic_decision_frame_json()
        )
        for key, expected in space.encode(game.obs).items():
            np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
