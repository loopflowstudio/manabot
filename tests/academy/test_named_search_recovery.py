"""Naming and the private named-card search survive reconnect.

Naming is a sequence of ordinary prompts: groups of sorted names, then names.
Reconnect restores the actual current prompt; earlier prompts' Commands are
stale; only the final name starts the private search.
"""

import json
from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from managym.experience import ExperienceFrame, InteractionOffer
from managym.experience_projection import decision_surface_view
import numpy as np
import pytest
import torch

from academy import server, trace as trace_store
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
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


def _toward(frame: ExperienceFrame, name: str) -> InteractionOffer:
    """The leaf offer naming `name`, or the group whose label span contains it."""
    for offer in frame.offers:
        if offer.label == f"Name {name}":
            return offer
        first, _, last = offer.label.removeprefix("Names ").partition(" through ")
        if offer.label.startswith("Names ") and first <= name <= last:
            return offer
    raise AssertionError(f"{name} is unreachable")


@pytest.mark.parametrize("case", ["installed", "large-search", "large-policy"])
def test_sozin_search_reconnect_retains_selection_visibility_and_stale_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: Literal["installed", "large-search", "large-policy"],
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    config = {
        "hero_deck": {"Swamp": 24, "The Rise of Sozin": 12},
        "villain_deck": {"Plains": 20, "Gray Ogre": 20},
        "villain_type": "passive",
        "auto_pass": False,
        "seed": 165,
        "decklists": "closed",
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": config})
            initial = socket.receive_json()
            frame = ExperienceFrame.model_validate(initial["recovery"]["frame"])
            cast = False
            # Reach chapter II using ordinary Commands, without injecting state.
            for ordinal in range(200):
                if any(o.label.startswith("Names ") for o in frame.offers):
                    break
                labels = ["Target Villain", "Play Swamp"]
                if not cast:
                    labels.append("Cast The Rise of Sozin")
                labels.extend(["Pass priority", "Decline"])
                offer = next(
                    (
                        offer
                        for label in labels
                        for offer in frame.offers
                        if offer.label == label
                    ),
                    frame.offers[0],
                )
                cast |= offer.label == "Cast The Rise of Sozin"
                socket.send_json(
                    {
                        "type": "command",
                        "command": _command(frame, offer, f"search-prefix.{ordinal}"),
                    }
                )
                result = socket.receive_json()
                assert result["status"] == "accepted"
                frame = ExperienceFrame.model_validate(result["update"]["frame"])
            else:
                pytest.fail("Sozin's naming decision was not reached")

            if case != "installed":
                # Inject a scenario root in native authority only. Rebuild the
                # server's cached prompt from that root before reconnecting.
                game = server.SESSION_REGISTRY[initial["session_id"]].game
                assert game.env is not None
                catalog = game.env.scenario_naming_vocabulary()
                game.obs = game.env.observation_for_player(0)
                game.published_prompt = None
                frame = ExperienceFrame.model_validate(game._experience_frame())
                assert len(catalog) > 512
                assert len(frame.offers) == -(-len(catalog) // managym.NAME_BRANCHING)
                assert frame.offers[0].label.startswith(f"Names {catalog[0].name} ")
                assert frame.offers[-1].label.endswith(f" through {catalog[-1].name}")
                assert (
                    next(i for i, c in enumerate(catalog) if c.name == "Gray Ogre")
                    > 511
                )
                assert any(c.name == "Fire Lord Sozin" for c in catalog)
                assert not decision_surface_view(game.env, 0)["agent"].get("inspected")

                if case == "large-policy":
                    assert game.trace is not None
                    monkeypatch.setattr(managym, "naming_catalog", lambda: catalog)
                    torch.set_num_threads(1)
                    torch.manual_seed(243)
                    space = ObservationSpace(ObservationConfig(policy_input_version=9))
                    vector = managym.VectorEnv(num_envs=1, seed=165)
                    vector.reset_all(game.trace.config.to_rust())
                    for event in game.trace.events:
                        vector.step([event.action])
                    buffers = space.encoder.allocate(1)
                    buffers.update(
                        rewards=np.zeros(1, dtype=np.float64),
                        terminated=np.zeros(1, dtype=np.uint8),
                        truncated=np.zeros(1, dtype=np.uint8),
                    )
                    vector.set_buffers(buffers)
                    vector.scenario_naming_vocabulary_into_buffers()
                    model = ActorCritic(
                        space, ModelConfig(hidden_dim=16, attention_on=False)
                    )
                    replay = game.env.clone_env()
                    trace_length = len(game.trace.events)
                    target = catalog[512].name

        def policy_choice(frame: ExperienceFrame) -> InteractionOffer:
            """The ordinary policy's choice at the current naming prompt.

            One supervised step toward the prompt's offer leading to row 512
            witnesses reachability; it is neither a strategic target nor a
            strength claim. Scalar and vector encodings must agree first.
            """
            assert game.env is not None and game.obs is not None
            values = space.encoder.allocate()
            game.env.encode_observation_into(game.obs, values)
            for key, expected in space.encode(game.obs).items():
                np.testing.assert_array_equal(values[key], expected, err_msg=key)
                np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
            assert json.loads(vector.semantic_decision_frames_json()[0]) == json.loads(
                game.env.semantic_decision_frame_json()
            )
            obs = {key: torch.from_numpy(value)[None] for key, value in values.items()}
            wanted = frame.offers.index(_toward(frame, target))
            for _ in range(20):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
                model.zero_grad()
                torch.nn.functional.cross_entropy(
                    model(obs)[0], torch.tensor([wanted])
                ).backward()
                optimizer.step()
                selected = int(model.get_action_and_value(obs, deterministic=True)[0])
                if selected == wanted:
                    return frame.offers[selected]
            raise AssertionError("policy did not reach the supervised offer")

        # Disconnect at each naming prompt: recovery restores the actual current
        # prompt and its exact Command binding, without granting search
        # inspection early. Choosing a group commits nothing.
        selections: list[int] = []
        while True:
            with client.websocket_connect("/ws/play") as socket:
                socket.send_json(
                    {
                        "type": "resume",
                        "session_id": initial["session_id"],
                        "resume_token": initial["resume_token"],
                    }
                )
                recovered_name = ExperienceFrame.model_validate(
                    socket.receive_json()["recovery"]["frame"]
                )
                assert recovered_name == frame
                game = server.SESSION_REGISTRY[initial["session_id"]].game
                assert game.env is not None
                assert not decision_surface_view(game.env, 0)["agent"].get("inspected")
                name_offer = (
                    policy_choice(frame)
                    if case == "large-policy"
                    else _toward(frame, "Gray Ogre")
                )
                selections.append(frame.offers.index(name_offer))
                if case == "large-policy":
                    vector.step_into_buffers([selections[-1]])
                name_command = _command(
                    frame, name_offer, f"name-card.{len(selections)}"
                )
                socket.send_json({"type": "command", "command": name_command})
                named = socket.receive_json()
                assert named["status"] == "accepted"
                frame = ExperienceFrame.model_validate(named["update"]["frame"])
                socket.send_json(
                    {
                        "type": "command",
                        "command": name_command | {"command_id": "name-stale"},
                    }
                )
                stale_name = socket.receive_json()
                assert stale_name["status"] == "rejected"
                assert stale_name["rejection"]["code"] == "stale_revision"
                if name_offer.label.startswith("Name "):
                    break
        assert len(selections) == 2
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            assert (
                ExperienceFrame.model_validate(
                    socket.receive_json()["recovery"]["frame"]
                )
                == frame
            )
            if case == "large-policy":
                assert (
                    game.env is not None
                    and game.trace is not None
                    and game.obs is not None
                )
                for selected in selections:
                    replay.step(selected)
                for event in game.trace.events[trace_length + len(selections) :]:
                    replay.step(event.action)
                    vector.step_into_buffers([event.action])
                assert replay.state_digest() == game.env.state_digest()
                assert json.loads(
                    vector.semantic_decision_frames_json()[0]
                ) == json.loads(game.env.semantic_decision_frame_json())
                for key, expected in space.encode(game.obs).items():
                    np.testing.assert_array_equal(
                        buffers[key][0], expected, err_msg=key
                    )
                recorded = game.trace.events[trace_length:][: len(selections)]
                assert [event.action for event in recorded] == selections
                assert recorded[-1].action_description == f"Name {target}"
                assert not decision_surface_view(game.env, 1)["agent"].get("inspected")
                return
            assert frame.action_space == "LOOK_AND_SELECT"
            choices = [o for o in frame.offers if o.action_type == "SELECT_CARD"]
            assert len(choices) > 4
            chosen = choices[0]
            command = _command(frame, chosen, "search-first")
            socket.send_json({"type": "command", "command": command})
            result = socket.receive_json()
            assert result["status"] == "accepted"
            pending = ExperienceFrame.model_validate(result["update"]["frame"])
            assert pending.action_space == "LOOK_AND_SELECT"
            assert [
                o.focus for o in pending.offers if o.action_type == "SELECT_CARD"
            ] == [o.focus for o in choices[1:]]
            game = server.SESSION_REGISTRY[initial["session_id"]].game
            assert game.env is not None
            assert game.obs is not None and game.trace is not None
            assert any(f.decision == 25 and f.mode == 3 for f in game.obs.continuation)
            digest = game.env.state_digest()
            events = len(game.trace.events)
            searched = decision_surface_view(game.env, 0)
            assert searched["agent"]["inspected"]
            assert chosen.focus[0] in {c["id"] for c in searched["opponent"]["exile"]}
            assert not searched["opponent"]["hand"]
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")

        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovery = socket.receive_json()["recovery"]
            recovered = ExperienceFrame.model_validate(recovery["frame"])
            assert recovered == pending
            assert game.env.state_digest() == digest
            assert decision_surface_view(game.env, 0) == searched
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")

            socket.send_json({"type": "command", "command": command})
            assert socket.receive_json()["status"] == "duplicate"
            socket.send_json(
                {
                    "type": "command",
                    "command": command | {"command_id": "search-stale"},
                }
            )
            rejected = socket.receive_json()
            assert rejected["status"] == "rejected"
            assert rejected["rejection"]["code"] == "stale_revision"
            assert game.env.state_digest() == digest
            assert len(game.trace.events) == events

            stop = next(
                o for o in recovered.offers if o.label == "Stop searching and shuffle"
            )
            socket.send_json(
                {"type": "command", "command": _command(recovered, stop, "search-stop")}
            )
            assert socket.receive_json()["status"] == "accepted"
            finished = decision_surface_view(game.env, 0)
            assert not finished["agent"].get("inspected")
            assert chosen.focus[0] in {c["id"] for c in finished["opponent"]["exile"]}
            assert not decision_surface_view(game.env, 1)["agent"].get("inspected")
