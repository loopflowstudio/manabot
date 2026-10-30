"""Real human Commands and ordinary agent actions choose opposite survival outcomes."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from academy import server, trace as trace_store
from academy.server import GameSession
from manabot.env.observation import ObservationSpace
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.trigger_order import DECK, SEED, configure_root, prefix_index, root
import managym


def _command(session: GameSession, index: int, identity: str) -> dict[str, object]:
    prompt = session._publish_current_prompt()
    assert prompt is not None
    return {
        "command_id": identity,
        "match_id": session.match_id,
        "expected_revision": prompt.revision,
        "prompt_id": prompt.prompt_id,
        "offer_id": index,
        "answers": [],
    }


def _session(path: Path) -> GameSession:
    session = GameSession(trace_dir=path)
    session.new_game(
        {
            "hero_deck": DECK,
            "villain_deck": {"Swamp": 40},
            "villain_type": "passive",
            "seed": SEED,
            "auto_pass": False,
        }
    )
    assert session.env is not None
    session.obs = configure_root(session.env)
    session.published_prompt = None
    session.presentation.reset()
    return session


def test_order_reconnect_recovers_bound_choices_without_automatic_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    original = GameSession.new_game

    def prepared(self: GameSession, config: object) -> dict[str, object]:
        original(
            self,
            {
                "hero_deck": DECK,
                "villain_deck": {"Swamp": 40},
                "villain_type": "passive",
                "seed": SEED,
                "auto_pass": False,
            },
        )
        assert self.env is not None
        self.obs = configure_root(self.env)
        self.published_prompt = None
        self.presentation.reset()
        return self._wire_message()

    monkeypatch.setattr(GameSession, "new_game", prepared)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": {}})
            initial = socket.receive_json()
            record = server.SESSION_REGISTRY[initial["session_id"]]
            for ordinal in range(8):
                assert record.game.obs is not None
                if (
                    record.game.obs.action_space.action_space_type
                    == managym.ActionSpaceEnum.ORDER_TRIGGERS
                ):
                    break
                command = _command(
                    record.game, prefix_index(record.game.obs), f"ws-prefix-{ordinal}"
                )
                socket.send_json({"type": "command", "command": command})
                assert socket.receive_json()["status"] == "accepted"
            assert record.game.env is not None
            digest = record.game.env.state_digest()
            command = _command(record.game, 0, "lost-order-response")
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            frame = socket.receive_json()["recovery"]["frame"]
            assert frame["action_space"] == "ORDER_TRIGGERS"
            assert len(frame["offers"]) == 2
            assert record.game.env.state_digest() == digest
            socket.send_json({"type": "command", "command": command})
            assert socket.receive_json()["status"] == "accepted"
            committed = record.game.env.state_digest()
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": initial["session_id"],
                    "resume_token": initial["resume_token"],
                }
            )
            recovered = socket.receive_json()["recovery"]["frame"]
            assert recovered["action_space"] != "ORDER_TRIGGERS"
            assert record.game.env.state_digest() == committed
            socket.send_json({"type": "command", "command": command})
            assert socket.receive_json()["status"] == "duplicate"
            assert record.game.env.state_digest() == committed


@pytest.mark.parametrize(
    "first,alive", [("Callous Inspector", True), ("Bastion of Remembrance", False)]
)
def test_survival_orders_cross_consumers(
    tmp_path: Path, first: str, alive: bool
) -> None:
    session = _session(tmp_path)
    env, obs = root()
    replay, _ = root()
    assert session.env is not None and session.obs is not None
    initial_digest = env.state_digest()
    assert session.env.state_digest() == initial_digest == replay.state_digest()
    evidence = json.loads(
        (
            Path(__file__).parents[2] / "academy/fixtures/trigger-order-w17.json"
        ).read_text()
    )
    capture = json.loads(env.capture_decision(0).to_json())
    assert initial_digest == evidence["root_digest"]
    assert capture["world_version"] == evidence["world_version"]
    assert capture["content_digest"] == evidence["content_digest"]
    assert DECK == evidence["hero_deck"] and SEED == evidence["seed"]

    for ordinal in range(8):
        if obs.action_space.action_space_type == managym.ActionSpaceEnum.ORDER_TRIGGERS:
            break
        index = prefix_index(obs)
        frame = SemanticDecisionContract.from_env(env).frame
        command = json.dumps(
            {
                "command_id": f"prefix-{ordinal}",
                "expected_revision": frame.revision,
                "offer_id": frame.offers[index]["id"],
                "answers": [],
            }
        )
        replay.execute_semantic_command_json(command)
        assert (
            session.hero_command(_command(session, index, f"prefix-{ordinal}"))[
                "status"
            ]
            == "accepted"
        )
        obs, _, done, _, _ = env.step(index)
        assert not done
        assert session.env.state_digest() == env.state_digest() == replay.state_digest()
    else:
        raise AssertionError("missing ordering decision")

    assert len(obs.action_space.actions) == 2
    assert all(
        action.action_type == managym.ActionEnum.CHOOSE_TRIGGER
        for action in obs.action_space.actions
    )
    labels = [action.description or "" for action in obs.action_space.actions]
    assert all(label.startswith("Put on stack next — ") for label in labels)
    inspector = next(label for label in labels if "Callous Inspector" in label)
    bastion = next(label for label in labels if "Bastion of Remembrance" in label)
    assert "Deal 1 damage to you. Create 1 Clue token" in inspector
    assert "Menace" not in inspector
    assert "Each opponent loses 1 life and you gain 1 life" in bastion
    assert "enters the battlefield" not in bastion
    session.auto_pass = True
    session._advance()
    assert session.env.state_digest() == env.state_digest()
    assert (
        session.obs.action_space.action_space_type
        == managym.ActionSpaceEnum.ORDER_TRIGGERS
    )
    session.auto_pass = False
    scalar = ObservationSpace().encode(obs)
    native = env.encode_observation(obs)
    for key in native:
        np.testing.assert_array_equal(native[key], scalar[key], err_msg=key)
    assert (
        obs.action_space.actions[0].continuation_index
        != obs.action_space.actions[1].continuation_index
    )
    other = env.observation_for_player(1)
    assert not other.action_space.actions
    assert all(frame.kind <= 3 for frame in other.continuation)
    text = capture_text_decision(env)
    assert text is not None
    assert all(label in text.text for label in labels)

    branch = env.clone_env()
    pending_digest = env.state_digest()
    pending_frame = env.semantic_decision_frame_json()
    index = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if first in (action.description or "")
    )
    frame = SemanticDecisionContract.from_env(env).frame
    command = json.dumps(
        {
            "command_id": "order",
            "expected_revision": frame.revision,
            "offer_id": frame.offers[index]["id"],
            "answers": [],
        }
    )
    replay.execute_semantic_command_json(command)
    human = _command(session, index, "order")
    accepted = session.hero_command(human)
    assert accepted["status"] == "accepted"
    obs, _, _, _, _ = env.step(index)
    assert session.env.state_digest() == env.state_digest() == replay.state_digest()
    stack_descriptions = {
        entry.description
        for entry in env.observation_for_player(0).stack_objects
        if entry.description is not None
    }
    # Native skipping may already have resolved what no player could answer.
    assert stack_descriptions <= {
        label.removeprefix("Put on stack next — ").split(" — event:")[0]
        for label in labels
    }
    if not env.is_game_over():
        stack_text = capture_text_decision(env)
        assert stack_text is not None
        assert all(description in stack_text.text for description in stack_descriptions)
    digest = session.env.state_digest()
    assert session.hero_command(human)["status"] == "duplicate"
    assert session.env.state_digest() == digest
    assert branch.state_digest() == pending_digest
    assert branch.semantic_decision_frame_json() == pending_frame
    with pytest.raises(Exception, match="stale|match is over"):
        env.execute_semantic_command_json(command)

    # Only pass priority until the two trigger resolutions establish the outcome.
    for ordinal in range(8):
        hero = env.observation_for_player(0)
        if env.is_game_over() or len(hero.stack_objects) == 1:
            break
        frame = SemanticDecisionContract.from_env(env).frame
        pass_offer = frame.find_verb("pass_priority")
        replay.execute_semantic_command_json(
            json.dumps(
                {
                    "command_id": f"pass-{ordinal}",
                    "expected_revision": frame.revision,
                    "offer_id": pass_offer["id"],
                    "answers": [],
                }
            )
        )
        obs, _, _, _, _ = env.step(
            next(
                i
                for i, action in enumerate(obs.action_space.actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            )
        )
    assert env.is_game_over() is not alive
    assert env.observation_for_player(0).agent.life == (1 if alive else 0)
    assert replay.state_digest() == env.state_digest()

    for ordinal in range(8):
        assert session.obs is not None
        if (
            session.env.is_game_over()
            or len(session.env.observation_for_player(0).stack_objects) == 1
        ):
            break
        index = next(
            i
            for i, action in enumerate(session.obs.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        assert (
            session.hero_command(_command(session, index, f"human-pass-{ordinal}"))[
                "status"
            ]
            == "accepted"
        )
    assert session.env.is_game_over() is not alive
    assert session.env.observation_for_player(0).agent.life == (1 if alive else 0)
    session.close("test")
