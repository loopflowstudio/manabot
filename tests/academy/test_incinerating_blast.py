"""Replay Blast's private discard choice at the point lethal damage is marked."""

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from manabot.env.observation import ObservationSpace, encode_observation
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import keep_opening_hands
import managym
from managym.academy import server, trace as trace_store
from managym.academy.server import GameSession
from managym.decision import Command, DecisionFrame
from tests.academy.socket_messages import settled_message


def pending_blast(hidden: str) -> managym.Env:
    deck = dict.fromkeys(
        ("Mountain", "Incinerating Blast", "Forest", "Island", "Plains", "Gray Ogre"),
        10,
    )
    env = managym.Env(seed=23)
    obs, _ = env.reset(
        managym.GameSetup(
            [managym.PlayerConfig(name, deck) for name in ("caster", "opponent")],
            decklists="closed",
        )
    )
    keep_opening_hands(env, obs)
    env.scenario_clear_hand(0)
    for name in ("Incinerating Blast", "Forest", hidden):
        env.scenario_force_card_in_hand(0, name)
    for _ in range(5):
        env.scenario_force_battlefield(0, "Mountain", ready=True)
    env.scenario_force_battlefield(1, "Gray Ogre", ready=True)
    obs = env.scenario_refresh()
    cast = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
    )
    obs, _, _, _, _ = env.step(cast)
    assert [p.damage for p in obs.opponent_permanents] == [6]
    return env


def test_blast_commands_complete_once_from_equal_restores(tmp_path: Path) -> None:
    env = pending_blast("Island")
    witness = env.search_witness_json()
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert {o["label"] for o in frame.offers} == {
        "Select Forest",
        "Select Island",
        "DeclineChoice",
    }
    position = env.snapshot()
    outcomes = set()
    for offer in frame.offers:
        command = Command("blast", frame.revision, offer["id"]).to_json()
        first, second = position.restore(), position.restore()
        transition = json.loads(first.execute_semantic_command_json(command))
        assert json.loads(second.execute_semantic_command_json(command)) == transition
        receipt = transition["receipt"]
        assert receipt["after_revision"] == receipt["before_revision"] + 1
        assert receipt["next_decision"] and receipt["events"]
        assert (
            first.semantic_decision_frame_json()
            == second.semantic_decision_frame_json()
        )
        committed = first.search_witness_json()
        assert second.search_witness_json() == committed
        outcomes.add(first.state_digest())
        discarded = offer["verb"] != "decline"
        expected = ["Incinerating Blast"]
        if discarded:
            expected.append(offer["label"].removeprefix("Select "))
        expected.sort()
        for viewer in (0, 1):
            obs = first.observation_for_player(viewer)
            caster_cards = obs.agent_cards if viewer == 0 else obs.opponent_cards
            graveyard = sorted(
                c.characteristics.name
                for c in caster_cards
                if c.zone == managym.ZoneEnum.GRAVEYARD
            )
            assert graveyard == expected
            projection = recovery_frame(first, viewer, tmp_path)["projection"]
            facts = projection["decision_context"]["recent_facts"]
            assert facts.count("Incinerating Blast finished resolving.") == 1
            side = projection["agent" if viewer == 0 else "opponent"]
            assert sorted(card["name"] for card in side["graveyard"]) == graveyard
            caster = obs.agent if viewer == 0 else obs.opponent
            assert caster.turn_history.cards_drawn == int(discarded)
            assert not obs.stack_objects
        with pytest.raises(managym.AgentError, match="stale_revision"):
            first.execute_semantic_command_json(command)
        assert first.search_witness_json() == committed
    assert len(outcomes) == 3
    assert env.search_witness_json() == witness


def recovery_frame(env: managym.Env, viewer: int, path: Path) -> dict[str, Any]:
    session = GameSession(trace_dir=path, id_factory=str)
    session.env = env
    assert env.current_agent_index() == 0
    session.obs = env.observation_for_player(0)
    session.human_seat = viewer
    return session.current_recovery("reconnect")["frame"]


def test_pending_blast_privacy_and_recovery(tmp_path: Path) -> None:
    space = ObservationSpace()
    names = ("Island", "Plains")
    roots = [pending_blast(name) for name in names]
    for env, name in zip(roots, names, strict=True):
        frame = recovery_frame(env, 0, tmp_path)
        assert frame == recovery_frame(env.snapshot().restore(), 0, tmp_path)
        assert {o["label"] for o in frame["offers"]} == {
            "Select Forest",
            f"Select {name}",
            "Decline",
        }
        text = capture_text_decision(env)
        assert len(text.actions) == 3
        assert any(f"Select {name}" in a.description for a in text.actions)
        with pytest.raises(managym.AgentError, match="no active decision"):
            env.capture_decision(1)

    a, b = roots
    assert a.semantic_observation_json(1) == b.semantic_observation_json(1)
    waiting = recovery_frame(a, 1, tmp_path)
    assert not waiting["offers"] and waiting["prompt"] is None
    assert waiting == recovery_frame(b, 1, tmp_path)
    caster_a, caster_b = (env.observation_for_player(0) for env in roots)
    assert caster_a.toJSON() != caster_b.toJSON()
    cards_a, cards_b = (
        encode_observation(space, obs)["semantic_cards"] for obs in (caster_a, caster_b)
    )
    assert not np.array_equal(cards_a, cards_b)

    opponent_a, opponent_b = (env.observation_for_player(1) for env in roots)
    assert opponent_a.toJSON() == opponent_b.toJSON()
    x, y = (encode_observation(space, obs) for obs in (opponent_a, opponent_b))
    assert x.keys() == y.keys()
    for key in x:
        np.testing.assert_array_equal(x[key], y[key], err_msg=key)


def test_reconnect_to_restored_blast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    position = pending_blast("Plains").snapshot()
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)

    def restored(session: GameSession, config: dict[str, Any]) -> dict[str, Any]:
        session.env = position.restore()
        session.obs = session.env.observation_for_player(0)
        return session._wire_message()

    monkeypatch.setattr(GameSession, "new_game", restored)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json({"type": "new_game", "config": {}})
            opened = settled_message(socket)
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": opened["session_id"],
                    "resume_token": opened["resume_token"],
                }
            )
            recovered = settled_message(socket)["recovery"]["frame"]
            assert recovered == opened["recovery"]["frame"]
            assert {o["label"] for o in recovered["offers"]} == {
                "Select Forest",
                "Select Plains",
                "Decline",
            }
