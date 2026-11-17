"""Live control Commands survive retry and reconnect as distinct recording steps."""

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession

import managym
from managym.academy import server, trace
from managym.academy.recording import Recording
from tests.academy.socket_messages import settled_message
from tests.fixtures.mana_controls_server import mana_game


def command(frame: dict[str, Any], offer: int, identity: str) -> dict[str, Any]:
    return {
        "command_id": identity,
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": offer,
        "answers": [],
    }


def submit(
    socket: WebSocketTestSession, cmd: dict[str, Any], grant: int | None = None
) -> dict[str, Any]:
    socket.send_json({"type": "command", "command": cmd, "grant_revision": grant})
    outcome = settled_message(socket)
    assert outcome.get("status") in {"accepted", "rejected", "duplicate"}, (
        outcome.get("type"),
        outcome.get("message"),
        outcome.get("error"),
    )
    return outcome


def test_float_and_undo_through_live_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    monkeypatch.setattr(server.GameSession, "new_game", mana_game)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {"type": "new_game", "config": {"mana_controls_fixture": True}}
            )
            start = settled_message(socket)
            frame = start["frame"]
            game = server.SESSION_REGISTRY[start["session_id"]].game
            assert game.env is not None
            before = game.env.state_digest()
            controls = start["mana_controls"]
            forest = next(
                o for o in controls["offers"] if o.get("output") == {"green": 1}
            )
            assert forest["source"]["id"]["entity"] == next(
                p["id"]
                for p in frame["projection"]["agent"]["battlefield"]
                if p["name"] == "Forest"
            )
            assert forest["id"] not in [o["id"] for o in frame["offers"]]
            for execute, offer_id in (
                (game.env.execute_semantic_command_json, forest["id"]),
                (game.env.execute_mana_control_json, frame["offers"][0]["id"]),
            ):
                with pytest.raises(managym.AgentError):
                    execute(
                        json.dumps(
                            {
                                "command_id": "wrong-set",
                                "expected_revision": controls["revision"],
                                "offer_id": offer_id,
                                "answers": [],
                            }
                        )
                    )
            tap = command(frame, forest["id"], "float")
            accepted = submit(socket, tap)["update"]
            assert accepted["log"] == ["Mana control: Forest tapped for {G}"]
            floated = accepted["frame"]
            assert floated["revision"] == frame["revision"] + 1
            assert floated["projection"]["agent"]["mana"][4] == 1
            undo = accepted["mana_controls"]
            assert undo["undo"]["reason"] is None
            assert submit(socket, tap)["status"] == "duplicate"
            for change, code in (
                ({}, "stale_revision"),
                ({"expected_revision": floated["revision"]}, "stale_prompt"),
            ):
                rejection = submit(socket, {**tap, **change, "command_id": code})
                assert rejection["rejection"]["code"] == code
                assert rejection["mana_controls"] == undo
                assert rejection["recovery"]["frame"] == floated
        with client.websocket_connect("/ws/play") as socket:
            socket.send_json(
                {
                    "type": "resume",
                    "session_id": start["session_id"],
                    "resume_token": start["resume_token"],
                }
            )
            restored = settled_message(socket)
            assert restored["mana_controls"] == undo
            assert restored["frame"] == floated
            undone = submit(socket, command(floated, undo["undo"]["offer"], "undo"))[
                "update"
            ]
            assert undone["log"] == ["Mana control: Undo: Forest tapped for {G}"]
            assert undone["mana_controls"]["undo"] is None
            assert undone["frame"]["revision"] == floated["revision"] + 1
            assert game.env.state_digest() == before
            # A following ordinary choice still joins to its own canonical row.
            floated_again = submit(
                socket, command(undone["frame"], forest["id"], "second-float")
            )["update"]
            current = floated_again["frame"]
            ordinary = next(
                o for o in current["offers"] if o["label"] == "Cast Gray Ogre"
            )
            pending = game._pending_decision
            assert pending is not None
            assert (
                submit(socket, command(current, ordinary["id"], "ordinary"))["status"]
                == "accepted"
            )
            branch = game.fork_study(pending.address.serialize())
            try:
                assert branch.current_projection()["agent"]["mana"][4] == 1
                with pytest.raises(managym.AgentError, match="disabled"):
                    pending.root.mana_controls_json()
            finally:
                branch.return_to_recorded()

        payload = trace.load_trace(game.match_id, tmp_path)
        recording = Recording(game.match_id, payload)
        transitions = [
            (i, row["control"])
            for i, row in enumerate(payload["events"])
            if row.get("control")
        ]
        assert [row["verb"] for _, row in transitions] == ["float", "undo", "float"]
        assert transitions[0][1]["state_before"] == before
        assert transitions[1][1]["reverses"] == "float"
        for index, row in transitions:
            assert row["receipt"]["command_id"] == row["native_command"]["command_id"]
            assert index not in recording.decisions
            assert (
                recording.read(f"after:{index}")["description"]
                == "Mana control: " + row["label"]
            )
        assert any(d.command_id == "ordinary" for d in recording.decisions.values())


def test_only_pilot_gets_controls_and_pooled_mana_survives_stop_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    monkeypatch.setattr(server.GameSession, "new_game", mana_game)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/ws/play") as pilot,
    ):
        pilot.send_json({"type": "new_game", "config": {"mana_controls_fixture": True}})
        initial = settled_message(pilot)
        invite = parse_qs(initial["table"]["watcher_invite"].lstrip("#"))
        with client.websocket_connect("/ws/play") as watcher:
            watcher.send_json(
                {
                    "type": "join_table",
                    "table_id": invite["table"][0],
                    "invite_token": invite["watch"][0],
                }
            )
            joined = settled_message(watcher)
            assert joined["type"] == "observation"
            assert "mana_controls" not in joined
            settled_message(pilot)
            forest = next(
                o
                for o in initial["mana_controls"]["offers"]
                if o.get("output") == {"green": 1}
            )
            grant = initial["table"]["access"]["grant_revision"]
            update = submit(
                pilot, command(initial["frame"], forest["id"], "tap"), grant
            )["update"]
            assert "mana_controls" not in settled_message(watcher)["update"]
            pilot.send_json(
                {
                    "type": "set_stops",
                    "grant_revision": grant,
                    "stops": {"my": [], "opponent": []},
                    "auto_pass": True,
                }
            )
            stopped = settled_message(pilot)
            assert stopped["frame"]["projection"]["agent"]["mana"][4] == 1
            assert stopped["mana_controls"] == update["mana_controls"]
            assert "mana_controls" not in settled_message(watcher)
