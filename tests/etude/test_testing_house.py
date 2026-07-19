"""Two-participant integration proof for the bounded testing-house table."""

from __future__ import annotations

from copy import deepcopy
import json

from fastapi.testclient import TestClient

from etude import server, trace as trace_store
from etude.authored_match_receipt import play_fixed_authored_match
from etude.server import app

CONFIG = {
    "hero_deck": {"Mountain": 12, "Lightning Bolt": 12},
    "villain_deck": {"Island": 20},
    "villain_type": "passive",
    "seed": 3,
    "auto_pass": False,
}


def _invite(table: dict) -> tuple[str, str]:
    fragment = table["watcher_invite"]
    assert fragment.startswith("#table=")
    table_part, token_part = fragment[1:].split("&")
    return table_part.split("=", 1)[1], token_part.split("=", 1)[1]


def _command(frame: dict, command_id: str, offer_index: int = 0) -> dict:
    return {
        "command_id": command_id,
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": frame["offers"][offer_index]["id"],
        "answers": [],
    }


def _command_for_verb(frame: dict, command_id: str, verb: str) -> dict:
    index = next(
        index for index, offer in enumerate(frame["offers"]) if offer["verb"] == verb
    )
    return _command(frame, command_id, index)


def _authority_fingerprint(record: server.SessionRecord) -> dict:
    game = record.game
    assert game.env is not None
    return {
        "state": game.env.state_digest(),
        "revision": game.revision,
        "trace_events": len(game.trace.events) if game.trace else 0,
        "canonical_decisions": len(game.canonical_decisions),
        "stops": deepcopy(game.stops),
        "stop_on_stack": game.stop_on_stack,
        "auto_pass": game.auto_pass,
        "table_revision": record.table_revision,
        "roles": {
            viewer_id: (participant.role.value, participant.grant_revision)
            for viewer_id, participant in record.participants.items()
        },
        "personal_beliefs": {
            viewer_id: deepcopy(participant.personal_beliefs)
            for viewer_id, participant in record.participants.items()
        },
        "shared_beliefs": deepcopy(record.shared_beliefs),
        "attempts": tuple(sorted(game._study_attempts)),
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_live_decision_summaries_match_full_replay_without_reconstruction(
    monkeypatch,
):
    game, _ = play_fixed_authored_match()
    record = server.SessionRecord(
        session_id="summary-characterization",
        resume_token="summary-characterization-token",
        game=game,
    )
    projection = server.project_replay(
        game.canonical_replay(), server.HERO_PLAYER_INDEX
    )
    addressed = server.projection_with_addresses(projection)
    expected = [
        {
            "address": decision["address"],
            "ordinal": decision["ordinal"],
            "revision": decision["revision"],
            "prompt_id": decision["prompt_id"],
            "offer_id": decision["offer_id"],
        }
        for decision in addressed["decisions"]
    ]
    assert len(game.canonical_decisions) == 132
    assert len(expected) == 55

    def forbidden(*_args, **_kwargs):
        raise AssertionError("table summaries reconstructed a canonical replay")

    monkeypatch.setattr(server.GameSession, "canonical_replay", forbidden)
    monkeypatch.setattr(server, "project_replay", forbidden)
    monkeypatch.setattr(server, "projection_with_addresses", forbidden)

    actual = server._live_decision_summaries(record)
    assert _canonical_bytes(actual) == _canonical_bytes(expected)


def _open_table(client: TestClient):
    pilot = client.websocket_connect("/ws/play")
    pilot_socket = pilot.__enter__()
    pilot_socket.send_json({"type": "new_game", "config": CONFIG})
    initial = pilot_socket.receive_json()
    table_id, invite_token = _invite(initial["table"])

    watcher = client.websocket_connect("/ws/play")
    watcher_socket = watcher.__enter__()
    watcher_socket.send_json(
        {
            "type": "join_table",
            "table_id": table_id,
            "invite_token": invite_token,
        }
    )
    joined = watcher_socket.receive_json()
    pilot_presence = pilot_socket.receive_json()
    assert joined["recovery"]["frame"] == pilot_presence["recovery"]["frame"]
    assert joined["table"]["access"]["role"] == "watcher"
    assert pilot_presence["table"]["access"]["role"] == "pilot"
    return pilot, pilot_socket, watcher, watcher_socket, initial, joined


def test_closed_dispatch_denies_every_watcher_live_mutation_without_state_change(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(app) as client:
        pilot, pilot_socket, watcher, watcher_socket, initial, joined = _open_table(
            client
        )
        try:
            record = server.SESSION_REGISTRY[initial["session_id"]]
            watcher_access = joined["table"]["access"]
            watcher_grant = watcher_access["grant_revision"]
            frame = joined["recovery"]["frame"]
            watcher_id = watcher_access["identity"]["viewer_id"]
            entered_game_methods: list[str] = []
            for method_name in (
                "hero_command",
                "hero_action",
                "pass_turn",
                "set_stops",
                "new_game",
            ):
                original = getattr(record.game, method_name)

                def guarded(*args, _name=method_name, _original=original, **kwargs):
                    entered_game_methods.append(_name)
                    return _original(*args, **kwargs)

                monkeypatch.setattr(record.game, method_name, guarded)
            mutations = [
                {
                    "type": "command",
                    "grant_revision": watcher_grant,
                    "command": _command(frame, "watcher-command"),
                },
                {
                    "type": "action",
                    "grant_revision": watcher_grant,
                    "index": 0,
                },
                {"type": "pass_turn", "grant_revision": watcher_grant},
                {
                    "type": "set_stops",
                    "grant_revision": watcher_grant,
                    "stops": {"my": [], "opponent": []},
                    "stop_on_stack": False,
                    "auto_pass": False,
                },
                {
                    "type": "new_game",
                    "grant_revision": watcher_grant,
                    "config": CONFIG,
                },
                {
                    "type": "rematch",
                    "grant_revision": watcher_grant,
                    "config": CONFIG,
                },
                {
                    "type": "transfer_pilot",
                    "grant_revision": watcher_grant,
                    "target_viewer_id": watcher_id,
                },
            ]
            for mutation in mutations:
                before = _authority_fingerprint(record)
                watcher_socket.send_json(mutation)
                denied = watcher_socket.receive_json()
                assert denied["type"] == "control_error"
                assert denied["code"] == "forbidden"
                assert _authority_fingerprint(record) == before
                assert entered_game_methods == []

            before = _authority_fingerprint(record)
            watcher_socket.send_json({"type": "chat", "message": "hello"})
            unsupported = watcher_socket.receive_json()
            assert unsupported == {
                "type": "control_error",
                "code": "unsupported_message",
                "message": "Unsupported message type: chat",
            }
            assert _authority_fingerprint(record) == before
            assert entered_game_methods == []
        finally:
            watcher.__exit__(None, None, None)
            pilot.__exit__(None, None, None)


def test_personal_belief_is_private_until_explicit_table_share(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(app) as client:
        pilot, pilot_socket, watcher, watcher_socket, _initial, joined = _open_table(
            client
        )
        try:
            grant = joined["table"]["access"]["grant_revision"]
            watcher_socket.send_json(
                {
                    "type": "author_belief",
                    "grant_revision": grant,
                    "scenario_id": "advice-scenario-a",
                }
            )
            personal = watcher_socket.receive_json()
            assert personal["type"] == "belief_changed"
            assert personal["belief"]["audience"] == {"kind": "personal"}
            assert personal["table"]["beliefs"] == [personal["belief"]]

            watcher_socket.send_json(
                {
                    "type": "share_belief",
                    "grant_revision": grant,
                    "belief_id": personal["belief"]["id"],
                }
            )
            pilot_shared = pilot_socket.receive_json()
            watcher_shared = watcher_socket.receive_json()
            for shared in (pilot_shared, watcher_shared):
                assert shared["type"] == "belief_changed"
                assert shared["belief"]["audience"]["kind"] == "table"
                assert shared["belief"]["provenance"]["shared_at_table_revision"]
                assert shared["table"]["beliefs"] == [shared["belief"]]
        finally:
            watcher.__exit__(None, None, None)
            pilot.__exit__(None, None, None)


def test_watcher_branch_is_owner_only_while_pilot_advances_live_match(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(app) as client:
        pilot, pilot_socket, watcher, watcher_socket, initial, joined = _open_table(
            client
        )
        try:
            pilot_grant = initial["table"]["access"]["grant_revision"]
            watcher_grant = joined["table"]["access"]["grant_revision"]
            frame = joined["recovery"]["frame"]
            pilot_socket.send_json(
                {
                    "type": "command",
                    "grant_revision": pilot_grant,
                    "command": _command_for_verb(frame, "pilot-first", "pass_priority"),
                }
            )
            pilot_update = pilot_socket.receive_json()
            watcher_update = watcher_socket.receive_json()
            assert pilot_update["update"]["frame"] == watcher_update["update"]["frame"]
            decision = watcher_update["table"]["decisions"][0]

            watcher_socket.send_json(
                {
                    "type": "restore_decision",
                    "grant_revision": watcher_grant,
                    "address": decision["address"],
                }
            )
            restored_event = watcher_socket.receive_json()
            restored = restored_event["restored"]
            retry_command = deepcopy(restored["command"])
            retry_command["command_id"] = "watcher-retry"
            record = server.SESSION_REGISTRY[initial["session_id"]]
            authority_before = _authority_fingerprint(record)
            watcher_socket.send_json(
                {
                    "type": "retry_decision",
                    "grant_revision": watcher_grant,
                    "address": decision["address"],
                    "command": retry_command,
                }
            )
            branch = watcher_socket.receive_json()
            assert branch["type"] == "branch_updated", branch
            assert branch["phase"] == "retry"
            authority_after = _authority_fingerprint(record)
            authority_after["attempts"] = authority_before["attempts"]
            assert authority_after == authority_before

            anonymous_return = client.post(
                f"/api/study-attempts/{branch['attempt_id']}/return"
            )
            assert anonymous_return.status_code == 403
            wrong_lease_return = client.post(
                f"/api/study-attempts/{branch['attempt_id']}/return",
                headers={"x-etude-participant-token": initial["resume_token"]},
            )
            assert wrong_lease_return.status_code == 404
            assert branch["attempt_id"] in record.game._study_attempts

            pilot_socket.send_json(
                {
                    "type": "return_from_branch",
                    "grant_revision": pilot_grant,
                    "attempt_id": branch["attempt_id"],
                }
            )
            wrong_owner = pilot_socket.receive_json()
            assert wrong_owner["type"] == "control_error"
            assert wrong_owner["code"] == "not_found"
            assert branch["attempt_id"] in record.game._study_attempts

            live_frame = pilot_update["update"]["frame"]
            pilot_socket.send_json(
                {
                    "type": "command",
                    "grant_revision": pilot_grant,
                    "command": _command(live_frame, "pilot-second"),
                }
            )
            next_pilot = pilot_socket.receive_json()
            assert next_pilot["type"] == "command_outcome"
            assert branch["attempt_id"] not in str(next_pilot)
            next_watcher = watcher_socket.receive_json()
            assert next_watcher["type"] == "command_outcome"

            watcher_socket.send_json(
                {
                    "type": "return_from_branch",
                    "grant_revision": watcher_grant,
                    "attempt_id": branch["attempt_id"],
                }
            )
            returned = watcher_socket.receive_json()
            assert returned["type"] == "branch_returned"
            assert returned["restored"] == restored
        finally:
            watcher.__exit__(None, None, None)
            pilot.__exit__(None, None, None)


def test_role_transfer_rejects_old_grant_and_preserves_both_leases(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(app) as client:
        pilot, pilot_socket, watcher, watcher_socket, initial, joined = _open_table(
            client
        )
        try:
            old_pilot_token = initial["resume_token"]
            old_pilot_id = initial["table"]["access"]["identity"]["viewer_id"]
            watcher_id = joined["table"]["access"]["identity"]["viewer_id"]
            pilot_socket.send_json(
                {
                    "type": "transfer_pilot",
                    "grant_revision": 1,
                    "target_viewer_id": watcher_id,
                }
            )
            old_access = pilot_socket.receive_json()
            new_access = watcher_socket.receive_json()
            assert old_access["table"]["access"]["role"] == "watcher"
            assert new_access["table"]["access"]["role"] == "pilot"
            assert old_access["table"]["access"]["grant_revision"] == 2
            assert new_access["table"]["access"]["grant_revision"] == 2

            old_frame = initial["recovery"]["frame"]
            pilot_socket.send_json(
                {
                    "type": "command",
                    "grant_revision": 1,
                    "command": _command(old_frame, "old-grant"),
                }
            )
            denied = pilot_socket.receive_json()
            assert denied["type"] == "control_error"
            assert denied["code"] == "stale_grant"

            watcher_socket.send_json(
                {
                    "type": "command",
                    "grant_revision": 2,
                    "command": _command(old_frame, "new-pilot-command"),
                }
            )
            accepted = watcher_socket.receive_json()
            old_pilot_update = pilot_socket.receive_json()
            assert accepted["status"] == "accepted"
            assert old_pilot_update["update"]["frame"] == accepted["update"]["frame"]

            pilot.__exit__(None, None, None)
            resumed_context = client.websocket_connect("/ws/play")
            resumed = resumed_context.__enter__()
            try:
                resumed.send_json(
                    {
                        "type": "resume",
                        "session_id": initial["session_id"],
                        "resume_token": old_pilot_token,
                    }
                )
                resumed_old = resumed.receive_json()
                current_new = watcher_socket.receive_json()
                assert (
                    resumed_old["table"]["access"]["identity"]["viewer_id"]
                    == old_pilot_id
                )
                assert resumed_old["table"]["access"]["role"] == "watcher"
                assert current_new["table"]["access"]["role"] == "pilot"
            finally:
                resumed_context.__exit__(None, None, None)
        finally:
            watcher.__exit__(None, None, None)


def test_both_roles_continue_in_place_to_same_study_table(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(app) as client:
        pilot, pilot_socket, watcher, watcher_socket, _initial, joined = _open_table(
            client
        )
        try:
            payload = joined
            for _ in range(400):
                if payload["type"] == "game_over":
                    break
                pilot_socket.send_json(
                    {
                        "type": "action",
                        "grant_revision": 1,
                        "index": payload["actions"][0]["index"],
                    }
                )
                pilot_payload = pilot_socket.receive_json()
                watcher_payload = watcher_socket.receive_json()
                assert (
                    pilot_payload["recovery"]["frame"]
                    == watcher_payload["recovery"]["frame"]
                )
                payload = watcher_payload
            else:
                raise AssertionError("shared match did not terminate")

            assert payload["type"] == "game_over"
            assert payload["table"]["mode"] == "study"
            assert pilot_payload["table"]["mode"] == "study"
            assert (
                pilot_payload["table"]["decisions"]
                == watcher_payload["table"]["decisions"]
            )
            assert pilot_payload["table"]["table_id"] == joined["table"]["table_id"]
        finally:
            watcher.__exit__(None, None, None)
            pilot.__exit__(None, None, None)
