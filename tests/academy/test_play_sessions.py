"""Repeated Play keeps one private setup and ordinary, native-seat recordings."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server, trace
from managym.academy.attempts import AttemptStore
from managym.academy.opponent import set_policy_registry
from managym.academy.recording import Recording
from managym.academy.shared_table_protocol import RematchMessage
from managym.academy.watch import WatchRequest, create_watch
from managym.policy import PolicyRegistry
from tests.academy.socket_messages import settled_message
from tests.academy.test_policy_runtime import IndependentFactory
from tests.academy.test_server import _command, _offer

CONFIG: dict[str, Any] = {
    "hero_deck": {"Island": 8},
    "villain_deck": {"Mountain": 8},
    "villain_type": "passive",
    "seed": "18446744073709551615",
    "auto_pass": False,
}


def continuation(game: server.GameSession, **changes: Any) -> RematchMessage:
    membership = game.attempt_store.session_for_attempt(game.match_id)
    assert membership is not None
    return RematchMessage(
        type="rematch",
        grant_revision=0,
        request_id="next",
        attempt_id=game.match_id,
        setup_revision=membership["revision"],
        **changes,
    )


def finish(game: server.GameSession) -> None:
    for index in range(100):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            return
        offer = next(
            (
                o
                for o in frame["offers"]
                if o["action_type"] == "PRIORITY_PASS_PRIORITY"
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"finish.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
    raise AssertionError("short deck-out fixture did not finish")


@pytest.mark.parametrize("seat", [0, 1])
def test_native_seats_complete_record_reconstruct_and_study(
    tmp_path: Path, seat: int
) -> None:
    game = server.GameSession(tmp_path)
    frame = game.new_game({**CONFIG, "human_seat": seat})["frame"]
    assert frame["projection"]["agent"]["player_index"] == seat
    assert frame["prompt"]["actor"] == seat
    assert game.trace is not None
    configs = game.trace.config.to_rust()
    assert configs.mana_taps == (False, False)
    assert configs[seat].decklist == CONFIG["hero_deck"]
    finish(game)
    payload = game.attempt_store.load(game.match_id)
    assert payload is not None
    assert payload["winner"] == 0  # Native seat 0 skips the first draw.
    assert payload["final_observation"]["agent"]["player_index"] == seat
    human = next(
        p
        for p in game.attempt_store.metadata(game.match_id)["players"]
        if p["kind"] == "human"
    )
    assert human["seat"] == seat
    recording = Recording(game.match_id, payload)
    choices = [p for p in recording.positions if p.kind == "choice"]
    assert choices and all(p.decision.viewer == seat for p in choices if p.decision)
    source = recording.read(choices[0].key)
    assert source["observation"]["agent"]["player_index"] == seat
    restored = game.fork_study(choices[0].key)
    assert restored is not None
    prepared = trace.prepare_trace_payload(payload)
    assert prepared["events"][0]["observation"]["agent"]["player_index"] == seat


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("mode", [True, "usable"])
def test_mana_tap_survives_reconnect_and_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seat: int, mode: managym.ManaTaps
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    config = {
        **CONFIG,
        "human_seat": seat,
        "mana_taps": mode,
        "hero_deck": {"Island": 8, "Consider": 4}
        if mode == "usable"
        else {"Island": 16},
        "villain_deck": {"Mountain": 16},
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as ws:
            ws.send_json({"type": "new_game", "config": config})
            start = settled_message(ws)
            assert start["play_session"]["draft"]["mana_taps"] == mode
            frame = start["recovery"]["frame"]
            land = _offer(frame, action_type="PRIORITY_PLAY_LAND")
            ws.send_json({"type": "command", "command": _command(frame, land, "land")})
            pending = settled_message(ws)["update"]["frame"]
            source = pending["projection"]["agent"]["battlefield"][0]
            tap = _offer(
                pending, action_type="TAP_FOR_COST", label="Tap Island for mana"
            )
            assert tap["actor"] == seat
            assert tap["focus"] == [source["id"]]
            assert tap["subjects"][0]["object"]["display_id"] == source["id"]
            command = _command(pending, tap, "float")

        with client.websocket_connect("/ws/play") as ws:
            ws.send_json(
                {
                    "type": "resume",
                    "session_id": start["session_id"],
                    "resume_token": start["resume_token"],
                }
            )
            assert settled_message(ws)["recovery"]["frame"] == pending
            ws.send_json({"type": "command", "command": command})
            accepted = settled_message(ws)
            assert accepted["status"] == "accepted"
            ws.send_json(
                {"type": "command", "command": {**command, "command_id": "stale"}}
            )
            rejected = settled_message(ws)
            assert rejected["rejection"]["code"] == "stale_revision"
            assert rejected["recovery"]["frame"] == accepted["update"]["frame"]

            game = server.SESSION_REGISTRY[start["session_id"]].game
            payload = game.attempt_store.load(game.match_id)
            assert payload is not None
            recorded = trace.GameConfig(**payload["config"])
            assert recorded.to_rust().mana_taps == (
                mode if seat == 0 else False,
                mode if seat == 1 else False,
            )
            recording = Recording(game.match_id, payload)
            position = next(
                p
                for p in recording.positions
                if p.decision
                and p.decision.command_id == "float"
                and p.kind == "choice"
            )
            decision = recording.read(position.key)["decision"]
            assert decision["frame"] == pending
            assert decision["offer"] == tap
            assert decision["command"] == command


def test_swap_repeat_receipts_refusal_and_stop_draft(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    first = game.match_id
    request = continuation(game, intent="swap_sides")
    game.continue_game(request, game.attempt_owner)
    second = game.match_id
    assert first != second
    assert game.trace is not None
    assert game.trace.config.human_seat == 1
    assert game.trace.config.hero_deck == CONFIG["hero_deck"]
    assert game.trace.config.seed == 2**64 - 1
    game.continue_game(request, game.attempt_owner)
    assert game.match_id == second
    with pytest.raises(ValueError, match="different setup"):
        game.continue_game(
            request.model_copy(update={"intent": "rematch"}), game.attempt_owner
        )
    failed = continuation(
        game, config={"villain_type": "policy", "opponent_sha256": "f" * 64}
    ).model_copy(update={"request_id": "refusal"})
    with pytest.raises(ValueError):
        game.continue_game(failed, game.attempt_owner)
    assert game.match_id == second
    game.set_stops({"my": [], "opponent": []}, True, False)
    membership = game.attempt_store.session_for_attempt(second)
    assert membership is not None
    session = game.attempt_store.session(membership["session_id"], {game.attempt_owner})
    assert session["draft"]["stops"] == {"my": [], "opponent": []}
    assert session["revision"] > request.setup_revision
    game.close("new_game")


def test_session_summary_is_private_complete_and_paginated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new_game", "config": CONFIG})
        first = settled_message(ws)
        assert "play_session" in first
        session = first["play_session"]
        token = first["resume_token"]
        owner = server.participant_id(token)
        record = server.SESSION_REGISTRY[first["session_id"]]
        finish(record.game)
        for number in range(2):
            request = continuation(
                record.game, intent="swap_sides" if number == 0 else "rematch"
            ).model_copy(update={"request_id": f"next.{number}"})
            record.game.continue_game(request, owner)
            if number == 0:
                finish(record.game)
        store = record.game.attempt_store
        rows = store.session_results(session["id"], {owner})
        opponent = next(
            player
            for player in store.metadata(record.game.match_id)["players"]
            if player["kind"] == "bot"
        )
        assert all(row["opponent_name"] == opponent["name"] for row in rows)
        assert all(row["opponent_id"] == opponent["player_id"] for row in rows)
        assert sum(r["completed"] for r in rows) == 2
        assert sum(r["active"] for r in rows) == 1
        headers = {"x-etude-participant-tokens": token}
        response = client.get(f"/api/play-sessions/{session['id']}", headers=headers)
        assert response.status_code == 200
        assert client.get(f"/api/play-sessions/{session['id']}").status_code == 403
        page = client.get(
            "/api/traces",
            params={"session": session["id"], "limit": 1},
            headers=headers,
        )
        assert len(page.json()) == 1
        assert page.headers.get("x-etude-next-cursor")
        assert (
            client.get("/api/traces", params={"session": session["id"]}).status_code
            == 400
        )
        assert (
            "hero_source"
            not in client.get(f"/api/traces/{first['table']['attempt_id']}").json()[
                "config"
            ]
        )
        with store.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM session_games WHERE session_id=?",
                    (session["id"],),
                ).fetchone()[0]
                == 3
            )
        record.game.close("new_game")
        assert (
            sum(r["stopped"] for r in store.session_results(session["id"], {owner}))
            == 1
        )


def test_existing_store_upgrade_and_restart_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = server.GameSession(tmp_path)
    game.new_game({**CONFIG, "request_id": "initial"})
    identifier = game.match_id
    with game.attempt_store.connect() as db:
        db.execute(
            "UPDATE attempts SET process_id=? WHERE id=?", ("old-process", identifier)
        )
    restarted = AttemptStore(tmp_path / "play.sqlite")
    assert restarted.load(identifier)["end_reason"] == "server_restart"
    assert (
        restarted.continuation_receipt(
            game.attempt_owner,
            "initial",
            server.request_fingerprint({**CONFIG, "request_id": "initial"}),
        )
        == identifier
    )


def test_restart_recovers_only_setup_then_explicit_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as ws:
            ws.send_json(
                {
                    "type": "new_game",
                    "config": {
                        **CONFIG,
                        "player_token": "r" * 36,
                        "player_name": "Researcher",
                    },
                }
            )
            original = settled_message(ws)
        identifier = original["table"]["attempt_id"]
        # Simulate the only state lost by process death, without stopping a
        # real host or claiming the old native authority can be reconstructed.
        server.SESSION_REGISTRY.clear()
        store = AttemptStore(tmp_path / "play.sqlite")
        with store.connect() as db:
            db.execute(
                "UPDATE attempts SET process_id='previous-process' WHERE id=?",
                (identifier,),
            )
        with client.websocket_connect("/ws/play") as ws:
            ws.send_json(
                {
                    "type": "resume",
                    "session_id": original["session_id"],
                    "resume_token": original["resume_token"],
                    "play_session_id": original["play_session"]["id"],
                }
            )
            recovered = settled_message(ws)
            assert recovered["type"] == "table_snapshot"
            assert recovered["session_interrupted"] is True
            assert "frame" not in recovered and "recovery" not in recovered
            assert store.load(identifier)["end_reason"] == "server_restart"
            ws.send_json(
                {
                    "type": "rematch",
                    "grant_revision": recovered["table"]["access"]["grant_revision"],
                    "request_id": "after-restart",
                    "attempt_id": identifier,
                    "setup_revision": recovered["play_session"]["revision"],
                }
            )
            next_game = settled_message(ws)
            assert next_game["play_session"]["game_number"] == 2
            assert next_game["table"]["attempt_id"] != identifier
            assert next_game["session_interrupted"] is False
            old_human = next(
                p for p in store.metadata(identifier)["players"] if p["kind"] == "human"
            )
            new_human = next(
                p
                for p in store.metadata(next_game["table"]["attempt_id"])["players"]
                if p["kind"] == "human"
            )
            assert new_human["player_id"] == old_human["player_id"]
            assert new_human["name"] == "Researcher"


def test_saved_draft_is_versioned_without_changing_live_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new_game", "config": CONFIG})
        original = settled_message(ws)
        identifier = original["play_session"]["id"]
        headers = {"x-etude-participant-token": original["resume_token"]}
        body = {
            "table_id": original["session_id"],
            "grant_revision": original["table"]["access"]["grant_revision"],
            "revision": 1,
            "config": {"human_seat": 1, "seed": None},
        }
        saved = client.put(
            f"/api/play-sessions/{identifier}", headers=headers, json=body
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["draft"]["human_seat"] == 1
        assert saved.json()["config"]["human_seat"] == 0
        assert (
            client.put(
                f"/api/play-sessions/{identifier}", headers=headers, json=body
            ).status_code
            == 409
        )
        assert (
            client.put(
                f"/api/play-sessions/{identifier}",
                headers={"x-etude-participant-token": "x" * 32},
                json=body,
            ).status_code
            == 409
        )


@pytest.mark.parametrize("seat", [0, 1])
def test_imported_sideboards_policy_identity_and_override_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seat: int,
) -> None:
    registry = PolicyRegistry(providers={})
    set_policy_registry(registry)
    first = IndependentFactory("First", runner="first-runner")
    second = IndependentFactory("Second", runner="second-runner")
    first_id, second_id = registry.register(first), registry.register(second)
    try:
        game = server.GameSession(tmp_path)
        game.new_game(
            {
                "villain_type": "policy",
                "opponent_sha256": first_id,
                "human_seat": seat,
                "decklists": "closed",
                "seed": 3,
                "deck_import": {
                    "hero": {
                        "source": "8 Island\nSideboard\n1 Swamp",
                        "filename": "human.txt",
                    },
                    "villain": {
                        "source": "8 Mountain\nSideboard\n1 Forest",
                        "filename": "bot.txt",
                    },
                },
            }
        )
        assert first.sessions[-1].seat == 1 - seat
        original_source = game.trace.config.hero_source
        request = continuation(game, config={"opponent_sha256": second_id})
        game.continue_game(request, game.attempt_owner)
        assert game.trace.config.opponent["sha256"] == second_id
        assert game.trace.config.hero_source == original_source
        assert game.trace.config.to_rust()[seat].sideboard == {"Swamp": 1}
        assert game.trace.config.to_rust()[1 - seat].sideboard == {"Forest": 1}
        assert game.trace.config.to_rust().decklists == "closed"
        assert game.current_message()["frame"]["projection"]["opponent"]["hand"] == []
        game.close("new_game")
    finally:
        set_policy_registry(None)


def test_released_store_upgrade_keeps_attempts_ungrouped(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    finish(game)
    identifier = game.match_id
    original = game.attempt_store.load(identifier)
    # The released schema has attempts/players/feedback but no session tables.
    with game.attempt_store.connect() as db:
        db.execute("DROP TABLE continuation_receipts")
        db.execute("DROP TABLE session_games")
        db.execute("DROP TABLE play_sessions")
    upgraded = AttemptStore(tmp_path / "play.sqlite")
    assert upgraded.session_for_attempt(identifier) is None
    assert upgraded.load(identifier) == original


def test_shared_watch_setup_admits_imports_and_keeps_private_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    created = create_watch(
        WatchRequest(
            players=["passive", "passive"],
            seed=None,
            decklists="closed",
            deck_import={
                "hero": {"source": "8 Island", "filename": "first.txt"},
                "villain": {"source": "8 Mountain", "filename": "second.txt"},
            },
        )
    )
    with TestClient(server.app) as client:
        private = client.get(
            f"/api/traces/{created['id']}/recording",
            headers={"x-etude-participant-token": created["token"]},
        )
        assert private.status_code == 200, private.text
        setup = private.json()["private_setup"]
        assert setup["seed"] is None
        assert setup["deck_import"]["hero"]["source"] == "8 Island"
        assert setup["decklists"] == "closed"
        assert setup["watch_players"] == ["passive", "passive"]
        assert client.get(f"/api/traces/{created['id']}/recording").status_code == 403
    server._drop_session(created["id"], "new_game")


def test_native_draw_is_recorded_as_a_draw(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    assert game.env is not None
    # A constructed simultaneous-loss position, never a game for a policy or person.
    game.env.scenario_set_life(0, 0)
    game.env.scenario_set_life(1, 0)
    game.obs = game.env.scenario_refresh()
    game.obs, *_ = game.env.step(0)  # Run the native state-based checks.
    assert game.obs.game_over
    assert game.env.winner_index() is None
    game._persist_attempt(end_reason="game_over")
    membership = game.attempt_store.session_for_attempt(game.match_id)
    assert membership is not None
    result = game.attempt_store.session_results(
        membership["session_id"], {game.attempt_owner}
    )[0]
    assert result["completed"] == result["draws"] == 1
    assert result["wins"] == result["losses"] == 0


@pytest.mark.parametrize("seed", [True, 1.5, "1.5", "-1", str(2**64)])
def test_invalid_seed_never_retires_current_game(tmp_path: Path, seed: object) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    identifier = game.match_id
    with pytest.raises(ValueError, match="seed"):
        game.new_game({**CONFIG, "seed": seed})
    assert game.match_id == identifier
    assert game.attempt_store.load(identifier)["end_reason"] == "active"
    game.close("new_game")


@pytest.mark.parametrize("seat", [0, 1])
def test_player_targets_name_roles_at_either_native_seat(
    tmp_path: Path, seat: int
) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            **CONFIG,
            "hero_deck": {"Mountain": 10, "Lightning Bolt": 10},
            "villain_deck": {"Plains": 20},
            "human_seat": seat,
        }
    )
    for index in range(100):
        frame = game.current_message()["frame"]
        targets = {
            subject["object"]["subject"]["id"]: offer["label"]
            for offer in frame["offers"]
            for subject in offer.get("subjects", [])
            if subject["role"] == "target"
            and subject["object"]["subject"]["kind"] == "player"
        }
        if targets:
            assert targets == {seat: "Target Hero", 1 - seat: "Target Villain"}
            game.close("new_game")
            return
        offer = next(
            (
                o
                for prefix in ("Cast Lightning Bolt", "Play Mountain", "Pass")
                for o in frame["offers"]
                if o["label"].startswith(prefix)
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"target.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
    raise AssertionError("no player target was offered")
