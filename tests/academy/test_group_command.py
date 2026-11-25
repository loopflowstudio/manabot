"""Authority tests for Academy grouped decisions (`live_group`, `/group-preview`, `group_command`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame
from tests.academy.socket_messages import settled_message

COMBAT_CONFIG: dict[str, Any] = {
    "hero_deck": {"Mountain": 30, "Gray Ogre": 10},
    "villain_deck": {"Forest": 30, "Llanowar Elves": 10},
    "villain_type": "passive",
    "seed": 11,
    "auto_pass": False,
}

SURVEIL_CONFIG: dict[str, Any] = {
    "hero_deck": {"Forest": 24, "Esper Origins": 16},
    "villain_deck": {"Island": 40},
    "villain_type": "passive",
    "seed": 7,
    "auto_pass": False,
}


def _invite(table: dict[str, Any]) -> tuple[str, str]:
    fragment = str(table["watcher_invite"])
    assert fragment.startswith("#table=")
    table_part, token_part = fragment[1:].split("&")
    return table_part.split("=", 1)[1], token_part.split("=", 1)[1]


def _step_command(
    game: server.GameSession,
    frame: ExperienceFrame,
    offer_id: int,
    command_id: str | None = None,
) -> ExperienceFrame:
    assert frame.prompt is not None
    outcome = game.hero_command(
        {
            "command_id": command_id or f"cmd-{frame.revision}-{frame.prompt.id}",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "offer_id": offer_id,
            "answers": [],
        }
    )
    assert outcome["status"] == "accepted", outcome
    return ExperienceFrame.model_validate(outcome["update"]["frame"])


def _advance_session_to_three_ogres_attack(
    game: server.GameSession,
) -> ExperienceFrame:
    assert game.env is not None
    frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    while keep := next((o for o in frame.offers if o.label == "Keep hand"), None):
        frame = _step_command(game, frame, keep.id)
    for _ in range(3):
        game.env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    game.obs = game.env.scenario_refresh()
    game.published_prompt = None
    frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    while frame.action_space != "DECLARE_ATTACKER":
        passing = next(o for o in frame.offers if o.verb == "pass_priority")
        frame = _step_command(game, frame, passing.id)
    assert frame.prompt is not None
    return frame


def _advance_ws_to_surveil(
    pilot_socket: Any,
    watcher_socket: Any,
    initial_msg: dict[str, Any],
    grant_revision: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Play lands and cast Esper Origins over the socket until Surveil surfaces."""
    pilot_msg = initial_msg
    watcher_msg: dict[str, Any] = initial_msg
    for step in range(80):
        group = pilot_msg.get("group")
        if isinstance(group, dict) and group.get("kind") == "surveil":
            return pilot_msg, watcher_msg
        frame = (
            pilot_msg["update"]["frame"]
            if pilot_msg["type"] == "command_outcome"
            else pilot_msg["frame"]
        )
        offers = frame["offers"]
        chosen = next(
            (o for o in offers if o["label"] == "Keep hand"),
            None,
        )
        if chosen is None:
            chosen = next((o for o in offers if o["verb"] == "play_land"), None)
        if chosen is None:
            chosen = next(
                (o for o in offers if "Esper Origins" in o["label"]),
                None,
            )
        if chosen is None:
            chosen = next(
                (o for o in offers if o["verb"] in {"pay_cost", "pass_priority"}),
                offers[0],
            )
        pilot_socket.send_json(
            {
                "type": "command",
                "grant_revision": grant_revision,
                "command": {
                    "command_id": f"surveil-setup-{step}",
                    "match_id": frame["match_id"],
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": chosen["id"],
                    "answers": [],
                },
            }
        )
        pilot_msg = settled_message(pilot_socket)
        watcher_msg = settled_message(watcher_socket)
        assert pilot_msg["status"] == "accepted", pilot_msg
    raise AssertionError("did not reach Esper Origins Surveil")


def test_group_command_records_identical_canonical_decisions_and_digest_as_single_commands(
    tmp_path: Path,
) -> None:
    single = server.GameSession(
        tmp_path / "single", id_factory=lambda _kind: "match.parity"
    )
    grouped = server.GameSession(
        tmp_path / "grouped", id_factory=lambda _kind: "match.parity"
    )
    single.new_game(COMBAT_CONFIG)
    grouped.new_game(COMBAT_CONFIG)

    single_frame = _advance_session_to_three_ogres_attack(single)
    grouped_frame = _advance_session_to_three_ogres_attack(grouped)
    assert single_frame == grouped_frame

    active_group = grouped.live_group()
    assert active_group is not None
    assert active_group["kind"] == "declare_attackers"
    ogres = [entry["item"] for entry in active_group["declarable"]]
    assert len(ogres) == 3

    # Session A: submit 3 individual single-step commands (attack with first 2, decline 3rd).
    frame_a = single_frame
    before_count = len(single.canonical_decisions)
    for idx, should_attack in enumerate((True, True, False)):
        assert frame_a.action_space == "DECLARE_ATTACKER"
        offer = next(
            o for o in frame_a.offers if (o.label.startswith("Attack")) == should_attack
        )
        frame_a = _step_command(single, frame_a, offer.id, command_id=f"single-{idx}")

    # Session B: submit 1 group_command with the same 2 attackers.
    assert grouped_frame.prompt is not None
    outcome = grouped.hero_group(
        {
            "command_id": "grp-attack-1",
            "match_id": grouped_frame.match_id,
            "expected_revision": grouped_frame.revision,
            "prompt_id": grouped_frame.prompt.id,
            "frame_hash": grouped_frame.frame_hash,
            "plan": {"kind": "declare_attackers", "attackers": ogres[:2]},
        }
    )
    assert outcome["status"] == "accepted", outcome
    frame_b = ExperienceFrame.model_validate(outcome["update"]["frame"])

    assert single.env is not None and grouped.env is not None
    assert single.env.state_digest() == grouped.env.state_digest()
    assert frame_a.projection.opponent.life == 16
    assert frame_b.projection.opponent.life == 16

    assert len(single.canonical_decisions) == len(grouped.canonical_decisions)
    for row_a, row_b in zip(
        single.canonical_decisions, grouped.canonical_decisions, strict=True
    ):
        assert row_a.ordinal == row_b.ordinal
        assert row_a.viewer == row_b.viewer
        assert row_a.source == row_b.source
        assert row_a.revision == row_b.revision
        assert row_a.prompt_id == row_b.prompt_id
        assert row_a.offer_id == row_b.offer_id
        assert row_a.offer == row_b.offer
        assert row_a.frame.frame_hash == row_b.frame.frame_hash
        assert row_a.frame.projection == row_b.frame.projection

    grouped_rows = grouped.canonical_decisions[before_count : before_count + 3]
    assert [row.command_id for row in grouped_rows] == [
        "authority.group.grp-attack-1.0",
        "authority.group.grp-attack-1.1",
        "authority.group.grp-attack-1.2",
    ]
    assert [row.source for row in grouped_rows] == ["client", "client", "client"]


def test_group_preview_is_read_only_hides_steps_and_enforces_auth_and_staleness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(websocket)
            record = server.SESSION_REGISTRY[initial["session_id"]]
            game = record.game
            assert game.env is not None
            headers = {"x-etude-participant-token": initial["resume_token"]}
            endpoint = f"/api/sessions/{record.session_id}/group-preview"

            # Legacy /combat-preview endpoint is removed.
            assert (
                client.post(
                    f"/api/sessions/{record.session_id}/combat-preview",
                    json={},
                    headers=headers,
                ).status_code
                == 404
            )

            # On a non-group prompt (mulligan / priority), live_group is None and preview returns 409 no_group.
            assert initial["group"] is None
            init_frame = ExperienceFrame.model_validate(initial["frame"])
            assert init_frame.prompt is not None
            no_group_resp = client.post(
                endpoint,
                json={
                    "match_id": init_frame.match_id,
                    "revision": init_frame.revision,
                    "prompt_id": init_frame.prompt.id,
                    "frame_hash": init_frame.frame_hash,
                    "plan": {"kind": "declare_attackers", "attackers": []},
                },
                headers=headers,
            )
            assert no_group_resp.status_code == 409
            assert no_group_resp.json()["detail"] == "group_preview_no_group"

            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_group()
            assert live_group is not None
            assert live_group["kind"] == "declare_attackers"
            assert live_group["match_id"] == frame.match_id
            assert live_group["revision"] == frame.revision
            assert live_group["prompt_id"] == frame.prompt.id
            assert live_group["frame_hash"] == frame.frame_hash
            assert "fingerprint" not in live_group
            assert [c["label"] for c in live_group["declarable"]] == ["Gray Ogre"] * 3
            ogres = [c["item"] for c in live_group["declarable"]]

            address = {
                "match_id": frame.match_id,
                "revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "frame_hash": frame.frame_hash,
            }
            plan_body = {
                **address,
                "plan": {"kind": "declare_attackers", "attackers": ogres[:2]},
            }

            # Missing participant token is forbidden.
            assert client.post(endpoint, json=plan_body).status_code == 403

            before_witness = game.env.search_witness_json()
            before_digest = game.env.state_digest()
            before_prompts = game._next_prompt_id
            before_decisions = len(game.canonical_decisions)

            response = client.post(endpoint, json=plan_body, headers=headers)
            assert response.status_code == 200, response.text
            body = response.json()
            assert set(body) == {
                "match_id",
                "revision",
                "prompt_id",
                "frame_hash",
                "viewer",
                "complete",
                "refused",
                "outcome",
            }
            assert "steps" not in body
            assert body["viewer"] == 0
            assert body["complete"] is True
            assert body["refused"] is None
            assert body["outcome"] == {
                "kind": "declare_attackers",
                "attackers": ogres[:2],
            }

            # Read-only guarantee.
            assert game.env.search_witness_json() == before_witness
            assert game.env.state_digest() == before_digest
            assert game._next_prompt_id == before_prompts
            assert len(game.canonical_decisions) == before_decisions

            # Stale revision, frame_hash, or mismatched plan kind -> 409 group_preview_stale.
            for mutation in (
                {"revision": frame.revision + 1},
                {"frame_hash": "0" * 64},
                {"plan": {"kind": "scry", "top": [], "bottom": []}},
            ):
                stale = client.post(
                    endpoint, json={**plan_body, **mutation}, headers=headers
                )
                assert stale.status_code == 409
                assert stale.json()["detail"] == "group_preview_stale"


def test_websocket_group_command_commits_atomically_with_early_commit_and_no_intermediate_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(websocket)
            record = server.SESSION_REGISTRY[initial["session_id"]]
            game = record.game
            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_group()
            assert live_group is not None
            ogres = [c["item"] for c in live_group["declarable"]]
            base_revision = frame.revision

            websocket.send_json(
                {
                    "type": "group_command",
                    "command": {
                        "command_id": "ws-grp-attack",
                        "match_id": frame.match_id,
                        "expected_revision": frame.revision,
                        "prompt_id": frame.prompt.id,
                        "frame_hash": frame.frame_hash,
                        "plan": {"kind": "declare_attackers", "attackers": ogres[:2]},
                    },
                }
            )

            outcomes: list[dict[str, Any]] = []
            while True:
                msg = websocket.receive_json()
                if msg["type"] == "work_status":
                    continue
                outcomes.append(msg)
                if (
                    msg["type"] == "command_outcome"
                    and (msg.get("work") or {}).get("state") != "running"
                ):
                    break

            assert len(outcomes) == 2
            early, settled = outcomes
            assert early["type"] == "command_outcome"
            assert early["status"] == "accepted"
            assert early["work"]["state"] == "running"
            assert early["update"]["base_revision"] == base_revision
            assert early["update"]["receipt"]["command_id"] == "ws-grp-attack"
            assert early["update"]["receipt"]["accepted_at"] == base_revision
            assert early["update"]["receipt"]["resulting_revision"] == base_revision + 3
            assert early["update"]["frame"]["revision"] == base_revision + 3

            # Zero intermediate half-declared frames (base_revision+1 or +2).
            seen_revisions = [o["update"]["frame"]["revision"] for o in outcomes]
            assert base_revision + 1 not in seen_revisions
            assert base_revision + 2 not in seen_revisions

            assert settled["type"] == "command_outcome"
            assert settled["status"] == "accepted"
            assert settled["update"]["receipt"] == early["update"]["receipt"]
            assert settled["update"]["frame"]["projection"]["opponent"]["life"] == 16


def test_group_refused_menace_and_group_incomplete_leave_game_untouched(
    tmp_path: Path,
) -> None:
    def attack_with_everything(context: server.DecisionContext) -> int:
        if context.action_space == "DECLARE_ATTACKER":
            attack = next(
                (o["id"] for o in context.offers if o["label"].startswith("Attack")),
                None,
            )
            if attack is not None:
                return int(attack)
        for o in context.offers:
            if o["label"] == "Keep hand" or o["verb"] == "pass_priority":
                return int(o["id"])
        return int(context.offers[0]["id"])

    menace_game = server.GameSession(
        tmp_path / "menace", villain_offer_policy=attack_with_everything
    )
    menace_game.new_game(
        {
            "hero_deck": {"Forest": 30, "Llanowar Elves": 10},
            "villain_deck": {"Mountain": 30, "Boggart Brute": 10},
            "villain_type": "passive",
            "seed": 17,
            "auto_pass": False,
        }
    )
    assert menace_game.env is not None
    frame = ExperienceFrame.model_validate(menace_game._wire_message()["frame"])
    while keep := next((o for o in frame.offers if o.label == "Keep hand"), None):
        frame = _step_command(menace_game, frame, keep.id)
    menace_game.env.scenario_force_battlefield(1, "Boggart Brute", ready=True)
    for _ in range(2):
        menace_game.env.scenario_force_battlefield(0, "Llanowar Elves", ready=True)
    menace_game.obs = menace_game.env.scenario_refresh()
    menace_game.published_prompt = None
    frame = ExperienceFrame.model_validate(menace_game._wire_message()["frame"])
    while frame.action_space != "DECLARE_BLOCKER":
        offer = next(
            (o for o in frame.offers if o.verb == "pass_priority"),
            None,
        ) or next(o for o in frame.offers if not o.label.startswith("Attack"))
        frame = _step_command(menace_game, frame, offer.id)
    assert frame.prompt is not None

    block_group = menace_game.live_group()
    assert block_group is not None
    assert block_group["kind"] == "declare_blockers"
    first_elf = block_group["declarable"][0]
    brute = first_elf["can_block"][0]

    before_digest = menace_game.env.state_digest()
    before_decisions = len(menace_game.canonical_decisions)
    before_revision = menace_game.revision

    # Lone blocker on Boggart Brute (menace) -> group_refused, no recovery, no mutation.
    refused = menace_game.hero_group(
        {
            "command_id": "grp-menace-lone",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "frame_hash": frame.frame_hash,
            "plan": {
                "kind": "declare_blockers",
                "blocks": [{"blocker": first_elf["item"], "attacker": brute}],
            },
        }
    )
    assert refused["status"] == "rejected"
    assert "recovery" not in refused
    assert refused["rejection"]["code"] == "group_refused"
    reason = refused["rejection"]["refused"]["reason"]
    assert "block" in reason and "Boggart Brute" in reason, reason
    assert refused["rejection"]["message"] == refused["rejection"]["refused"]["reason"]
    assert menace_game.env.state_digest() == before_digest
    assert len(menace_game.canonical_decisions) == before_decisions
    assert menace_game.revision == before_revision

    # Reserved command_id and stale frame_hash checks.
    reserved = menace_game.hero_group(
        {
            "command_id": "authority.spoof",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "frame_hash": frame.frame_hash,
            "plan": {"kind": "declare_blockers", "blocks": []},
        }
    )
    assert reserved["status"] == "rejected"
    assert reserved["rejection"]["code"] == "reserved_command_id"
    assert "recovery" not in reserved

    stale = menace_game.hero_group(
        {
            "command_id": "grp-stale-hash",
            "match_id": frame.match_id,
            "expected_revision": frame.revision,
            "prompt_id": frame.prompt.id,
            "frame_hash": "0" * 64,
            "plan": {"kind": "declare_blockers", "blocks": []},
        }
    )
    assert stale["status"] == "rejected"
    assert stale["rejection"]["code"] == "stale_revision"
    assert "recovery" in stale

    # Incomplete OrderLibrary plan -> group_incomplete naming the unplaced card, 0 rows written.
    surveil_game = server.GameSession(tmp_path / "surveil")
    surveil_game.new_game(SURVEIL_CONFIG)
    assert surveil_game.env is not None
    s_frame = ExperienceFrame.model_validate(surveil_game._wire_message()["frame"])
    for step in range(80):
        active = surveil_game.live_group()
        if active is not None and active["kind"] == "surveil":
            break
        offer = next(
            (o for o in s_frame.offers if o.label == "Keep hand"),
            None,
        )
        if offer is None:
            offer = next((o for o in s_frame.offers if o.verb == "play_land"), None)
        if offer is None:
            offer = next(
                (o for o in s_frame.offers if "Esper Origins" in o.label), None
            )
        if offer is None:
            offer = next(
                (o for o in s_frame.offers if o.verb in {"pay_cost", "pass_priority"}),
                s_frame.offers[0],
            )
        s_frame = _step_command(surveil_game, s_frame, offer.id, f"s-setup-{step}")
    else:
        pytest.fail("did not reach Surveil")

    assert s_frame.prompt is not None
    surveil_done = surveil_game.hero_group(
        {
            "command_id": "grp-surveil-keep",
            "match_id": s_frame.match_id,
            "expected_revision": s_frame.revision,
            "prompt_id": s_frame.prompt.id,
            "frame_hash": s_frame.frame_hash,
            "plan": {"kind": "surveil", "graveyard": []},
        }
    )
    assert surveil_done["status"] == "accepted", surveil_done
    order_frame = ExperienceFrame.model_validate(surveil_done["update"]["frame"])
    assert order_frame.prompt is not None
    order_group = surveil_game.live_group()
    assert order_group is not None
    assert order_group["kind"] == "order_library"
    assert order_group["end"] == "top"
    assert len(order_group["items"]) == 2

    order_digest = surveil_game.env.state_digest()
    order_decisions = len(surveil_game.canonical_decisions)
    incomplete = surveil_game.hero_group(
        {
            "command_id": "grp-order-incomplete",
            "match_id": order_frame.match_id,
            "expected_revision": order_frame.revision,
            "prompt_id": order_frame.prompt.id,
            "frame_hash": order_frame.frame_hash,
            "plan": {
                "kind": "order_library",
                "order": [order_group["items"][0]["item"]],
            },
        }
    )
    assert incomplete["status"] == "rejected"
    assert "recovery" not in incomplete
    assert incomplete["rejection"]["code"] == "group_incomplete"
    assert order_group["items"][1]["label"] in incomplete["rejection"]["message"]
    assert surveil_game.env.state_digest() == order_digest
    assert len(surveil_game.canonical_decisions) == order_decisions


def test_group_command_duplicate_id_across_pilot_transfer_and_cross_participant_denial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with (
            client.websocket_connect("/ws/play") as pilot_socket,
            client.websocket_connect("/ws/play") as watcher_socket,
        ):
            pilot_socket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(pilot_socket)
            table_id, invite_token = _invite(initial["table"])
            watcher_socket.send_json(
                {
                    "type": "join_table",
                    "table_id": table_id,
                    "invite_token": invite_token,
                }
            )
            joined = settled_message(watcher_socket)
            settled_message(pilot_socket)

            record = server.SESSION_REGISTRY[initial["session_id"]]
            game = record.game
            assert game.env is not None
            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_group()
            assert live_group is not None
            ogres = [c["item"] for c in live_group["declarable"]]

            group_payload = {
                "type": "group_command",
                "grant_revision": 1,
                "command": {
                    "command_id": "grp-dup-1",
                    "match_id": frame.match_id,
                    "expected_revision": frame.revision,
                    "prompt_id": frame.prompt.id,
                    "frame_hash": frame.frame_hash,
                    "plan": {"kind": "declare_attackers", "attackers": ogres[:1]},
                },
            }
            pilot_socket.send_json(group_payload)
            accepted = settled_message(pilot_socket)
            settled_message(watcher_socket)
            assert accepted["status"] == "accepted"
            receipt = accepted["update"]["receipt"]
            decisions_after = len(game.canonical_decisions)
            digest_after = game.env.state_digest()

            # Resending the same command_id returns duplicate and writes 0 rows.
            pilot_socket.send_json(group_payload)
            dup = settled_message(pilot_socket)
            assert dup["type"] == "command_outcome"
            assert dup["status"] == "duplicate"
            assert dup["receipt"] == receipt
            assert "recovery" in dup
            assert len(game.canonical_decisions) == decisions_after
            assert game.env.state_digest() == digest_after

            # Transfer pilot to watcher.
            watcher_id = joined["table"]["access"]["identity"]["viewer_id"]
            pilot_socket.send_json(
                {
                    "type": "transfer_pilot",
                    "grant_revision": 1,
                    "target_viewer_id": watcher_id,
                }
            )
            settled_message(pilot_socket)
            settled_message(watcher_socket)

            # Original submitter (now watcher with old grant_revision) still gets duplicate.
            pilot_socket.send_json(group_payload)
            dup_after_transfer = settled_message(pilot_socket)
            assert dup_after_transfer["type"] == "command_outcome"
            assert dup_after_transfer["status"] == "duplicate"
            assert dup_after_transfer["receipt"] == receipt
            assert len(game.canonical_decisions) == decisions_after
            assert game.env.state_digest() == digest_after

            # Another participant (the new pilot) reusing that command_id gets forbidden.
            watcher_socket.send_json({**group_payload, "grant_revision": 2})
            denied = settled_message(watcher_socket)
            assert denied["type"] == "control_error"
            assert denied["code"] == "forbidden"
            assert len(game.canonical_decisions) == decisions_after
            assert game.env.state_digest() == digest_after


def test_watcher_hides_private_surveil_group_and_transfer_pilot_delivers_live_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with (
            client.websocket_connect("/ws/play") as pilot_socket,
            client.websocket_connect("/ws/play") as watcher_socket,
        ):
            pilot_socket.send_json({"type": "new_game", "config": SURVEIL_CONFIG})
            initial = settled_message(pilot_socket)
            table_id, invite_token = _invite(initial["table"])
            watcher_socket.send_json(
                {
                    "type": "join_table",
                    "table_id": table_id,
                    "invite_token": invite_token,
                }
            )
            joined = settled_message(watcher_socket)
            settled_message(pilot_socket)

            pilot_surveil, watcher_surveil = _advance_ws_to_surveil(
                pilot_socket, watcher_socket, initial, grant_revision=1
            )
            surveil_group = pilot_surveil["group"]
            assert surveil_group is not None
            assert surveil_group["kind"] == "surveil"
            assert len(surveil_group["items"]) == 2
            assert watcher_surveil["group"] is None

            # Watcher is forbidden from /group-preview and group_command.
            record = server.SESSION_REGISTRY[initial["session_id"]]
            assert record.game.env is not None
            before_digest = record.game.env.state_digest()
            watcher_headers = {
                "x-etude-participant-token": joined["resume_token"],
            }
            preview_denied = client.post(
                f"/api/sessions/{record.session_id}/group-preview",
                json={
                    "match_id": surveil_group["match_id"],
                    "revision": surveil_group["revision"],
                    "prompt_id": surveil_group["prompt_id"],
                    "frame_hash": surveil_group["frame_hash"],
                    "plan": {"kind": "surveil", "graveyard": []},
                },
                headers=watcher_headers,
            )
            assert preview_denied.status_code == 403
            assert preview_denied.json()["detail"] == "group_preview_forbidden"

            watcher_socket.send_json(
                {
                    "type": "group_command",
                    "grant_revision": 1,
                    "command": {
                        "command_id": "watcher-surveil-attempt",
                        "match_id": surveil_group["match_id"],
                        "expected_revision": surveil_group["revision"],
                        "prompt_id": surveil_group["prompt_id"],
                        "frame_hash": surveil_group["frame_hash"],
                        "plan": {"kind": "surveil", "graveyard": []},
                    },
                }
            )
            denied_ws = settled_message(watcher_socket)
            assert denied_ws["type"] == "control_error"
            assert denied_ws["code"] == "forbidden"
            # Drain work_status updates delivered to pilot for the failed watcher request.
            for expected_state in ("running", "failed"):
                ws_status = pilot_socket.receive_json()
                assert ws_status["type"] == "work_status"
                assert ws_status["work"]["state"] == expected_state
            assert record.game.env.state_digest() == before_digest

            # Transfer pilot to watcher: role_changed immediately delivers the live group to
            # the new pilot and null to the former pilot without any reload.
            watcher_id = joined["table"]["access"]["identity"]["viewer_id"]
            pilot_socket.send_json(
                {
                    "type": "transfer_pilot",
                    "grant_revision": 1,
                    "target_viewer_id": watcher_id,
                }
            )
            former_pilot_role = settled_message(pilot_socket)
            new_pilot_role = settled_message(watcher_socket)
            assert former_pilot_role["type"] == "role_changed"
            assert former_pilot_role["table"]["access"]["role"] == "watcher"
            assert former_pilot_role["group"] is None

            assert new_pilot_role["type"] == "role_changed"
            assert new_pilot_role["table"]["access"]["role"] == "pilot"
            assert new_pilot_role["group"] == surveil_group

            # New pilot confirms Surveil keeping both on top -> OrderLibrary Top surfaces only for new pilot.
            watcher_socket.send_json(
                {
                    "type": "group_command",
                    "grant_revision": 2,
                    "command": {
                        "command_id": "new-pilot-surveil",
                        "match_id": surveil_group["match_id"],
                        "expected_revision": surveil_group["revision"],
                        "prompt_id": surveil_group["prompt_id"],
                        "frame_hash": surveil_group["frame_hash"],
                        "plan": {"kind": "surveil", "graveyard": []},
                    },
                }
            )
            new_pilot_order = settled_message(watcher_socket)
            former_pilot_order = settled_message(pilot_socket)
            assert new_pilot_order["status"] == "accepted"
            assert former_pilot_order["group"] is None
            order_group = new_pilot_order["group"]
            assert order_group is not None
            assert order_group["kind"] == "order_library"
            assert order_group["end"] == "top"
            assert len(order_group["items"]) == 2

            # New pilot confirms OrderLibrary Top with reversed order.
            watcher_socket.send_json(
                {
                    "type": "group_command",
                    "grant_revision": 2,
                    "command": {
                        "command_id": "new-pilot-order",
                        "match_id": order_group["match_id"],
                        "expected_revision": order_group["revision"],
                        "prompt_id": order_group["prompt_id"],
                        "frame_hash": order_group["frame_hash"],
                        "plan": {
                            "kind": "order_library",
                            "order": [
                                order_group["items"][1]["item"],
                                order_group["items"][0]["item"],
                            ],
                        },
                    },
                }
            )
            order_done = settled_message(watcher_socket)
            settled_message(pilot_socket)
            assert order_done["status"] == "accepted"
            assert order_done["group"] is None
