"""Authority tests for Academy grouped decisions (`live_decision_group`, `/decision-group-preview`, `decision_group_command`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.academy.site import SpaStaticFiles
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
    initial_msg: dict[str, Any],
) -> dict[str, Any]:
    """Play lands and cast Esper Origins over the socket until Surveil surfaces."""
    pilot_msg = initial_msg
    for step in range(80):
        group = pilot_msg.get("decision_group")
        if isinstance(group, dict) and group.get("kind") == "surveil":
            return pilot_msg
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
        assert pilot_msg["status"] == "accepted", pilot_msg
    raise AssertionError("did not reach Esper Origins Surveil")


def test_decision_group_command_records_identical_canonical_decisions_and_digest_as_single_commands(
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

    active_group = grouped.live_decision_group()
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

    # Session B: submit 1 decision_group_command with the same 2 attackers.
    assert grouped_frame.prompt is not None
    outcome = grouped.hero_decision_group(
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
        "authority.decision_group.grp-attack-1.0",
        "authority.decision_group.grp-attack-1.1",
        "authority.decision_group.grp-attack-1.2",
    ]
    assert [row.source for row in grouped_rows] == ["client", "client", "client"]


def test_group_preview_is_read_only_hides_steps_and_enforces_staleness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(websocket)
            game_id = initial["game"]
            record = server.LIVE_GAMES[game_id]
            game = record.game
            assert game.env is not None
            endpoint = f"/api/games/{game_id}/decision-group-preview"

            # Legacy /combat-preview endpoint is removed and returns 404 even with SPA mounted.
            static_dir = tmp_path / "spa"
            static_dir.mkdir()
            (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            original_routes = list(server.app.router.routes)
            try:
                server.app.mount(
                    "/",
                    SpaStaticFiles(directory=static_dir, html=True),
                    name="frontend-test",
                )
                assert (
                    client.post(
                        f"/api/games/{game_id}/combat-preview",
                        json={},
                    ).status_code
                    == 404
                )
            finally:
                server.app.router.routes[:] = original_routes

            # On a non-group prompt (mulligan / priority), live_group is None and preview returns 409 no_group.
            assert initial["decision_group"] is None
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
            )
            assert no_group_resp.status_code == 409
            assert (
                no_group_resp.json()["detail"] == "decision_group_preview_unavailable"
            )

            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_decision_group()
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

            before_witness = game.env.search_witness_json()
            before_digest = game.env.state_digest()
            before_prompts = game._next_prompt_id
            before_decisions = len(game.canonical_decisions)

            response = client.post(endpoint, json=plan_body)
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

            # Stale revision, frame_hash, or mismatched plan kind -> 409 decision_group_preview_stale.
            for mutation in (
                {"revision": frame.revision + 1},
                {"frame_hash": "0" * 64},
                {"plan": {"kind": "scry", "top": [], "bottom": []}},
            ):
                stale = client.post(endpoint, json={**plan_body, **mutation})
                assert stale.status_code == 409
                assert stale.json()["detail"] == "decision_group_preview_stale"


def test_websocket_decision_group_command_commits_atomically_with_early_commit_and_no_intermediate_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(websocket)
            record = server.LIVE_GAMES[initial["game"]]
            game = record.game
            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_decision_group()
            assert live_group is not None
            ogres = [c["item"] for c in live_group["declarable"]]
            base_revision = frame.revision

            websocket.send_json(
                {
                    "type": "decision_group_command",
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


def test_decision_group_refused_menace_and_decision_group_incomplete_leave_game_untouched(
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

    block_group = menace_game.live_decision_group()
    assert block_group is not None
    assert block_group["kind"] == "declare_blockers"
    first_elf = block_group["declarable"][0]
    brute = first_elf["can_block"][0]

    before_digest = menace_game.env.state_digest()
    before_decisions = len(menace_game.canonical_decisions)
    before_revision = menace_game.revision

    # Lone blocker on Boggart Brute (menace) -> decision_group_refused, no recovery, no mutation.
    refused = menace_game.hero_decision_group(
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
    assert refused["rejection"]["code"] == "decision_group_refused"
    reason = refused["rejection"]["refused"]["reason"]
    assert "block" in reason and "Boggart Brute" in reason, reason
    assert refused["rejection"]["message"] == refused["rejection"]["refused"]["reason"]
    assert menace_game.env.state_digest() == before_digest
    assert len(menace_game.canonical_decisions) == before_decisions
    assert menace_game.revision == before_revision

    # Reserved command_id and stale frame_hash checks.
    reserved = menace_game.hero_decision_group(
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

    stale = menace_game.hero_decision_group(
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

    # Incomplete OrderLibrary plan -> decision_group_incomplete naming the unplaced card, 0 rows written.
    surveil_game = server.GameSession(tmp_path / "surveil")
    surveil_game.new_game(SURVEIL_CONFIG)
    assert surveil_game.env is not None
    s_frame = ExperienceFrame.model_validate(surveil_game._wire_message()["frame"])
    for step in range(80):
        active = surveil_game.live_decision_group()
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
    surveil_done = surveil_game.hero_decision_group(
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
    order_group = surveil_game.live_decision_group()
    assert order_group is not None
    assert order_group["kind"] == "order_library"
    assert order_group["end"] == "top"
    assert len(order_group["items"]) == 2

    order_digest = surveil_game.env.state_digest()
    order_decisions = len(surveil_game.canonical_decisions)
    incomplete = surveil_game.hero_decision_group(
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
    assert incomplete["rejection"]["code"] == "decision_group_incomplete"
    assert order_group["items"][1]["label"] in incomplete["rejection"]["message"]
    assert surveil_game.env.state_digest() == order_digest
    assert len(surveil_game.canonical_decisions) == order_decisions


def test_decision_group_command_duplicate_id_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as pilot_socket:
            pilot_socket.send_json({"type": "new_game", "config": COMBAT_CONFIG})
            initial = settled_message(pilot_socket)

            record = server.LIVE_GAMES[initial["game"]]
            game = record.game
            assert game.env is not None
            frame = _advance_session_to_three_ogres_attack(game)
            assert frame.prompt is not None
            live_group = game.live_decision_group()
            assert live_group is not None
            ogres = [c["item"] for c in live_group["declarable"]]

            group_payload = {
                "type": "decision_group_command",
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


def test_surveil_then_order_library_decision_group_over_websocket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as pilot_socket:
            pilot_socket.send_json({"type": "new_game", "config": SURVEIL_CONFIG})
            initial = settled_message(pilot_socket)

            pilot_surveil = _advance_ws_to_surveil(pilot_socket, initial)
            surveil_group = pilot_surveil["decision_group"]
            assert surveil_group is not None
            assert surveil_group["kind"] == "surveil"
            assert len(surveil_group["items"]) == 2

            # Confirm Surveil keeping both on top -> OrderLibrary Top surfaces.
            pilot_socket.send_json(
                {
                    "type": "decision_group_command",
                    "command": {
                        "command_id": "pilot-surveil",
                        "match_id": surveil_group["match_id"],
                        "expected_revision": surveil_group["revision"],
                        "prompt_id": surveil_group["prompt_id"],
                        "frame_hash": surveil_group["frame_hash"],
                        "plan": {"kind": "surveil", "graveyard": []},
                    },
                }
            )
            pilot_order = settled_message(pilot_socket)
            assert pilot_order["status"] == "accepted"
            order_group = pilot_order["decision_group"]
            assert order_group is not None
            assert order_group["kind"] == "order_library"
            assert order_group["end"] == "top"
            assert len(order_group["items"]) == 2

            # Confirm OrderLibrary Top with reversed order.
            pilot_socket.send_json(
                {
                    "type": "decision_group_command",
                    "command": {
                        "command_id": "pilot-order",
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
            order_done = settled_message(pilot_socket)
            assert order_done["status"] == "accepted"
            assert order_done["decision_group"] is None


CURATOR_CONFIG: dict[str, Any] = {
    "hero_deck": {"Island": 24, "Curator of Destinies": 16},
    "villain_deck": {"Forest": 40},
    "villain_type": "passive",
    "seed": 13,
    "auto_pass": False,
}


def _advance_session_to_curator_partition(
    game: server.GameSession,
) -> ExperienceFrame:
    assert game.env is not None
    frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    while keep := next((o for o in frame.offers if o.label == "Keep hand"), None):
        frame = _step_command(game, frame, keep.id)
    for _ in range(6):
        game.env.scenario_force_battlefield(0, "Island", ready=True)
    game.env.scenario_clear_hand(0)
    game.env.scenario_force_card_in_hand(0, "Curator of Destinies")
    game.obs = game.env.scenario_refresh()
    game.published_prompt = None
    frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    for step in range(40):
        live = game.live_decision_group()
        if live is not None and live["kind"] == "partition_piles":
            return frame
        offer = next(
            (o for o in frame.offers if "Curator of Destinies" in o.label), None
        ) or next(
            (o for o in frame.offers if o.verb in {"pay_cost", "pass_priority"}),
            frame.offers[0],
        )
        frame = _step_command(game, frame, offer.id, command_id=f"curator-{step}")
    raise AssertionError("did not reach Curator of Destinies PartitionPiles")


def test_partition_piles_decision_group_command_parity_watcher_privacy_and_choose_pile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    single = server.GameSession(
        tmp_path / "single-curator", id_factory=lambda _kind: "match.curator"
    )
    grouped = server.GameSession(
        tmp_path / "grouped-curator", id_factory=lambda _kind: "match.curator"
    )
    single.new_game(CURATOR_CONFIG)
    grouped.new_game(CURATOR_CONFIG)

    single_frame = _advance_session_to_curator_partition(single)
    grouped_frame = _advance_session_to_curator_partition(grouped)
    assert single_frame == grouped_frame

    active_group = grouped.live_decision_group()
    assert active_group is not None
    assert active_group["kind"] == "partition_piles"
    assert len(active_group["items"]) == 5
    assert active_group["committed_face_up"] == []
    assert active_group["committed_face_down"] == []

    # Session A: submit 5 individual commands [up, down, up, down, down].
    frame_a = single_frame
    before_count = len(single.canonical_decisions)
    for idx, offer_id in enumerate((0, 1, 0, 1, 1)):
        frame_a = _step_command(single, frame_a, offer_id, command_id=f"part-{idx}")

    # Session B: submit 1 decision_group_command with cards 0 and 2 face-up.
    assert grouped_frame.prompt is not None
    face_up = [active_group["items"][0]["item"], active_group["items"][2]["item"]]
    outcome = grouped.hero_decision_group(
        {
            "command_id": "grp-partition-1",
            "match_id": grouped_frame.match_id,
            "expected_revision": grouped_frame.revision,
            "prompt_id": grouped_frame.prompt.id,
            "frame_hash": grouped_frame.frame_hash,
            "plan": {"kind": "partition_piles", "face_up": face_up},
        }
    )
    assert outcome["status"] == "accepted", outcome
    assert single.env is not None and grouped.env is not None
    assert single.env.state_digest() == grouped.env.state_digest()
    assert len(single.canonical_decisions) == len(grouped.canonical_decisions)
    grouped_rows = grouped.canonical_decisions[before_count : before_count + 5]
    assert [row.command_id for row in grouped_rows] == [
        f"authority.decision_group.grp-partition-1.{idx}" for idx in range(5)
    ]

    # Submit the whole 5-card split in one atomic decision_group_command over WebSocket.
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as pilot_socket:
            pilot_socket.send_json({"type": "new_game", "config": CURATOR_CONFIG})
            initial = settled_message(pilot_socket)

            record = server.LIVE_GAMES[initial["game"]]
            ws_frame = _advance_session_to_curator_partition(record.game)
            assert ws_frame.prompt is not None
            pilot_group = record.game.live_decision_group()
            assert pilot_group is not None and pilot_group["kind"] == "partition_piles"

            pilot_socket.send_json(
                {
                    "type": "decision_group_command",
                    "command": {
                        "command_id": "ws-partition-1",
                        "match_id": pilot_group["match_id"],
                        "expected_revision": pilot_group["revision"],
                        "prompt_id": pilot_group["prompt_id"],
                        "frame_hash": pilot_group["frame_hash"],
                        "plan": {
                            "kind": "partition_piles",
                            "face_up": [pilot_group["items"][0]["item"]],
                        },
                    },
                }
            )
            split_done = settled_message(pilot_socket)
            assert split_done["status"] == "accepted"
            assert split_done["decision_group"] is None
