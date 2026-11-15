"""Constructed combat atomicity and exact child-root proofs, without model inference."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

import pytest

from conformance.combat_planning import (
    CombatStage,
    combat_config,
    configure_combat,
    declaration_command,
)
import managym
from managym.academy import server
from managym.academy.replay_index import CanonicalReplayV1
from managym.academy.server import GameSession, PendingCanonicalDecision
from managym.academy.shared_table_protocol import REQUEST_ADAPTER, ViewerRole
from managym.decision import Command as SemanticCommand
from managym.experience import ExperienceFrame


def _session(
    path: Path, *, stage: CombatStage = "attackers", human_seat: Literal[0, 1] = 0
) -> GameSession:
    session = GameSession(path, capture_authority_evidence=True)
    session.new_game({**combat_config(), "human_seat": human_seat})
    assert session.env is not None
    session.obs = configure_combat(session.env, stage, human_seat=human_seat)
    session.published_prompt = None
    session._publish_current_prompt()
    return session


def _frame(session: GameSession) -> ExperienceFrame:
    context = session.published_prompt
    assert context is not None
    return ExperienceFrame.model_validate(context.frame)


@dataclass(frozen=True)
class _Snapshot:
    native: str
    revision: int
    next_prompt: int
    canonical: str
    presentation: str
    accepted: str
    study_roots: dict[int, str]
    pending: PendingCanonicalDecision | None


def _snapshot(session: GameSession) -> _Snapshot:
    assert session.env is not None
    return _Snapshot(
        session.env.search_witness_json(),
        session.revision,
        session._next_prompt_id,
        session.canonical_replay().model_dump_json(),
        json.dumps(session.presentation_events),
        json.dumps(session.accepted_commands),
        {
            ordinal: root.search_witness_json()
            for ordinal, root in session._study_roots.items()
        },
        session._pending_decision,
    )


@pytest.mark.parametrize("human_seat", [0, 1])
def test_combat_confirmation_records_actual_children_and_retries_once(
    tmp_path: Path,
    human_seat: Literal[0, 1],
) -> None:
    session = _session(tmp_path, human_seat=human_seat)
    assert session.env is not None
    frame = _frame(session)
    command = declaration_command(frame, "combat.confirm", select=True)
    root = session.env.clone_env()
    env_identity = session.env
    before = _snapshot(session)
    validation = session.validate_declaration(command, "validate.1")
    assert validation["status"] == "valid"
    assert _snapshot(session) == before
    outcome = session.hero_declaration(command.model_dump(mode="json"))
    assert outcome["status"] == "accepted"
    assert session.env is env_identity
    receipt = outcome["update"]["receipt"]
    assert receipt["actor"] == human_seat
    children = receipt["declaration"]["child_command_ids"]
    rows = {row.command_id: row for row in session.canonical_decisions}
    transitions = {}
    for transition in session.authority_transitions:
        assert transition.command is not None
        transitions[transition.command["command_id"]] = transition
    for identity in children:
        row = rows[identity]
        retained = session._study_roots[row.ordinal]
        assert retained.search_witness_json() == root.search_witness_json()
        transition = transitions[identity]
        assert transition.semantic_receipt is not None
        native = SemanticCommand(
            identity, transition.semantic_receipt.before_revision, row.offer_id
        )
        root.step_semantic_command(native.to_json())
    assert len(children) == 3
    assert receipt["declaration"]["completion"]["assignments"] == [
        answer.model_dump(mode="json") for answer in command.plan.assignments
    ]
    committed = _snapshot(session)
    duplicate = session.hero_declaration(command.model_dump(mode="json"))
    assert duplicate["status"] == "duplicate"
    assert _snapshot(session) == committed
    duplicate["receipt"]["declaration"]["child_command_ids"].clear()
    assert (
        session.accepted_commands[command.command_id]["declaration"][
            "child_command_ids"
        ]
        == children
    )
    assert (
        session.canonical_replay().command_receipts[-1].command_id == command.command_id
    )


def test_combat_invalid_menace_plan_preserves_game_and_recording(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, stage="blockers")
    frame = _frame(session)
    command = declaration_command(frame, "combat.invalid")
    assert frame.declaration is not None
    target = next(
        row.subject
        for row in frame.declaration.destinations
        if row.label == "Boggart Brute"
    )
    command.plan.assignments[0].destination = target
    before = _snapshot(session)
    assert session.validate_declaration(command, "invalid.1")["status"] == "invalid"
    rejected = session.hero_declaration(command.model_dump(mode="json"))
    assert rejected["status"] == "rejected"
    assert rejected["rejection"]["code"] == "invalid_declaration"
    assert _snapshot(session) == before
    command.plan.assignments[1].destination = target
    assert session.validate_declaration(command, "valid.2")["status"] == "valid"
    assert (
        session.hero_declaration(command.model_dump(mode="json"))["status"]
        == "accepted"
    )


def test_combat_recording_failure_rolls_back_before_native_publication(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    command = declaration_command(_frame(session), "combat.write-fails", select=True)
    before = _snapshot(session)

    with session.attempt_store.connect() as db:
        saved = tuple(
            db.execute(
                "SELECT revision,trace_json FROM attempts WHERE id=?",
                (session.match_id,),
            ).fetchone()
        )
        db.execute(
            "CREATE TEMP TRIGGER reject_confirmation BEFORE UPDATE ON attempts WHEN NEW.revision > OLD.revision BEGIN SELECT RAISE(FAIL, 'injected recording failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        session.hero_declaration(command.model_dump(mode="json"))
    assert _snapshot(session) == before
    with session.attempt_store.connect() as db:
        assert (
            tuple(
                db.execute(
                    "SELECT revision,trace_json FROM attempts WHERE id=?",
                    (session.match_id,),
                ).fetchone()
            )
            == saved
        )


def test_combat_parent_receipt_rejects_missing_or_reordered_children(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    command = declaration_command(_frame(session), "combat.linkage", select=True)
    session.hero_declaration(command.model_dump(mode="json"))
    record = session.canonical_replay().model_dump(mode="json")
    bad = deepcopy(record)
    bad["command_receipts"][-1]["declaration"]["child_command_ids"].reverse()
    with pytest.raises(ValueError, match="ordered commit"):
        CanonicalReplayV1.model_validate(bad)
    bad = deepcopy(record)
    bad["command_receipts"][-1]["declaration"]["child_command_ids"][0] = "absent"
    with pytest.raises(ValueError, match="absent"):
        CanonicalReplayV1.model_validate(bad)


def test_combat_fixed_prefix_survives_clear_and_old_validation_is_stale(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    old = declaration_command(_frame(session), "old.draft")
    context = session.published_prompt
    assert context is not None
    offer = next(
        offer
        for action, offer in zip(context.actions, context.offers, strict=True)
        if action.get("declared") is True
    )
    session.hero_command(
        {
            "command_id": "prefix.attack",
            "match_id": session.match_id,
            "expected_revision": context.revision,
            "prompt_id": context.prompt_id,
            "offer_id": offer["id"],
            "answers": [],
        }
    )
    before = _snapshot(session)
    assert session.validate_declaration(old, "stale.draft")["status"] == "stale"
    assert _snapshot(session) == before
    frame = _frame(session)
    assert frame.declaration is not None
    fixed = [row for row in frame.declaration.subjects if row.fixed]
    assert len(fixed) == 1 and fixed[0].assigned_to is not None
    clear = declaration_command(frame, "prefix.clear")
    assert len(clear.plan.assignments) == 2
    result = session.hero_declaration(clear.model_dump(mode="json"))
    completion = result["update"]["receipt"]["declaration"]["completion"]["assignments"]
    assert len(completion) == 3
    assert sum(answer["destination"] is not None for answer in completion) == 1


def test_combat_late_recording_preparation_failure_preserves_live_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    command = declaration_command(_frame(session), "late.failure", select=True)
    before = _snapshot(session)
    original = GameSession._step_and_record

    def fail_after_child(
        self: GameSession,
        actor: str,
        action_index: int,
        actions: list[dict[str, Any]],
        auto: bool = False,
        context: server.DecisionContext | None = None,
        command: dict[str, Any] | None = None,
        source: str | None = None,
        prepared_child: managym.DeclarationChild | None = None,
    ) -> None:
        original(
            self,
            actor,
            action_index,
            actions,
            auto,
            context,
            command,
            source,
            prepared_child,
        )
        if prepared_child is not None:
            raise RuntimeError("injected after private child recording")

    monkeypatch.setattr(GameSession, "_step_and_record", fail_after_child)
    with pytest.raises(RuntimeError, match="private child"):
        session.hero_declaration(command.model_dump(mode="json"))
    assert _snapshot(session) == before


def test_combat_retry_ownership_survives_post_commit_policy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    record = server.SessionRecord("table", "pilot-token", session)
    pilot = record.pilot()
    watcher = server.ParticipantLease("watcher", "watcher-token", ViewerRole.WATCHER)
    record.participants[watcher.viewer_id] = watcher
    command = declaration_command(_frame(session), "policy.failure", select=True)
    request = REQUEST_ADAPTER.validate_python(
        {
            "type": "declaration",
            "grant_revision": 1,
            "command": command.model_dump(mode="json"),
        }
    )
    before = _snapshot(session)
    with pytest.raises(server.TableDispatchError, match="cannot declaration"):
        server._dispatch_table_request(record, watcher, request)
    assert _snapshot(session) == before

    def unavailable() -> None:
        raise RuntimeError("injected policy unavailable after commit")

    monkeypatch.setattr(session, "_observe_policies", unavailable)
    with pytest.raises(RuntimeError, match="after commit"):
        server._dispatch_table_request(record, pilot, request)
    committed = _snapshot(session)
    assert committed.revision > before.revision
    assert record.accepted_command_submitters[command.command_id] == pilot.viewer_id
    result, changed = server._dispatch_table_request(record, pilot, request)
    assert result["status"] == "duplicate" and not changed
    assert _snapshot(session) == committed
    with pytest.raises(server.TableDispatchError, match="another participant"):
        server._dispatch_table_request(record, watcher, request)
    assert _snapshot(session) == committed


def test_combat_parent_cannot_use_native_child_identity(tmp_path: Path) -> None:
    session = _session(tmp_path)
    command = declaration_command(_frame(session), "compound-0-0", select=True)
    before = _snapshot(session)
    result = session.hero_declaration(command.model_dump(mode="json"))
    assert result["rejection"]["code"] == "reserved_command_id"
    assert _snapshot(session) == before


def test_socket_draft_validation_does_not_claim_live_advancement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from tests.academy.socket_messages import settled_message

    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    with TestClient(server.app).websocket_connect("/ws/play") as socket:
        socket.send_json({"type": "new_game", "config": combat_config()})
        created = settled_message(socket)
        record = server.SESSION_REGISTRY[created["session_id"]]
        session = record.game
        assert session.env is not None
        session.obs = configure_combat(session.env, "attackers")
        session.published_prompt = None
        session._publish_current_prompt()
        command = declaration_command(_frame(session), "preview", select=True)
        before = _snapshot(session)
        work = session.work.read()
        for request_id in ("preview.1", "preview.2"):
            socket.send_json(
                {
                    "type": "validate_declaration",
                    "request_id": request_id,
                    "command": command.model_dump(mode="json"),
                }
            )
            # Read the socket directly: no running-work message may precede the
            # validation response or disable the composer's pending request.
            response = socket.receive_json()
            assert response["type"] == "declaration_validation"
            assert response["request_id"] == request_id
            assert response["status"] == "valid"
            assert (response.get("work") or {}).get("state") != "running"
            assert session.work.read() == work
            assert _snapshot(session) == before
    server._drop_session(record.session_id, "test")
