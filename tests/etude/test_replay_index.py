from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from etude import server, trace as trace_store
from etude.replay_index import (
    CanonicalReplayProjectionV1,
    CanonicalReplayV1,
    DecisionAddressV2,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecisionAddress,
    project_replay,
    projection_with_addresses,
    restore_decision,
)
from etude.server import GameSession, app
from scripts.generate_replay_fixtures import _play_pinned_match

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "protocol/fixtures/bolt-target.json").read_text(
        encoding="utf-8"
    )
)
PROTOCOL_DIR = Path(__file__).parents[2] / "protocol"
REPLAY_SCHEMA = json.loads(
    (PROTOCOL_DIR / "canonical-replay-v1.schema.json").read_text(encoding="utf-8")
)


def _row(ordinal: int, viewer: int, revision: int) -> dict:
    frame = deepcopy(FIXTURE["recovery"]["frame"])
    frame["revision"] = revision
    frame["prompt"]["id"] = 100 + ordinal
    frame["prompt"]["actor"] = viewer
    if viewer == 1:
        frame["projection"]["agent"], frame["projection"]["opponent"] = (
            frame["projection"]["opponent"],
            frame["projection"]["agent"],
        )
        frame["projection"]["agent"]["player_index"] = 1
        frame["projection"]["opponent"]["player_index"] = 0
    for offer in frame["offers"]:
        offer["actor"] = viewer
    offer = deepcopy(frame["offers"][0])
    command = {
        "command_id": f"command-{ordinal}",
        "match_id": frame["match_id"],
        "expected_revision": revision,
        "prompt_id": 100 + ordinal,
        "offer_id": offer["id"],
        "answers": [],
    }
    return {
        "ordinal": ordinal,
        "viewer": viewer,
        "source": "client" if viewer == 0 else "policy",
        "revision": revision,
        "prompt_id": 100 + ordinal,
        "offer_id": offer["id"],
        "command_id": command["command_id"],
        "presentation_cursor": 0,
        "frame": frame,
        "offer": offer,
        "command": command,
    }


def _replay() -> CanonicalReplayV1:
    return CanonicalReplayV1.model_validate(
        {
            "version": 1,
            "replay_id": "replay-fixture",
            "match_id": FIXTURE["recovery"]["frame"]["match_id"],
            "content_hash": FIXTURE["recovery"]["frame"]["content_hash"],
            "asset_manifest_hash": FIXTURE["recovery"]["frame"]["asset_manifest_hash"],
            "decisions": [_row(0, 0, 2), _row(1, 1, 3)],
            "presentation_tracks": [
                {"viewer": 0, "head": 0, "events": []},
                {"viewer": 1, "head": 0, "events": []},
            ],
        }
    )


def test_address_round_trip_and_authorized_restore_are_exact():
    replay = _replay()
    row = replay.decisions[0]
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()

    assert address.startswith("erd1.")
    assert ReplayDecisionAddress.parse(address).serialize() == address
    restored = restore_decision(replay, address, authorized_viewer=0)
    assert restored.frame == row.frame
    assert restored.offer == row.offer
    assert restored.command == row.command
    assert restored.continuation == []

    with pytest.raises(DecisionNotFoundError):
        restore_decision(replay, address, authorized_viewer=1)


def test_row_address_uses_only_replay_identity_and_exact_decision_row():
    replay = _replay()
    row = replay.decisions[0]

    direct = ReplayDecisionAddress.from_row(
        replay_id=replay.replay_id,
        match_id=replay.match_id,
        row=row,
    )

    assert direct == ReplayDecisionAddress.from_decision(replay, row)
    assert direct.replay_id == replay.replay_id
    assert direct.match_id == replay.match_id
    assert direct.decision_sha256 == row.sha256()


def test_v2_row_address_matches_projection_address_without_reconstruction():
    replay = _replay()
    row = replay.decisions[0]

    direct = DecisionAddressV2.from_row(
        replay_id=replay.replay_id,
        match_id=replay.match_id,
        row=row,
    )

    assert direct == DecisionAddressV2.from_decision(replay, row)
    assert direct.serialize().startswith("ed2.")


@pytest.mark.parametrize("viewer", (0, 1))
def test_shared_viewer_projection_conforms_to_rust_and_python(viewer):
    payload = json.loads(
        (
            PROTOCOL_DIR / "fixtures" / f"canonical-replay-player-{viewer}.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(REPLAY_SCHEMA).validate(payload)
    projection = CanonicalReplayProjectionV1.model_validate(payload)
    assert projection.model_dump(mode="json") == payload
    assert {row.viewer for row in projection.decisions} == {viewer}


def test_missing_drifted_and_malformed_addresses_fail_closed():
    replay = _replay()
    address = ReplayDecisionAddress.from_decision(replay, replay.decisions[0])

    missing = address.model_copy(update={"ordinal": 99}).serialize()
    with pytest.raises(DecisionNotFoundError):
        restore_decision(replay, missing, authorized_viewer=0)

    drifted = address.model_copy(update={"revision": 99}).serialize()
    with pytest.raises(DecisionNotFoundError):
        restore_decision(replay, drifted, authorized_viewer=0)

    for malformed in ("", "erd2.abc", "erd1.invalid!", address.serialize() + "="):
        with pytest.raises(InvalidAddressError):
            ReplayDecisionAddress.parse(malformed)


def test_projection_contains_only_one_viewer_and_preserves_global_ordinals():
    replay = _replay()
    player_zero = project_replay(replay, 0)
    player_one = project_replay(replay, 1)

    assert [row.ordinal for row in player_zero.decisions] == [0]
    assert [row.ordinal for row in player_one.decisions] == [1]
    assert player_zero.decisions[0].viewer == 0
    assert player_one.decisions[0].viewer == 1
    assert "address" in projection_with_addresses(player_zero)["decisions"][0]

    mixed = player_zero.model_dump(mode="json")
    mixed["decisions"].append(player_one.decisions[0].model_dump(mode="json"))
    with pytest.raises(ValidationError, match="mixes viewer"):
        CanonicalReplayProjectionV1.model_validate(mixed)


@pytest.mark.parametrize(
    ("field", "drifted"),
    (
        ("match_id", "different-match"),
        ("content_hash", "0" * 64),
        ("asset_manifest_hash", "0" * 64),
    ),
)
def test_projection_rejects_empty_replay_id_and_drifted_identity(field, drifted):
    projection = project_replay(_replay(), 0).model_dump(mode="json")

    empty_replay_id = deepcopy(projection)
    empty_replay_id["replay_id"] = ""
    with pytest.raises(ValidationError, match="requires replay_id"):
        CanonicalReplayProjectionV1.model_validate(empty_replay_id)

    drifted_projection = deepcopy(projection)
    drifted_projection[field] = drifted
    with pytest.raises(ValidationError, match="identity differs"):
        CanonicalReplayProjectionV1.model_validate(drifted_projection)


def test_authority_rejects_duplicate_and_gapped_global_ordinals():
    replay = _replay().model_dump(mode="json")
    duplicate = deepcopy(replay)
    duplicate["decisions"][1]["ordinal"] = 0
    with pytest.raises(ValidationError, match="ordinal gap"):
        CanonicalReplayV1.model_validate(duplicate)

    gap = deepcopy(replay)
    gap["decisions"][1]["ordinal"] = 2
    with pytest.raises(ValidationError, match="ordinal gap"):
        CanonicalReplayV1.model_validate(gap)


def test_authority_rejects_private_opponent_hand_in_a_decision_frame():
    replay = _replay().model_dump(mode="json")
    replay["decisions"][0]["frame"]["projection"]["opponent"]["hand"] = [
        {
            "id": 99,
            "registry_key": 99,
            "name": "Secret card",
            "zone": "HAND",
            "owner_id": 1,
            "power": 0,
            "toughness": 0,
            "mana_value": 1,
            "types": {
                "is_creature": False,
                "is_land": False,
                "is_spell": True,
                "is_artifact": False,
                "is_enchantment": False,
                "is_planeswalker": False,
                "is_battle": False,
            },
        }
    ]
    with pytest.raises(ValidationError, match="opponent-private hand"):
        CanonicalReplayV1.model_validate(replay)


def test_pinned_match_is_deterministic_complete_and_exactly_restorable():
    first = _play_pinned_match()
    second = _play_pinned_match()
    assert first.model_dump_json() == second.model_dump_json()
    assert [row.ordinal for row in first.decisions] == list(range(len(first.decisions)))
    assert {row.viewer for row in first.decisions} == {0, 1}
    assert [row.revision for row in first.decisions] == sorted(
        row.revision for row in first.decisions
    )

    for viewer in (0, 1):
        projection = project_replay(first, viewer)
        restored_events = []
        for row in projection.decisions:
            address = ReplayDecisionAddress.from_decision(first, row).serialize()
            restored = restore_decision(first, address, viewer)
            assert restored.frame == row.frame
            assert restored.offer == row.offer
            assert restored.command == row.command
            restored_events.extend(restored.continuation)
        assert restored_events == projection.presentation


def test_session_indexes_policy_choices_but_not_automatic_passes(tmp_path):
    session = GameSession(
        tmp_path,
        id_factory=lambda kind: f"test-{kind}",
        clock=lambda: "2026-07-16T00:00:00+00:00",
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 7,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
        }
    )
    for index in range(2_000):
        assert session.obs is not None
        if session.obs.game_over:
            break
        frame = session._experience_frame()
        outcome = session.hero_command(
            {
                "command_id": f"human-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][0]["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
        assert session.published_prompt is None or session.published_prompt.viewer == 0
        policy_ids = [
            row.command_id for row in session.canonical_decisions if row.viewer == 1
        ]
        assert not any(command_id in json.dumps(outcome) for command_id in policy_ids)
    else:
        pytest.fail("deterministic match did not terminate")

    assert session.trace is not None
    deliberate_events = [event for event in session.trace.events if not event.auto]
    automatic_events = [event for event in session.trace.events if event.auto]
    assert automatic_events
    assert len(session.canonical_decisions) == len(deliberate_events)
    assert len(session.trace.events) == session.revision
    assert all(
        row.source.value == "policy"
        for row in session.canonical_decisions
        if row.viewer == 1
    )
    assert all(
        row.source.value == "client"
        for row in session.canonical_decisions
        if row.viewer == 0
    )
    assert all(
        event["caused_by"] is None
        for event in session.canonical_presentation[0]
        if event["from_revision"]
        in {row.revision for row in session.canonical_decisions if row.viewer == 1}
    )


def test_http_serves_only_player_zero_projection_and_strips_authority(
    monkeypatch, tmp_path
):
    replay = _play_pinned_match()
    trace_payload = {
        "config": {},
        "events": [],
        "final_observation": {},
        "winner": None,
        "end_reason": "game_over",
        "timestamp": "2026-07-16T00:00:00+00:00",
        "canonical_replay": replay.model_dump(mode="json"),
    }
    (tmp_path / "canonical.json").write_text(
        json.dumps(trace_payload), encoding="utf-8"
    )
    (tmp_path / "legacy.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in trace_payload.items()
                if key != "canonical_replay"
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()

    with TestClient(app) as client:
        projection_response = client.get("/api/traces/canonical/decisions")
        assert projection_response.status_code == 200
        projection = projection_response.json()
        assert projection["viewer"] == 0
        assert {row["viewer"] for row in projection["decisions"]} == {0}
        address = projection["decisions"][0]["address"]

        restored = client.get(f"/api/traces/canonical/decisions/{address}")
        assert restored.status_code == 200
        assert restored.json()["frame"] == projection["decisions"][0]["frame"]

        malformed = client.get("/api/traces/canonical/decisions/erd1.invalid!")
        assert malformed.status_code == 400
        assert client.get("/api/traces/legacy/decisions").status_code == 409

        ordinary = client.get("/api/traces/canonical?reveal_hidden=true")
        assert ordinary.status_code == 200
        assert "canonical_replay" not in ordinary.json()
