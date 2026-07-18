"""Exact canonical Study fork, structured command, and return proof."""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from etude.replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecisionAddress,
    restore_decision,
)
from etude.server import GameSession
from etude.study_branch import (
    StudyBranchUnavailableError,
    StudyExecutionReceipt,
    StudyForkProvider,
)
import managym


def _completed_session(tmp_path) -> tuple[GameSession, CanonicalReplayV1]:
    session = GameSession(
        tmp_path,
        id_factory=lambda kind: f"study-fork-{kind}",
        clock=lambda: "2026-07-17T00:00:00+00:00",
        villain_offer_policy=lambda context: int(context.offers[-1]["id"]),
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 7,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": False,
        }
    )
    for index in range(2_000):
        assert session.obs is not None
        if session.obs.game_over:
            break
        frame = session._experience_frame()
        outcome = session.hero_command(
            {
                "command_id": f"study-source-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][0]["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
    else:
        pytest.fail("deterministic Study source match did not terminate")

    session.close("game_over")
    assert session.trace is not None
    assert session.trace.canonical_replay is not None
    return session, CanonicalReplayV1.model_validate(session.trace.canonical_replay)


def _first_cast(
    session: GameSession, replay: CanonicalReplayV1
) -> tuple[object, str, dict, dict]:
    for row in replay.decisions:
        if row.viewer != 0:
            continue
        address = ReplayDecisionAddress.from_decision(replay, row).serialize()
        branch = session.fork_study(address)
        try:
            projection = branch.structured_offers()
        except StudyBranchUnavailableError:
            branch.return_to_recorded()
            continue
        cast = next(
            (offer for offer in projection["offers"] if offer["verb"] == "cast"),
            None,
        )
        branch.return_to_recorded()
        if cast is not None:
            return row, address, projection, cast
    pytest.fail("deterministic Study source has no native cast offer")


def _submission(offer: dict) -> dict:
    return {
        "offer_id": offer["id"],
        "answers": [
            {
                "kind": "candidates",
                "role": choice["role"],
                "candidates": [
                    candidate["id"]
                    for candidate in choice["candidates"]["initial"][
                        : int(choice["min"])
                    ]
                ],
            }
            for choice in offer["choices"]
        ],
    }


def test_historical_address_forks_executes_structured_command_and_returns(tmp_path):
    session, replay = _completed_session(tmp_path)
    row = next(row for row in replay.decisions if row.viewer == 0)
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()
    recorded_replay = json.dumps(
        session.trace.canonical_replay, sort_keys=True, separators=(",", ":")
    )
    recorded_events = list(session.trace.events)

    baseline_return = session.fork_study(address).return_to_recorded()
    branch = session.fork_study(address)
    sibling = session.fork_study(address)
    original_offers = branch.structured_offers()
    sibling_offers = sibling.structured_offers()
    assert sibling_offers == original_offers
    assert original_offers["actor"] == 0
    pass_offer = next(
        offer for offer in original_offers["offers"] if offer["verb"] == "pass_priority"
    )
    observation, _, _, _, _, legacy_actions = branch.submit(
        {"offer_id": pass_offer["id"], "answers": []}
    )

    assert legacy_actions == 1
    assert observation.agent.player_index == 0
    assert observation.opponent.player_index == 1
    assert all(int(card.zone) != 1 for card in observation.opponent_cards)
    assert sibling.structured_offers() == sibling_offers

    returned = branch.return_to_recorded()
    sibling_returned = sibling.return_to_recorded()
    expected = restore_decision(replay, address, authorized_viewer=0)
    assert returned.source_digest == baseline_return.source_digest
    assert sibling_returned.source_digest == baseline_return.source_digest
    assert returned.address == address
    assert returned.frame == expected.frame
    assert returned.offer == expected.offer
    assert returned.command == expected.command
    assert returned.presentation_cursor == expected.presentation_cursor
    assert returned.continuation == expected.continuation
    assert returned.execution.model_dump() == {
        "driver": "full_clone/current_game_v1",
        "command_path": "structured_offers/step_structured_v1",
        "published_offer_sets": 1,
        "accepted_commands": 1,
        "rejected_commands": 0,
        "committed_engine_actions": 1,
        "fallback_commands": 0,
    }
    assert sibling_returned.execution.published_offer_sets == 2
    assert sibling_returned.execution.accepted_commands == 0
    assert sibling_returned.execution.fallback_commands == 0
    assert returned.frame.projection.agent.player_index == returned.viewer
    assert not returned.frame.projection.opponent.hand
    with pytest.raises(StudyBranchUnavailableError, match="returned to replay"):
        branch.structured_offers()

    fresh = session.fork_study(address)
    assert fresh.structured_offers() == original_offers
    fresh_returned = fresh.return_to_recorded()
    assert fresh_returned.source_digest == baseline_return.source_digest
    assert fresh_returned.frame == returned.frame
    assert fresh_returned.offer == returned.offer
    assert fresh_returned.presentation_cursor == returned.presentation_cursor
    assert fresh_returned.continuation == returned.continuation
    assert session.trace.events == recorded_events
    assert not any(session.authority_fallback_counters.values())
    assert (
        json.dumps(
            session.trace.canonical_replay, sort_keys=True, separators=(",", ":")
        )
        == recorded_replay
    )


def test_study_fork_fails_closed_for_invalid_missing_and_other_viewer(tmp_path):
    session, replay = _completed_session(tmp_path)
    player_zero = next(row for row in replay.decisions if row.viewer == 0)
    player_one = next(row for row in replay.decisions if row.viewer == 1)

    with pytest.raises(InvalidAddressError):
        session.fork_study("erd1.invalid!")

    missing = ReplayDecisionAddress.from_decision(replay, player_zero).model_copy(
        update={"ordinal": len(replay.decisions) + 1}
    )
    with pytest.raises(DecisionNotFoundError, match="decision not found"):
        session.fork_study(missing.serialize())

    other_viewer = ReplayDecisionAddress.from_decision(replay, player_one).serialize()
    with pytest.raises(DecisionNotFoundError, match="decision not found"):
        session.fork_study(other_viewer)


def test_submission_failures_are_typed_one_shot_and_zero_fallback(tmp_path):
    session, replay = _completed_session(tmp_path)
    _, address, original_projection, cast = _first_cast(session, replay)
    baseline = session.fork_study(address).return_to_recorded()
    branch = session.fork_study(address)

    branch.structured_offers()
    with pytest.raises(StudyBranchUnavailableError, match="unknown offer"):
        branch.submit({"offer_id": 2**31 - 1, "answers": []})
    with pytest.raises(StudyBranchUnavailableError, match="Publish structured"):
        branch.submit(_submission(cast))

    assert branch.structured_offers() == original_projection
    observation, _, _, _, _, engine_actions = branch.submit(_submission(cast))
    returned = branch.return_to_recorded()

    assert engine_actions == 1
    assert not observation.opponent_cards
    assert returned.source_digest == baseline.source_digest
    assert returned.execution.published_offer_sets == 2
    assert returned.execution.accepted_commands == 1
    assert returned.execution.rejected_commands == 2
    assert returned.execution.committed_engine_actions == 1
    assert returned.execution.fallback_commands == 0


def test_projected_object_incarnation_cannot_replace_native_binding(tmp_path):
    session, replay = _completed_session(tmp_path)
    _, address, _, expected_cast = _first_cast(session, replay)
    branch = session.fork_study(address)
    projected = branch.structured_offers()
    cast = next(offer for offer in projected["offers"] if offer["verb"] == "cast")
    expected_ref = dict(expected_cast["source"]["id"])

    cast["source"]["id"]["incarnation"] += 1_000
    observation, _, _, _, _, _ = branch.submit(_submission(cast))
    returned = branch.return_to_recorded()

    moved = next(
        card for card in observation.agent_cards if card.id == expected_ref["entity"]
    )
    assert int(moved.zone) == int(managym.ZoneEnum.STACK)
    assert expected_ref["incarnation"] >= 0
    assert returned.execution.accepted_commands == 1
    assert returned.execution.fallback_commands == 0


def test_unsupported_native_surface_is_typed_and_returnable(tmp_path):
    session, replay = _completed_session(tmp_path)
    unsupported = next(
        row
        for row in replay.decisions
        if row.viewer == 0 and row.frame.action_space == "DISCARD_THEN_DRAW"
    )
    address = ReplayDecisionAddress.from_decision(replay, unsupported).serialize()
    branch = session.fork_study(address)

    with pytest.raises(
        StudyBranchUnavailableError, match="no native structured offer surface"
    ):
        branch.structured_offers()

    returned = branch.return_to_recorded()
    assert returned.execution.published_offer_sets == 0
    assert returned.execution.accepted_commands == 0
    assert returned.execution.rejected_commands == 0
    assert returned.execution.fallback_commands == 0


def test_retained_root_drift_consumes_return_and_blocks_later_forks(tmp_path):
    session, replay = _completed_session(tmp_path)
    row, address, _, _ = _first_cast(session, replay)
    assert session._study_provider is not None
    retained, _ = session._study_provider._roots[int(row.ordinal)]
    isolated_root = retained.clone_env()
    provider = StudyForkProvider(replay, {int(row.ordinal): isolated_root})
    branch = provider.fork(address, authorized_viewer=0)

    isolated_root.step(0)
    with pytest.raises(StudyBranchUnavailableError, match="root drifted"):
        branch.return_to_recorded()
    with pytest.raises(StudyBranchUnavailableError, match="returned to replay"):
        branch.structured_offers()
    with pytest.raises(StudyBranchUnavailableError, match="root drifted"):
        provider.fork(address, authorized_viewer=0)


def test_execution_receipt_schema_rejects_negative_counts_and_fallback() -> None:
    with pytest.raises(ValidationError):
        StudyExecutionReceipt(
            published_offer_sets=-1,
            accepted_commands=0,
            rejected_commands=0,
            committed_engine_actions=0,
        )
    with pytest.raises(ValidationError):
        StudyExecutionReceipt(
            published_offer_sets=0,
            accepted_commands=0,
            rejected_commands=0,
            committed_engine_actions=0,
            fallback_commands=1,
        )
