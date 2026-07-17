"""Exact canonical Study fork, structured command, and return proof."""

from __future__ import annotations

import json

import pytest

from etude.replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecisionAddress,
)
from etude.server import GameSession
from etude.study_branch import StudyBranchUnavailableError


def _completed_session(tmp_path) -> tuple[GameSession, CanonicalReplayV1]:
    session = GameSession(
        tmp_path,
        id_factory=lambda kind: f"study-fork-{kind}",
        clock=lambda: "2026-07-17T00:00:00+00:00",
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


def test_historical_address_forks_executes_structured_command_and_returns(tmp_path):
    session, replay = _completed_session(tmp_path)
    row = next(row for row in replay.decisions if row.viewer == 0)
    address = ReplayDecisionAddress.from_decision(replay, row).serialize()
    recorded_replay = json.dumps(
        session.trace.canonical_replay, sort_keys=True, separators=(",", ":")
    )
    recorded_events = list(session.trace.events)

    branch = session.fork_study(address)
    original_offers = branch.structured_offers()
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

    returned = branch.return_to_recorded()
    assert returned.address == address
    assert returned.frame == row.frame
    assert returned.offer == row.offer
    assert returned.command == row.command
    assert returned.presentation_cursor == row.presentation_cursor
    with pytest.raises(StudyBranchUnavailableError, match="returned to replay"):
        branch.structured_offers()

    fresh = session.fork_study(address)
    assert fresh.structured_offers() == original_offers
    assert session.trace.events == recorded_events
    assert (
        json.dumps(session.trace.canonical_replay, sort_keys=True, separators=(",", ":"))
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
