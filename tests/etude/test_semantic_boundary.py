from __future__ import annotations

import pytest

from etude.semantic_boundary import SemanticExperienceBoundary
import managym
from managym.decision import SEMANTIC_DECISION_VERSION, SemanticContractError


def _env() -> managym.Env:
    env = managym.Env(seed=29, skip_trivial=True)
    deck = {"Mountain": 10, "Gray Ogre": 20, "Lightning Bolt": 4}
    env.reset(
        [
            managym.PlayerConfig("hero", dict(deck)),
            managym.PlayerConfig("villain", dict(deck)),
        ]
    )
    return env


def test_boundary_projects_a_viewer_safe_observation() -> None:
    env = _env()
    boundary = SemanticExperienceBoundary()
    frame = boundary.decision_frame(env)
    actor = frame.actor

    observation = boundary.observe(env, actor)
    assert observation.schema_version == SEMANTIC_DECISION_VERSION
    assert observation.viewer == actor
    assert observation.revision == frame.revision
    assert observation.decision is not None
    # Viewer safety: no opponent-private hand identities leak into the state.
    assert observation.opponent_hand_is_hidden()
    # The viewer state is the viewer-safe projection, not a full authority dump.
    assert "opponent_cards" in observation.viewer_state
    assert "agent_cards" in observation.viewer_state


def test_boundary_suppresses_the_decision_for_a_non_acting_viewer() -> None:
    env = _env()
    boundary = SemanticExperienceBoundary()
    actor = boundary.decision_frame(env).actor
    other = 1 - actor

    observation = boundary.observe(env, other)
    assert observation.viewer == other
    assert observation.decision is None, "non-acting viewer must not see the decision"
    assert observation.opponent_hand_is_hidden()


def test_boundary_applies_a_command_and_binds_the_receipt_revisions() -> None:
    env = _env()
    boundary = SemanticExperienceBoundary()
    frame = boundary.decision_frame(env)
    pass_offer = frame.find_verb("pass_priority")

    command = {
        "command_id": "etude:boundary:1",
        "expected_revision": frame.revision,
        "offer_id": pass_offer["id"],
        "answers": [],
    }
    transition = boundary.apply(env, command)
    receipt = transition.receipt

    assert receipt.command_id == "etude:boundary:1"
    assert receipt.before_revision == frame.revision
    assert receipt.after_revision > receipt.before_revision
    # The next observation is viewer-safe for the command's actor.
    assert transition.observation.viewer == frame.actor
    assert transition.observation.opponent_hand_is_hidden()


def test_boundary_fails_closed_on_a_stale_command_without_mutation() -> None:
    env = _env()
    boundary = SemanticExperienceBoundary()
    frame = boundary.decision_frame(env)
    pass_offer = frame.find_verb("pass_priority")

    # Advance the authority once.
    boundary.apply(
        env,
        {
            "command_id": "advance",
            "expected_revision": frame.revision,
            "offer_id": pass_offer["id"],
            "answers": [],
        },
    )
    after = boundary.decision_frame(env).revision
    assert after > frame.revision

    # A command bound to the stale frame revision must fail closed.
    stale = {
        "command_id": "stale",
        "expected_revision": frame.revision,
        "offer_id": pass_offer["id"],
        "answers": [],
    }
    with pytest.raises(SemanticContractError, match="stale"):
        boundary.apply(env, stale)
    assert boundary.decision_frame(env).revision == after, (
        "no mutation on stale command"
    )
