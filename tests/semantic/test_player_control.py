"""Player control crosses native, policy, text and Academy boundaries."""

import json

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.managym.decision import SemanticContractError
from manabot.managym.experience import LegacyHeroObservation
from manabot.managym.experience_projection import decision_surface_view
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym


def _choose(env: managym.Env, kind: managym.ActionEnum) -> None:
    raw = env.observation_for_player(env.current_agent_index())
    index = next(
        i
        for i, action in enumerate(raw.action_space.actions)
        if action.action_type == kind
    )
    env.step(index)


def _controlled_root(paid: bool) -> managym.Env:
    env = priority_root(
        ("Secret of Bloodbending", "Accumulate Wisdom"),
        ("Island",) * 14,
        opponent=("Gray Ogre",),
    )
    env.scenario_clear_hand(1)
    env.scenario_force_card_in_hand(1, "Accumulate Wisdom")
    env.scenario_refresh()
    raw = env.observation_for_player(0)
    source = next(
        c.id
        for c in raw.agent_cards
        if c.name == "Secret of Bloodbending" and int(c.zone) == 1
    )
    index = next(
        i
        for i, a in enumerate(raw.action_space.actions)
        if a.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL and source in a.focus
    )
    env.step(index)
    _choose(
        env, managym.ActionEnum.PAY_COST if paid else managym.ActionEnum.DECLINE_CHOICE
    )
    _choose(env, managym.ActionEnum.CHOOSE_TARGET)
    if paid:
        _choose(env, managym.ActionEnum.PAY_COST)
    for _ in range(160):
        raw = env.observation_for_player(env.current_agent_index())
        if any(frame.kind == 14 for frame in raw.continuation):
            return env
        quiet = next(
            (
                i
                for i, a in enumerate(raw.action_space.actions)
                if a.action_type
                in (
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                    managym.ActionEnum.DECLINE_CHOICE,
                )
            ),
            0,
        )
        env.step(quiet)
    raise AssertionError("delegated decision was not reached")


@pytest.mark.parametrize("paid", [False, True])
def test_bloodbending_text_policy_and_academy_share_authority(paid: bool) -> None:
    env = _controlled_root(paid)
    assert env.current_agent_index() == 0
    raw = env.observation_for_player(0)
    decision = capture_text_decision(env)
    assert "delegated decision" in decision.text
    assert "chooses for" in decision.text
    duration = "turn" if paid else "combat phase"
    assert f"during this {duration}" in decision.text
    assert "Accumulate Wisdom" in decision.text
    expected = ObservationSpace().encode(raw)
    for key, actual in env.encode_observation(raw).items():
        np.testing.assert_array_equal(actual, expected[key], err_msg=key)
    projection = LegacyHeroObservation.model_validate(decision_surface_view(env, 0))
    assert projection.decision_context is not None
    assert projection.decision_context.actor == 0
    assert "chooses for Seat 2" in projection.decision_context.instruction
    assert projection.player_controls is not None
    assert projection.player_controls[0].whole_turn == paid
    assert projection.player_controls[0].effective
    assert any(c.name == "Accumulate Wisdom" for c in projection.agent.inspected or [])
    assert projection.opponent.hand == []
    other = json.loads(env.semantic_observation_json(1))
    assert other["decision"] is None
    assert not other["viewer_state"]["action_space"]["actions"]
    assert not any(c["zone"] == 1 for c in other["viewer_state"]["opponent_cards"])
    selected = next(a for a in decision.actions if "Pass priority" in a.description)
    submit_response(env, decision, selected.reference)
    digest = env.state_digest()
    with pytest.raises(SemanticContractError):
        submit_response(env, decision, selected.reference)
    assert env.state_digest() == digest
