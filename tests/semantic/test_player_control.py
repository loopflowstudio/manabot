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


def _controlled_root(
    paid: bool, *, caster: int = 0, controlled_hand: str = "Accumulate Wisdom"
) -> managym.Env:
    env = priority_root(
        ("Secret of Bloodbending", controlled_hand),
        ("Island",) * 14,
        opponent=("Gray Ogre",),
    )
    if caster == 1:
        for _ in range(80):
            raw = env.observation_for_player(env.current_agent_index())
            if (
                raw.agent.player_index == caster
                and raw.turn.active_player_id == raw.agent.id
                and int(raw.turn.phase) == 1
            ):
                break
            env.step(
                next(
                    (
                        i
                        for i, action in enumerate(raw.action_space.actions)
                        if action.action_type
                        == managym.ActionEnum.PRIORITY_PASS_PRIORITY
                    ),
                    0,
                )
            )
        else:
            raise AssertionError("reversed caster's main phase was not reached")
        env.scenario_clear_hand(caster)
        env.scenario_force_card_in_hand(caster, "Secret of Bloodbending")
        for _ in range(14):
            env.scenario_force_battlefield(caster, "Island", ready=True)
    victim = 1 - caster
    env.scenario_clear_hand(victim)
    env.scenario_force_card_in_hand(victim, controlled_hand)
    for _ in range(2):
        env.scenario_force_battlefield(victim, "Island", ready=True)
    env.scenario_refresh()
    raw = env.observation_for_player(caster)
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


@pytest.mark.parametrize("paid", [False, True])
@pytest.mark.parametrize("caster", [0, 1])
def test_delegated_search_keeps_viewer_and_resource_ownership(
    paid: bool, caster: int
) -> None:
    env = _controlled_root(paid, caster=caster, controlled_hand="Giant Koi")
    victim = 1 - caster
    # Islandcycling is legal in both a combat-only and a whole-turn interval.
    before = LegacyHeroObservation.model_validate(decision_surface_view(env, caster))
    assert before.decision_context is not None
    assert f"chooses for Seat {victim + 1}" in before.decision_context.instruction
    _choose(env, managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY)
    own_response = False
    for _ in range(20):
        projection = LegacyHeroObservation.model_validate(
            decision_surface_view(env, caster)
        )
        context = projection.decision_context
        assert context is not None
        assert projection.agent.player_index == caster
        assert projection.opponent.player_index == victim
        if context.stage == "private_search":
            break
        own_response |= "chooses for" not in context.instruction
        _choose(env, managym.ActionEnum.PRIORITY_PASS_PRIORITY)
    else:
        raise AssertionError("delegated Islandcycling search was not reached")
    assert own_response, "the chooser still makes their own priority decisions"
    assert context.actor == caster
    assert f"chooses for Seat {victim + 1}" in context.instruction
    assert "permitted library candidates" in context.instruction
    assert projection.opponent.hand == []
    library = [c for c in projection.agent.inspected or [] if c.zone == "LIBRARY"]
    assert library and all(c.owner_id == projection.opponent.id for c in library)
    assert any(c.name == "Island" for c in library)
    # The same native identity recovers the exact search; no UI ownership changes.
    fork = env.clone_env()
    assert decision_surface_view(fork, caster) == decision_surface_view(env, caster)
    _choose(env, managym.ActionEnum.SELECT_CARD)
    after = LegacyHeroObservation.model_validate(decision_surface_view(env, caster))
    assert not any(c.zone == "LIBRARY" for c in after.agent.inspected or [])
    assert after.agent.player_index == caster
    assert after.opponent.known_hand
