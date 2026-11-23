"""Directed Adventure fixture pending the first playable Adventure definition."""

import managym
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
from managym.experience_projection import decision_surface_view


def choose(env: managym.Env, wording: str) -> None:
    decision = capture_text_decision(env)
    [action] = [action for action in decision.actions if wording in action.description]
    submit_response(env, decision, action.reference)


def test_spew_flame_resolution_grants_only_the_smaug_profile() -> None:
    creature = "Smaug, the Great Calamity"
    adventure = "Spew Flame"
    target = "Gray Ogre"
    env = priority_root(
        (creature, "Counterspell"),
        ("Mountain",) * 12,
        opponent=(target, "Island", "Island"),
    )
    actor = env.current_agent_index()
    assert actor is not None
    env.scenario_clear_hand(1 - actor)
    env.scenario_force_card_in_hand(1 - actor, "Counterspell")
    env.scenario_refresh()
    hand = capture_text_decision(env)
    casts = [
        action for action in hand.actions if action.description.startswith("Cast ")
    ]
    assert len(casts) == 2
    [inset] = [action for action in casts if adventure in action.description]
    assert any(creature in action.description for action in casts if action != inset)
    submit_response(env, hand, inset.reference)
    targets = capture_text_decision(env)
    [ogre] = [action for action in targets.actions if target in action.description]
    submit_response(env, targets, ogre.reference)
    choose(env, "Pass priority")
    countered = env.clone_env()
    choose(countered, "Counterspell")
    choose(countered, adventure)
    choose(countered, "Pass priority")
    choose(countered, "Pass priority")
    after_counter = capture_text_decision(countered)
    counter_board = decision_surface_view(countered, actor)
    assert any(
        card["name"] == target for card in counter_board["opponent"]["battlefield"]
    )
    assert not any(creature in action.description for action in after_counter.actions)
    assert any(
        creature in line and "zone graveyard" in line
        for line in after_counter.text.splitlines()
    )

    choose(env, "Pass priority")
    exile = capture_text_decision(env)
    resolved_board = decision_surface_view(env, actor)
    assert any(card["name"] == creature for card in resolved_board["agent"]["exile"])
    assert any(
        card["name"] == target for card in resolved_board["opponent"]["graveyard"]
    )
    [recast] = [
        action for action in exile.actions if action.description.startswith("Cast ")
    ]
    assert creature in recast.description and adventure not in recast.description
    submit_response(env, exile, recast.reference)
