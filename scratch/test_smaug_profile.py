"""Play both printed Smaug profiles through the viewer-facing Command surface."""

import managym
from manabot.env.observation import ObservationSpace, column, encode_observation
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
from managym.experience_projection import decision_surface_view, describe_actions


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
    observation = env.observation_for_player(actor)
    rows = encode_observation(ObservationSpace(), observation)["actions"]
    projected = [
        a for a in describe_actions(observation) if a["type"] == "PRIORITY_CAST_SPELL"
    ]
    assert {a["description"] for a in projected} == {
        f"Cast {creature}",
        f"Cast {adventure}",
    }
    for action in projected:
        assert rows[action["index"], column("actions", "adventure")] == (
            adventure in action["description"]
        )
    submit_response(env, hand, inset.reference)
    targets = capture_text_decision(env)
    [ogre] = [action for action in targets.actions if target in action.description]
    submit_response(env, targets, ogre.reference)
    choose(env, "Pay the offered cost")
    choose(env, "Pass priority")
    for viewer in (0, 1):
        [stack] = decision_surface_view(env, viewer)["stack_objects"]
        assert adventure in stack["label"]
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
    assert "permanent spell only" in exile.text
    submit_response(env, exile, recast.reference)
    [spell] = env.observation_for_player(actor).stack_objects
    assert spell.characteristics is not None and spell.characteristics.name == creature
    choose(env, "Pass priority")
    choose(env, "Pass priority")
    board = decision_surface_view(env, actor)
    assert any(card["name"] == creature for card in board["agent"]["battlefield"])
