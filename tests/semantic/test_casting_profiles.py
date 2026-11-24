"""Play printed spell profiles through the viewer-facing Command surface."""

import pytest

from manabot.env.observation import ObservationSpace, column, encode_observation
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym
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
        assert rows[action["index"], column("actions", "spell_profile")] == (
            1 if adventure in action["description"] else 0
        )
    submit_response(env, hand, inset.reference)
    targets = capture_text_decision(env)
    assert "casting its Adventure" in targets.text
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


@pytest.mark.parametrize(
    ("front", "other", "lands", "code", "costs", "back_creature"),
    [
        (
            "Twinmaw Stormbrood",
            "Charring Bite",
            ("Mountain",) * 5 + ("Plains",),
            2,
            (6, 2),
            False,
        ),
        (
            "Monica Rambeau",
            "Photon, Living Light",
            ("Mountain",) * 3 + ("Plains",) * 3,
            3,
            (3, 5),
            True,
        ),
    ],
)
def test_printed_profiles_have_distinct_commands_and_policy_rows(
    front: str,
    other: str,
    lands: tuple[str, ...],
    code: int,
    costs: tuple[int, int],
    back_creature: bool,
) -> None:
    env = priority_root((front,), lands, opponent=("Gray Ogre",))
    actor = env.current_agent_index()
    assert actor is not None
    observation = env.observation_for_player(actor)
    actions = [
        action
        for action in describe_actions(observation)
        if action["type"] == "PRIORITY_CAST_SPELL"
    ]
    rows = encode_observation(ObservationSpace(), observation)["actions"]
    assert {
        (
            action["description"],
            rows[action["index"], column("actions", "spell_profile")],
        )
        for action in actions
    } == {(f"Cast {front}", 0), (f"Cast {other}", code)}
    for name, mana_value, creature in (
        (front, costs[0], True),
        (other, costs[1], back_creature),
    ):
        branch = env.clone_env()
        decision = capture_text_decision(branch)
        [cast] = [
            action
            for action in decision.actions
            if action.description.startswith("Cast ")
            and (other in action.description) == (name == other)
        ]
        submit_response(branch, decision, cast.reference)
        if not creature:
            choose(branch, "Gray Ogre")
        if not branch.observation_for_player(actor).stack_objects:
            choose(branch, "Pay the offered cost")
        for viewer in (0, 1):
            [spell] = branch.observation_for_player(viewer).stack_objects
            characteristics = spell.characteristics
            assert characteristics is not None
            assert characteristics.name == name
            assert characteristics.mana_cost.mana_value == mana_value
            assert characteristics.card_types.is_creature == creature
