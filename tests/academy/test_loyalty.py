"""Loyalty costs on the offers shared by Academy, Commands and text players."""

from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import reset_to_first_main
import managym
from managym.decision import Command, DecisionFrame
from managym.experience_projection import describe_actions, serialize_observation


def test_theorist_loyalty_offer_and_committed_cost() -> None:
    name = "The Theorist, Jace Beleren"
    deck = {"Island": 30, name: 10}
    env = managym.Env(seed=23)
    seat = reset_to_first_main(
        env,
        managym.GameSetup([managym.PlayerConfig(label, deck) for label in ("a", "b")]),
    )
    env.scenario_clear_hand(seat)
    env.scenario_force_card_in_hand(seat, name)
    env.scenario_force_card_in_hand(seat, "Island")
    for _ in range(4):
        env.scenario_force_battlefield(seat, "Island", ready=True)
    obs = env.scenario_refresh()
    hand = serialize_observation(obs)["agent"]["hand"]
    assert next(card for card in hand if card["name"] == name)["printed_loyalty"] == 3
    cast = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
    )
    obs = env.step(cast)[0]
    while (
        obs.stack_objects
        or obs.action_space.action_space_type == managym.ActionSpaceEnum.PAY_OR_NOT
    ):
        index = next(
            i
            for i, action in enumerate(obs.action_space.actions)
            if action.action_type
            in (
                managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                managym.ActionEnum.PAY_COST,
            )
        )
        obs = env.step(index)[0]
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    plus = next(offer for offer in frame.offers if "[+1]" in offer["label"])
    assert "Illusion" in plus["label"]
    assert plus["label"] in [action["description"] for action in describe_actions(obs)]
    text = capture_text_decision(env)
    assert "printed loyalty 3" in text.text
    assert any("[−2]" in action.description for action in text.actions)
    assert all("[−6]" not in action.description for action in text.actions)
    fork = env.snapshot().restore()
    command = Command("add-loyalty", frame.revision, plus["id"]).to_json()
    assert env.execute_semantic_command_json(
        command
    ) == fork.execute_semantic_command_json(command)
    assert env.state_digest() == fork.state_digest()
    for viewer in (0, 1):
        board = serialize_observation(env.observation_for_player(viewer))
        player = board["agent" if viewer == seat else "opponent"]
        jace = next(
            permanent
            for permanent in player["battlefield"]
            if permanent["name"] == name
        )
        assert jace["counters"]["loyalty"] == 4
        assert jace["loyalty_activated_this_turn"] is True
