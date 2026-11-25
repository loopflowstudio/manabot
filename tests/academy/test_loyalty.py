"""Loyalty costs on the offers shared by Academy, Commands and text players."""

import json

from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.text_observation import reset_to_first_main
import managym
from managym.combat import CombatPlan, PlannedAttack, preview_combat
from managym.decision import Command, DecisionFrame
from managym.enums import ZoneEnum
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


def test_declared_defender_is_bound_to_the_offer_and_survives_cold_replay() -> None:
    walker = "The Theorist, Jace Beleren"
    env = managym.Env(seed=0)
    obs, _ = env.reset(
        [
            managym.PlayerConfig("A", {"Mountain": 16, "Gray Ogre": 16}),
            managym.PlayerConfig("B", {"Island": 16, walker: 16}),
        ]
    )
    for _ in range(200):
        if (
            obs.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
            and any(p.characteristics.name == walker for p in obs.opponent_permanents)
        ):
            break
        index = next(
            (
                i
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                )
                for i, a in enumerate(obs.action_space.actions)
                if a.action_type == kind
            ),
            0,
        )
        obs = env.step(index)[0]
    else:
        raise AssertionError("the deck never offered an attack on Jace")
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert walker in frame.offers[2]["label"]
    assert walker in capture_text_decision(env).actions[2].description
    assert walker in describe_actions(obs)[2]["description"]
    address = dict(
        viewer=frame.actor, revision=frame.revision, fingerprint=frame.fingerprint
    )
    ask = preview_combat(env, **address, plan=CombatPlan())
    creature = ask.declarable[0]
    (defender,) = creature.can_attack
    plan = CombatPlan(
        attackers=[PlannedAttack(**creature.creature.model_dump(), defender=defender)]
    )
    preview = preview_combat(env, **address, plan=plan)
    assert preview.refused is None
    step = next(s for s in preview.steps if s.attack)
    assert step.defender == defender
    subjects = json.loads(env.decision_offer_subjects_json(frame.actor))[
        step.action_index
    ]
    assert [(s["role"], s["object"]["subject"]["id"]) for s in subjects] == [
        ("source", step.creature.model_dump()),
        ("target", defender.model_dump()),
    ]
    bindings = frame.object_candidates
    assert len(bindings) == 1
    command = Command(
        "attack-Jace", frame.revision, step.action_index, object_preconditions=bindings
    )
    fork = env.snapshot().restore()
    assert env.execute_semantic_command_json(
        command.to_json()
    ) == fork.execute_semantic_command_json(command.to_json())
    restored = managym.GameSnapshot.from_replay_json(
        env.snapshot().to_replay_json()
    ).restore()
    assert restored.state_digest() == env.state_digest()


def test_player_only_attack_accepts_the_pre_planeswalker_command() -> None:
    env = managym.Env(seed=0)
    env.reset(
        [
            managym.PlayerConfig("A", {"Mountain": 16, "Gray Ogre": 16}),
            managym.PlayerConfig("B", {"Island": 32}),
        ]
    )
    for _ in range(11):
        env.step(0)
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    fingerprint = "79ec65c4b8de4729d09c2d17681c397967feff7ee99fbb11964603dd9a151073"
    assert frame.fingerprint == fingerprint
    assert [offer["id"] for offer in frame.offers] == [0, 1]
    command = Command(
        "attack-player",
        12,
        0,
        object_preconditions=(
            {
                "decision_fingerprint": fingerprint,
                "offer_id": 0,
                "role": 1,
                "candidate_id": 0,
            },
        ),
    )
    env.execute_semantic_command_json(command.to_json())
    restored = managym.GameSnapshot.from_replay_json(
        env.snapshot().to_replay_json()
    ).restore()
    assert restored.state_digest() == env.state_digest()


def test_empowered_token_visible_to_both_seats_and_cold_replay() -> None:
    env = managym.Env(seed=23)
    deck = {"Island": 20, "Mindseeker Oculus": 12}
    obs, _ = env.reset([managym.PlayerConfig(label, deck) for label in ("a", "b")])
    for _ in range(200):
        if any(
            permanent.characteristics.name == "Jace"
            for permanent in (*obs.agent_permanents, *obs.opponent_permanents)
        ):
            break
        index = next(
            (
                i
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                )
                for i, action in enumerate(obs.action_space.actions)
                if action.action_type == kind
            ),
            0,
        )
        obs = env.step(index)[0]
    else:
        raise AssertionError("Mindseeker Oculus never empowered Jace")
    for viewer in (0, 1):
        board = serialize_observation(
            env.observation_for_player(viewer), include_details=True
        )
        tokens = [
            permanent
            for role in ("agent", "opponent")
            for permanent in board[role]["battlefield"]
            if permanent["name"] == "Jace"
        ]
        assert len(tokens) == 1
        assert tokens[0]["is_token"] is True
        assert tokens[0]["printed_loyalty"] == 0
        assert tokens[0]["counters"]["loyalty"] == 4
    restored = managym.GameSnapshot.from_replay_json(
        env.snapshot().to_replay_json()
    ).restore()
    assert restored.state_digest() == env.state_digest()
    assert restored.semantic_decision_frame_json() == env.semantic_decision_frame_json()


def test_ajani_emblem_visible_to_both_seats_and_restored_from_play() -> None:
    name = "Ajani Resolute"
    env = managym.Env(seed=23)
    obs, _ = env.reset(
        [
            managym.PlayerConfig("Ajani", {"Plains": 24, name: 12}),
            managym.PlayerConfig("Opponent", {"Island": 36}),
        ]
    )
    for _ in range(1200):
        if any(
            card.zone == ZoneEnum.COMMAND
            for card in (*obs.agent_cards, *obs.opponent_cards)
        ):
            break
        actions = obs.action_space.actions
        index = None
        if any(
            a.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
            for a in actions
        ):
            offers = DecisionFrame.from_json(env.semantic_decision_frame_json()).offers
            index = next(
                (
                    offer["id"]
                    for cost in ("[−10]", "[0]")
                    for offer in offers
                    if cost in offer["label"]
                ),
                None,
            )
        if index is None:
            kinds = [managym.ActionEnum.PRIORITY_PLAY_LAND]
            if not any(p.characteristics.name == name for p in obs.agent_permanents):
                kinds.append(managym.ActionEnum.PRIORITY_CAST_SPELL)
            kinds.extend(
                (managym.ActionEnum.PAY_COST, managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            )
            index = next(
                (
                    i
                    for kind in kinds
                    for i, a in enumerate(actions)
                    if a.action_type == kind
                ),
                0,
            )
        obs = env.step(index)[0]
    else:
        raise AssertionError("Ajani never created an emblem")

    for viewer in (0, 1):
        board = serialize_observation(env.observation_for_player(viewer))
        side = board["agent" if viewer == 0 else "opponent"]
        (emblem,) = side["command"]
        assert emblem["name"] == "Emblem — Ajani Resolute"
        assert emblem["text_box"] == "Creatures you control get +2/+2."
        assert all(p["name"] != name for p in side["battlefield"])
    assert "Emblem — Ajani Resolute" in capture_text_decision(env).text
    for restored in (
        env.snapshot().restore(),
        managym.GameSnapshot.from_replay_json(
            env.snapshot().to_replay_json()
        ).restore(),
    ):
        assert restored.state_digest() == env.state_digest()
        assert (
            restored.semantic_decision_frame_json()
            == env.semantic_decision_frame_json()
        )
        assert serialize_observation(restored.observation_for_player(0))["agent"][
            "command"
        ] == [emblem]
