"""Pending source power is public with decision history disabled."""

import pytest

from manabot.env.observation import ObservationSpace, column
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym
from managym.experience_projection import decision_surface_view
from tests.fixtures.observation_rows import observation_arrays


def choose(env: managym.Env, description: str) -> None:
    decision = capture_text_decision(env, history_capacity=0)
    (action,) = [a for a in decision.actions if description in a.description]
    submit_response(env, decision, action.reference)


def assert_public_pending_power(env: managym.Env, power: int) -> None:
    for viewer in (0, 1):
        obs = env.observation_for_player(viewer)
        (ability,) = obs.stack_objects
        assert ability.source_power == power
        restored = managym.Observation.from_json(obs.toJSON())
        assert restored.stack_objects[0].source_power == power
        stack = observation_arrays(ObservationSpace(), obs)["stack"]
        assert stack[0, column("stack", "source_power")] == pytest.approx(power / 10)
        (entry,) = decision_surface_view(env, viewer)["stack_objects"]
        assert f"source power {power}" in entry["label"]
    assert (
        f"source power {power}" in capture_text_decision(env, history_capacity=0).text
    )


@pytest.mark.parametrize("power", [2, 3])
def test_pending_immolator_power_matches_damage(power: int) -> None:
    # Leave one untapped Mountain for the activation in both positions.
    lands = ("Mountain", "Mountain") if power == 3 else ("Mountain",)
    env = priority_root(
        ("Lightning Bolt",),
        ("Heartfire Immolator", *lands),
        opponent=("Craw Wurm",),
    )
    if power == 3:
        choose(env, "Lightning Bolt")
        choose(env, "Target Opponent")
        choose(env, "Pay the offered cost")
        # Resolve prowess, then Lightning Bolt.
        for _ in range(4):
            choose(env, "Pass priority")

    choose(env, "Activate ability")
    choose(env, "Craw Wurm")

    assert_public_pending_power(env, power)

    choose(env, "Pass priority")
    choose(env, "Pass priority")
    (wurm,) = env.observation_for_player(0).opponent_permanents
    assert wurm.damage == power


@pytest.mark.parametrize(
    ("source", "power"), [("Ashroot Animist", 4), ("Halana and Alena, Partners", 2)]
)
def test_pending_combat_trigger_power_reaches_each_viewer(
    source: str, power: int
) -> None:
    env = priority_root((), (source, "Gray Ogre", "Hill Giant"))
    for _ in range(24):
        decision = capture_text_decision(env, history_capacity=0)
        if env.observation_for_player(0).stack_objects:
            break
        actions = decision.actions
        action = next(
            (
                a
                for a in actions
                if "Gray Ogre" in a.description and "Target" in a.description
            ),
            None,
        )
        if action is None:
            action = next(
                a
                for a in actions
                if "Attack with " in a.description or "Pass priority" in a.description
            )
        submit_response(env, decision, action.reference)
    else:
        pytest.fail(f"{source} did not put its targeted ability on the stack")

    assert_public_pending_power(env, power)
