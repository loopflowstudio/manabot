"""Reproduce missing continuation facts in the proposed renderer's sole input.

Run with ``uv run python scratch/observation_collision.py``. Scenario helpers
establish one ordinary priority root; all subsequent choices use native Commands.
"""

import json

from managym.decision import Observation, apply_semantic_command

from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import capture_text_decision
import managym


def main() -> None:
    env = managym.Env(seed=23, skip_trivial=False)
    deck = {"Swamp": 20, "Deadly Precision": 20, "Gray Ogre": 4}
    env.reset([managym.PlayerConfig("a", deck), managym.PlayerConfig("b", deck)])
    for _ in range(12):
        contract = SemanticDecisionContract.from_env(env)
        observation = Observation.from_json(
            env.semantic_observation_json(contract.frame.actor)
        )
        if observation.viewer_state["turn"]["phase"] == 1:
            break
        contract.apply(env, contract.frame.find_verb("pass_priority")["id"])
    else:
        raise AssertionError("Priority did not reach the first main phase")
    actor = env.current_agent_index()
    assert actor is not None
    env.scenario_clear_hand(actor)
    env.scenario_force_card_in_hand(actor, "Deadly Precision")
    env.scenario_force_battlefield(actor, "Swamp", ready=True)
    env.scenario_force_battlefield(actor, "Gray Ogre", ready=True)
    env.scenario_force_battlefield(1 - actor, "Gray Ogre", ready=True)
    env.scenario_refresh()
    contract = SemanticDecisionContract.from_env(env)
    target_input = contract.apply(
        env, contract.frame.find_verb("cast")["id"]
    ).observation
    assert target_input.decision is not None
    offers = target_input.decision.offers
    assert len(offers) == 2

    contract = SemanticDecisionContract(target_input.decision)
    branches: list[managym.Env] = []
    for offer in offers:
        branch = env.clone_env()
        contract.apply(branch, offer["id"])
        branches.append(branch)

    left, right = branches
    left_input = json.loads(left.semantic_observation_json(actor))
    right_input = json.loads(right.semantic_observation_json(actor))
    left_context = json.loads(left.decision_context_json(actor))
    right_context = json.loads(right.decision_context_json(actor))
    assert left_context["stage"] == right_context["stage"] == "additional_cost"
    assert left_context["targets"] != right_context["targets"]
    assert left_input == right_input, "Expected identical complete renderer inputs"

    captures = [capture_text_decision(branch, history_capacity=32) for branch in branches]
    assert captures[0].capture_digest != captures[1].capture_digest
    assert captures[0].text != captures[1].text

    # Even the same bound Command is accepted in both branches, but advances
    # different continuations. No private state is inspected by this proof.
    contract = SemanticDecisionContract.from_env(left)
    sacrifice_offer = contract.frame.find_verb("select_card")
    command = contract.command(sacrifice_offer["id"])
    left_after, right_after = [
        apply_semantic_command(branch, command).observation for branch in branches
    ]
    assert (
        left_after.viewer_state["stack_objects"]
        != right_after.viewer_state["stack_objects"]
    )

    print(
        json.dumps(
            {
                "complete_observations_equal": True,
                "new_capture_digests_differ": True,
                "new_capture_text_differs": True,
                "schema_version": left_input["identity"]["schema_version"],
                "revision": left_input["identity"]["revision"],
                "stage": left_context["stage"],
                "left_targets": left_context["targets"],
                "right_targets": right_context["targets"],
                "same_command_accepted_in_both": True,
                "resulting_stack_targets_differ": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
