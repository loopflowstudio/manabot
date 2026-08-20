"""Real-engine bounded overfit proof for supervised belief learning."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np
import torch

from manabot.belief.agent import AgentMemory, Manabot
from manabot.belief.demo import _runtime_env
from manabot.belief.encoding import belief_schema_from_engine
from manabot.belief.learning import (
    capture_materialized_world_supervision,
    pack_belief_examples,
    train_bounded_overfit,
)
from manabot.belief.runtime import viewer_decision_from_engine
from manabot.belief.state import CompatibleDealBeliefModel, ViewerHistory
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from managym.decision import Command, DecisionFrame, Observation
from managym.possible_worlds import PossibleWorldSpace


def run_demo(*, steps: int = 80, seed: int = 197) -> dict[str, Any]:
    """Overfit one materialized world, then use the model in an ordinary decision."""

    env, _ = _runtime_env()
    engine = env._engine
    frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
    viewer = (frame.actor + 1) % 2
    initial_observation = Observation.from_json(
        engine.semantic_observation_json(viewer)
    )
    initial_history = ViewerHistory.from_observation(initial_observation)
    initial_space = PossibleWorldSpace.from_engine(engine, viewer)

    public_offer = next(offer for offer in frame.offers if offer["verb"] == "play_land")
    public_actor = frame.actor
    _, _, terminated, truncated, _, transition = env.step_semantic(
        Command(
            command_id="belief-learning-demo-public-event",
            expected_revision=frame.revision,
            offer_id=int(public_offer["id"]),
        )
    )
    if terminated or truncated:
        raise RuntimeError("belief-learning demo ended before its supervised root")
    current_observation = Observation.from_json(
        engine.semantic_observation_json(viewer)
    )
    history = initial_history.advance(
        transition.receipt,
        current_observation,
        acting=public_actor,
    )
    if not history.semantic_events:
        raise RuntimeError("belief-learning demo needs an informative public event")
    for pass_index in range(64):
        frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
        if frame.actor == viewer:
            encoded_observation = env.obs_space.encode(
                engine.observation_for_player(viewer)
            )
            break
        pass_offer = next(
            offer for offer in frame.offers if offer["verb"] == "pass_priority"
        )
        _, _, terminated, truncated, _, transition = env.step_semantic(
            Command(
                command_id=f"belief-learning-demo-pass-{pass_index}",
                expected_revision=frame.revision,
                offer_id=int(pass_offer["id"]),
            )
        )
        if terminated or truncated:
            raise RuntimeError("belief-learning demo ended before the viewer decision")
        current_observation = Observation.from_json(
            engine.semantic_observation_json(viewer)
        )
        history = history.advance(
            transition.receipt,
            current_observation,
            acting=frame.actor,
        )
    else:
        raise RuntimeError("belief-learning demo did not reach the viewer decision")
    space = PossibleWorldSpace.from_engine(engine, viewer)
    count_buckets = max(2, max(count for _, count in space.pool) + 1)
    schema = belief_schema_from_engine(
        engine,
        space,
        count_buckets=count_buckets,
    )
    example = capture_materialized_world_supervision(
        engine,
        world_space=space,
        viewer_history=history,
        schema=schema,
    )
    packed = pack_belief_examples((example,), schema)
    p0_probability = float(packed.log_p0[int(example.target_world)].exp().item())
    trained = train_bounded_overfit((example,), schema, steps=steps, seed=seed)
    learned = trained.model.update(
        previous=None,
        world_space=space,
        viewer_history=history,
    ).belief
    learned_probability = learned.probability_at(example.target_world)

    artifact_variant_history = replace(
        history,
        initial_observation_identity="f" * 64,
        events=tuple(f"{index:064x}" for index, _ in enumerate(history.events, 1)),
    )
    artifact_variant = trained.model.update(
        previous=None,
        world_space=space,
        viewer_history=artifact_variant_history,
    ).belief
    no_semantic_history = replace(history, semantic_events=())
    no_semantic_update = trained.model.update(
        previous=None,
        world_space=space,
        viewer_history=no_semantic_history,
    ).belief
    reference_current = (
        CompatibleDealBeliefModel()
        .update(
            previous=None,
            world_space=space,
            viewer_history=no_semantic_history,
        )
        .belief
    )

    reference_initial = (
        CompatibleDealBeliefModel()
        .update(
            previous=None,
            world_space=initial_space,
            viewer_history=initial_history,
        )
        .belief
    )
    learned_initial = trained.model.update(
        previous=None,
        world_space=initial_space,
        viewer_history=initial_history,
    ).belief

    card_vocab_size = 1 + max(row.card_def_id for row in schema.rows)
    torch.manual_seed(seed)
    policy_value = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=schema.count_buckets,
            belief_card_vocab_size=card_vocab_size,
        ),
    )
    manabot = Manabot(
        policy_value=policy_value,
        belief_model=trained.model,
        belief_schema=schema,
    )
    decision = viewer_decision_from_engine(engine, encoded_observation, history)
    autonomous = manabot.decide(decision, AgentMemory())
    supplied = manabot.evaluate_under_belief(decision, autonomous.belief_update.belief)

    alternate_index = next(
        world.index for world in space.worlds if world.index != example.target_world
    )
    alternate_engine = space.materialize(alternate_index, seed=seed + 1)
    alternate_observation = alternate_engine.semantic_observation_json(viewer)
    alternate_encoded_observation = env.obs_space.encode(
        alternate_engine.observation_for_player(viewer)
    )
    alternate_decision = viewer_decision_from_engine(
        alternate_engine,
        alternate_encoded_observation,
        history,
    )
    alternate = manabot.decide(alternate_decision, AgentMemory())

    return {
        "proof": "fresh-model-supervised-exact-world-overfit",
        "training_steps": steps,
        "training_examples": 1,
        "support_size": space.support_size,
        "viewer": viewer,
        "viewer_history_events": len(history.semantic_events),
        "viewer_history_event_identities": len(history.events),
        "semantic_public_history": [
            event.to_payload() for event in history.semantic_events
        ],
        "semantic_history_schema_identity": trained.model.history_schema_identity,
        "history_representation": "typed-public-commitment-events",
        "first_informative_action_by_opponent": public_actor != viewer,
        "inference_inputs": ["possible_world_space", "viewer_history"],
        "supervision_access": "authority-only-materialized-world",
        "supervision_receipt": example.supervision_receipt,
        "target_world": example.target_world,
        "p0_true_world_probability": p0_probability,
        "learned_true_world_probability": learned_probability,
        "initial_nll": trained.initial_nll,
        "final_nll": trained.final_nll,
        "nll_improvement": trained.initial_nll - trained.final_nll,
        "fresh_model_started_at_p0": bool(
            np.isclose(
                trained.initial_nll,
                -np.log(p0_probability),
                atol=1e-10,
                rtol=0.0,
            )
        ),
        "preinformative_exact_p0": bool(
            np.array_equal(
                reference_initial.probabilities,
                learned_initial.probabilities,
            )
        ),
        "artifact_identity_invariant": (
            artifact_variant_history.identity != history.identity
            and artifact_variant.digest == learned.digest
        ),
        "semantic_history_changes_distribution": (
            np.array_equal(
                no_semantic_update.probabilities,
                reference_current.probabilities,
            )
            and not np.array_equal(
                no_semantic_update.probabilities,
                learned.probabilities,
            )
        ),
        "ordinary_agent_used_learned_belief": (
            autonomous.belief_update.belief.digest == learned.digest
        ),
        "generated_override_byte_identical": (
            autonomous.result.output_bytes == supplied.output_bytes
        ),
        "viewer_hidden_swap_identical": (
            engine.semantic_observation_json(viewer) == alternate_observation
            and alternate_decision.world_space.identity == space.identity
            and alternate.belief_update.belief.digest == learned.digest
            and alternate.result.output_bytes == autonomous.result.output_bytes
            and alternate.command == autonomous.command
        ),
        "model_identity": trained.model.identity,
        "belief_update_receipt": autonomous.belief_update.update_receipt.identity,
        "command": autonomous.command.command_id,
    }


def main(*, steps: int = 80, seed: int = 197) -> None:
    print(json.dumps(run_demo(steps=steps, seed=seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
