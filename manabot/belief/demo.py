"""Engine-derived proof for the belief-forming manabot seam."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from manabot.belief.agent import AgentStep
from manabot.belief.encoding import (
    HAND_ZONE_ID,
    LIBRARY_ZONE_ID,
    BeliefEncodingSchema,
    belief_schema_from_engine,
    encode_belief,
)
from manabot.belief.runtime import ManabotPlayer, viewer_decision_from_engine
from manabot.belief.state import EmptyBeliefSupport, condition_belief, query_mass
from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model import Agent
from manabot.verify.util import INTERACTIVE_DECK
from managym.possible_worlds import PossibleWorldSpace, WorldQuery


def _runtime_env() -> tuple[Env, dict[str, np.ndarray]]:
    observation_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="belief-demo-hero",
            villain="belief-demo-villain",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        observation_space,
        Reward(RewardHypers()),
        seed=19,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    observation, _ = env.reset(seed=19)
    return env, observation


def _schema_and_agent(engine: Any, viewer: int) -> tuple[BeliefEncodingSchema, Agent]:
    space = PossibleWorldSpace.from_engine(engine, viewer)
    count_buckets = max(2, max(count for _, count in space.pool) + 1)
    schema = belief_schema_from_engine(
        engine,
        space,
        count_buckets=count_buckets,
    )
    card_vocab_size = 1 + max(row.card_def_id for row in schema.rows)
    torch.manual_seed(19)
    policy_value = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=schema.count_buckets,
            belief_card_vocab_size=card_vocab_size,
        ),
    )
    return schema, policy_value


def _run_swapped_world(
    engine: Any,
    observation: dict[str, np.ndarray],
    viewer: int,
    policy_value: Agent,
    schema: BeliefEncodingSchema,
) -> AgentStep:
    player = ManabotPlayer(policy_value, belief_schema=schema)
    player.start_game(engine, viewer)
    player.act(engine, observation)
    if player.last_step is None:
        raise RuntimeError("materialized ManabotPlayer did not produce an AgentStep")
    return player.last_step


def run_demo() -> dict[str, Any]:
    """Run the keystone checks through one real managym decision."""

    env, observation = _runtime_env()
    viewer = int(env.last_raw_obs.agent.player_index)
    schema, policy_value = _schema_and_agent(env._engine, viewer)
    player = ManabotPlayer(policy_value, belief_schema=schema)
    player.start_game(env, viewer)
    player.act(env, observation)
    if player.last_step is None or player.manabot is None or player.history is None:
        raise RuntimeError("ordinary ManabotPlayer did not produce an AgentStep")
    autonomous = player.last_step
    generated = autonomous.belief_update.belief
    decision = viewer_decision_from_engine(env._engine, observation, player.history)
    supplied = player.manabot.evaluate_under_belief(decision, generated)

    has_query = WorldQuery.has("Lightning Bolt")
    lacks_query = WorldQuery.lacks("Lightning Bolt")
    has_bolt = condition_belief(generated, has_query)
    lacks_bolt = condition_belief(generated, lacks_query)
    if isinstance(has_bolt, EmptyBeliefSupport) or isinstance(
        lacks_bolt, EmptyBeliefSupport
    ):
        raise RuntimeError("engine-derived Bolt root must have both query supports")
    has_result = player.manabot.evaluate_under_belief(decision, has_bolt)
    lacks_result = player.manabot.evaluate_under_belief(decision, lacks_bolt)
    has_view = encode_belief(has_bolt, schema)
    lacks_view = encode_belief(lacks_bolt, schema)
    bolt_rows = {
        "library": next(
            index
            for index, row in enumerate(schema.rows)
            if row.card_name == "Lightning Bolt"
            and row.hidden_zone_id == LIBRARY_ZONE_ID
        ),
        "hand": next(
            index
            for index, row in enumerate(schema.rows)
            if row.card_name == "Lightning Bolt" and row.hidden_zone_id == HAND_ZONE_ID
        ),
    }

    has_indexes, _ = generated.space.condition_indexes(has_query)
    lacks_indexes, _ = generated.space.condition_indexes(lacks_query)
    has_engine = generated.space.materialize(has_indexes[0], seed=101)
    lacks_engine = generated.space.materialize(lacks_indexes[0], seed=101)
    has_observation = has_engine.semantic_observation_json(viewer)
    lacks_observation = lacks_engine.semantic_observation_json(viewer)
    swap_a = _run_swapped_world(has_engine, observation, viewer, policy_value, schema)
    swap_b = _run_swapped_world(lacks_engine, observation, viewer, policy_value, schema)

    legal_actions = tuple(
        zip(
            decision.legal_commands,
            autonomous.result.legal_action_indexes,
            strict=True,
        )
    )
    legal_policy = {
        command.command_id: float(autonomous.result.policy[action_index])
        for command, action_index in legal_actions
    }
    delta = {
        command.command_id: float(
            has_result.policy[action_index] - lacks_result.policy[action_index]
        )
        for command, action_index in legal_actions
    }
    return {
        "generated_belief_identity": generated.identity,
        "viewer_history_identity": player.history.identity,
        "p_has_bolt": query_mass(generated, has_query),
        "legal_action_distribution": legal_policy,
        "value": float(autonomous.result.value),
        "command": autonomous.command.command_id,
        "generated_override_byte_identical": (
            autonomous.result.output_bytes == supplied.output_bytes
        ),
        "bolt_count_tokens": {
            zone: {
                "has": has_view.count_probabilities[index].tolist(),
                "lacks": lacks_view.count_probabilities[index].tolist(),
            }
            for zone, index in bolt_rows.items()
        },
        "policy_delta_has_minus_lacks": delta,
        "viewer_hidden_swap_identical": (
            has_observation == lacks_observation
            and swap_a.belief_update.belief.identity
            == swap_b.belief_update.belief.identity
            and swap_a.result.output_bytes == swap_b.result.output_bytes
        ),
        "receipts": {
            "belief_update": autonomous.belief_update.update_receipt.identity,
            "belief_encoding": autonomous.result.belief_encoding_receipt,
            "policy_inference": autonomous.result.inference_receipt,
        },
        "latency_ms": {
            "belief_update": autonomous.belief_update_seconds * 1000.0,
            "belief_encoding": autonomous.result.encoding_seconds * 1000.0,
            "policy_inference": autonomous.result.inference_seconds * 1000.0,
        },
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
