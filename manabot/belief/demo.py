"""Runnable retained-decision proof for the belief-forming manabot seam."""

from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from manabot.belief.agent import AgentMemory, Manabot, ViewerDecision
from manabot.belief.encoding import BeliefEncodingSchema, BeliefRow, encode_belief
from manabot.belief.state import (
    CompatibleDealBeliefModel,
    EmptyBeliefSupport,
    ViewerHistory,
    condition_belief,
    query_mass,
)
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from managym.decision import Command
from managym.possible_worlds import PossibleWorldSpace, WorldQuery


def _retained_observation() -> dict[str, torch.Tensor]:
    encoder = ObservationSpace().encoder
    batch = 1
    observation = {
        "agent_player": torch.zeros((batch, 1, encoder.player_dim)),
        "opponent_player": torch.zeros((batch, 1, encoder.player_dim)),
        "agent_cards": torch.zeros(
            (batch, encoder.cards_per_player, encoder.card_dim)
        ),
        "opponent_cards": torch.zeros(
            (batch, encoder.cards_per_player, encoder.card_dim)
        ),
        "agent_permanents": torch.zeros(
            (batch, encoder.perms_per_player, encoder.permanent_dim)
        ),
        "opponent_permanents": torch.zeros(
            (batch, encoder.perms_per_player, encoder.permanent_dim)
        ),
        "actions": torch.zeros((batch, encoder.max_actions, encoder.action_dim)),
        "action_focus": torch.full(
            (batch, encoder.max_actions, encoder.max_focus_objects),
            -1,
            dtype=torch.long,
        ),
        "agent_player_valid": torch.ones((batch, 1)),
        "opponent_player_valid": torch.ones((batch, 1)),
        "agent_cards_valid": torch.zeros((batch, encoder.cards_per_player)),
        "opponent_cards_valid": torch.zeros((batch, encoder.cards_per_player)),
        "agent_permanents_valid": torch.zeros((batch, encoder.perms_per_player)),
        "opponent_permanents_valid": torch.zeros(
            (batch, encoder.perms_per_player)
        ),
        "actions_valid": torch.zeros((batch, encoder.max_actions)),
    }
    observation["actions"][0, 0, 0] = 1.0
    observation["actions"][0, 1, 1] = 1.0
    observation["actions"][0, 2, 2] = 1.0
    observation["action_focus"][0, :3, 0] = 0
    observation["actions_valid"][0, :3] = 1.0
    return observation


def retained_history() -> ViewerHistory:
    return ViewerHistory(
        schema_identity="managym.viewer-history/v1",
        viewer=0,
        events=("turn:4", "opponent:draw", "viewer:priority"),
    )


def retained_space(history: ViewerHistory) -> PossibleWorldSpace:
    return PossibleWorldSpace.from_fixture(
        viewer=history.viewer,
        source_revision=17,
        source_viewer_state_hash="retained-viewer-state-17",
        source_history_identity=history.identity,
        pool={"Counterspell": 2, "Lightning Bolt": 2, "Mountain": 1},
        hands=(
            ({"Counterspell": 2}, 1),
            ({"Counterspell": 1, "Lightning Bolt": 1}, 4),
            ({"Counterspell": 1, "Mountain": 1}, 2),
            ({"Lightning Bolt": 2}, 1),
            ({"Lightning Bolt": 1, "Mountain": 1}, 2),
        ),
    )


def retained_schema(space: PossibleWorldSpace) -> BeliefEncodingSchema:
    card_ids = {"Counterspell": 0, "Lightning Bolt": 1, "Mountain": 2}
    return BeliefEncodingSchema(
        schema_identity="manabot.belief-tensor/count-marginals-v1",
        world_schema_identity=space.world_schema_identity,
        content_manifest_identity=space.content_manifest_identity,
        rows=tuple(
            sorted(
                BeliefRow(
                    owner_role_id=1,
                    hidden_zone_id=1,
                    card_def_id=card_ids[name],
                    card_name=name,
                )
                for name, _ in space.pool
            )
        ),
        count_buckets=3,
    )


def retained_decision(
    source: Mapping[str, Any] | None = None,
) -> tuple[ViewerDecision, BeliefEncodingSchema]:
    """Project a retained source through the viewer-safe decision boundary."""

    source = source or {}
    history = retained_history()
    space = retained_space(history)
    observation = source.get("viewer_observation", _retained_observation())
    commands = (
        Command("retained-pass", 17, 0),
        Command("retained-cast", 17, 1),
        Command("retained-activate", 17, 2),
    )
    return (
        ViewerDecision(
            observation_identity=space.source_viewer_state_hash,
            observation=observation,
            world_space=space,
            viewer_history=history,
            legal_commands=commands,
        ),
        retained_schema(space),
    )


def retained_manabot(schema: BeliefEncodingSchema) -> Manabot:
    torch.manual_seed(19)
    policy_value = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=schema.count_buckets,
            belief_card_vocab_size=3,
        ),
    )
    return Manabot(
        policy_value=policy_value,
        belief_model=CompatibleDealBeliefModel(),
        belief_schema=schema,
    )


def run_demo() -> dict[str, Any]:
    """Run the six keystone checks and return machine-readable evidence."""

    decision, schema = retained_decision()
    manabot = retained_manabot(schema)
    autonomous = manabot.decide(decision, AgentMemory())
    generated = autonomous.belief_update.belief
    supplied = manabot.evaluate_under_belief(decision, generated)

    has_query = WorldQuery.has("Lightning Bolt")
    lacks_query = WorldQuery.lacks("Lightning Bolt")
    has_bolt = condition_belief(generated, has_query)
    lacks_bolt = condition_belief(generated, lacks_query)
    if isinstance(has_bolt, EmptyBeliefSupport) or isinstance(
        lacks_bolt, EmptyBeliefSupport
    ):
        raise RuntimeError("retained Bolt fixture must have both query supports")
    has_result = manabot.evaluate_under_belief(decision, has_bolt)
    lacks_result = manabot.evaluate_under_belief(decision, lacks_bolt)
    has_view = encode_belief(has_bolt, schema)
    lacks_view = encode_belief(lacks_bolt, schema)
    bolt_row = next(
        index
        for index, row in enumerate(schema.rows)
        if row.card_name == "Lightning Bolt"
    )

    source_a = {
        "actual_hidden_hand": ("Lightning Bolt", "Lightning Bolt"),
        "viewer_observation": _retained_observation(),
    }
    source_b = {
        "actual_hidden_hand": ("Counterspell", "Counterspell"),
        "viewer_observation": _retained_observation(),
    }
    swapped_a, _ = retained_decision(source_a)
    swapped_b, _ = retained_decision(source_b)
    swap_a = manabot.decide(swapped_a, AgentMemory())
    swap_b = manabot.decide(swapped_b, AgentMemory())

    legal_policy = {
        command.command_id: float(autonomous.result.policy[action_index])
        for command, action_index in zip(
            decision.legal_commands,
            autonomous.result.legal_action_indexes,
            strict=True,
        )
    }
    delta = {
        command.command_id: float(
            has_result.policy[action_index] - lacks_result.policy[action_index]
        )
        for command, action_index in zip(
            decision.legal_commands,
            has_result.legal_action_indexes,
            strict=True,
        )
    }
    return {
        "generated_belief_identity": generated.identity,
        "p_has_bolt": query_mass(generated, has_query),
        "legal_action_distribution": legal_policy,
        "value": float(autonomous.result.value),
        "command": autonomous.command.command_id,
        "generated_override_byte_identical": (
            autonomous.result.output_bytes == supplied.output_bytes
        ),
        "bolt_count_token": {
            "has": has_view.count_probabilities[bolt_row].tolist(),
            "lacks": lacks_view.count_probabilities[bolt_row].tolist(),
        },
        "policy_delta_has_minus_lacks": delta,
        "viewer_hidden_swap_identical": (
            swap_a.belief_update.belief.identity == swap_b.belief_update.belief.identity
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
