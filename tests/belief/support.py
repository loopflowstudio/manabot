"""Focused test fixtures for belief contracts; production demos use managym."""

import torch

from manabot.belief import (
    HAND_ZONE_ID,
    LIBRARY_ZONE_ID,
    OPPONENT_OWNER_ROLE_ID,
    BeliefEncodingSchema,
    BeliefRow,
    CompatibleDealBeliefModel,
    Manabot,
    ViewerDecision,
    ViewerHistory,
)
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from managym.decision import Command, Observation
from managym.possible_worlds import PossibleWorldSpace


def fixture_history() -> ViewerHistory:
    return ViewerHistory.from_observation(
        Observation(
            schema_version=4,
            revision=17,
            viewer=0,
            viewer_state_hash="retained-viewer-state-17",
            viewer_state={},
            events=("opening-priority",),
            decision=None,
        )
    )


def fixture_space(history: ViewerHistory | None = None) -> PossibleWorldSpace:
    history = history or fixture_history()
    return PossibleWorldSpace.from_fixture(
        viewer=history.viewer,
        source_revision=history.current_revision,
        source_viewer_state_hash=history.current_viewer_state_hash,
        pool={"Counterspell": 2, "Lightning Bolt": 2, "Mountain": 1},
        hands=(
            ({"Counterspell": 2}, 1),
            ({"Counterspell": 1, "Lightning Bolt": 1}, 4),
            ({"Counterspell": 1, "Mountain": 1}, 2),
            ({"Lightning Bolt": 2}, 1),
            ({"Lightning Bolt": 1, "Mountain": 1}, 2),
        ),
    )


def fixture_schema(space: PossibleWorldSpace) -> BeliefEncodingSchema:
    card_ids = {name: index for index, (name, _) in enumerate(space.pool)}
    return BeliefEncodingSchema(
        schema_identity="manabot.belief-tensor/test-v1",
        world_schema_identity=space.world_schema_identity,
        content_manifest_identity=space.content_manifest_identity,
        rows=tuple(
            sorted(
                BeliefRow(
                    owner_role_id=OPPONENT_OWNER_ROLE_ID,
                    hidden_zone_id=zone_id,
                    card_def_id=card_ids[name],
                    card_name=name,
                )
                for name, _ in space.pool
                for zone_id in (LIBRARY_ZONE_ID, HAND_ZONE_ID)
            )
        ),
        count_buckets=3,
    )


def fixture_torch_observation() -> dict[str, torch.Tensor]:
    encoder = ObservationSpace().encoder
    batch = 1
    observation = {
        "agent_player": torch.zeros((batch, 1, encoder.player_dim)),
        "opponent_player": torch.zeros((batch, 1, encoder.player_dim)),
        "agent_cards": torch.zeros((batch, encoder.cards_per_player, encoder.card_dim)),
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
        "opponent_permanents_valid": torch.zeros((batch, encoder.perms_per_player)),
        "actions_valid": torch.zeros((batch, encoder.max_actions)),
    }
    observation["actions"][0, 0, 0] = 1.0
    observation["actions"][0, 1, 1] = 1.0
    observation["actions"][0, 2, 2] = 1.0
    observation["action_focus"][0, :3, 0] = 0
    observation["actions_valid"][0, :3] = 1.0
    return observation


def fixture_decision() -> tuple[ViewerDecision, BeliefEncodingSchema]:
    history = fixture_history()
    space = fixture_space(history)
    return (
        ViewerDecision(
            observation=fixture_torch_observation(),
            world_space=space,
            viewer_history=history,
            legal_commands=(
                Command("retained-pass", 17, 0),
                Command("retained-cast", 17, 1),
                Command("retained-activate", 17, 2),
            ),
        ),
        fixture_schema(space),
    )


def fixture_manabot(schema: BeliefEncodingSchema) -> Manabot:
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
