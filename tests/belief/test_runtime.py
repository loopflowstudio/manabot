"""Focused checks for the ordinary ManabotPlayer runtime boundary."""

import json

import pytest
import torch

from manabot.belief import (
    BeliefError,
    ManabotPlayer,
    belief_schema_from_engine,
)
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent, make_player
from managym.possible_worlds import PossibleWorldSpace
from tests.belief.support import fixture_torch_observation


class NativeContractEngine:
    """Small current-ABI engine double; the CLI demo remains the native proof."""

    def __init__(self, *, remap_card_ids: bool = False) -> None:
        self.revision = 17
        self.viewer_hash = "runtime-viewer-17"
        definitions = [
            {"card_def_id": 0, "registry_name": "Counterspell"},
            {"card_def_id": 1, "registry_name": "Lightning Bolt"},
            {"card_def_id": 2, "registry_name": "Mountain"},
        ]
        if remap_card_ids:
            definitions[0]["registry_name"], definitions[1]["registry_name"] = (
                definitions[1]["registry_name"],
                definitions[0]["registry_name"],
            )
        self.definitions = definitions
        self.worlds = (
            ({"Counterspell": 2}, 1),
            ({"Counterspell": 1, "Lightning Bolt": 1}, 4),
            ({"Counterspell": 1, "Mountain": 1}, 2),
            ({"Lightning Bolt": 2}, 1),
            ({"Lightning Bolt": 1, "Mountain": 1}, 2),
        )

    def content_pack_manifest(self) -> dict:
        return {
            "schema_version": 1,
            "content_digest": "runtime-content-v1",
            "definitions": self.definitions,
        }

    def semantic_observation_json(self, viewer: int) -> str:
        offers = [{"id": index, "label": f"action-{index}"} for index in range(3)]
        return json.dumps(
            {
                "identity": {
                    "schema_version": 4,
                    "revision": self.revision,
                    "viewer": viewer,
                    "viewer_state_hash": self.viewer_hash,
                },
                "viewer_state": {},
                "events": [],
                "decision": {
                    "schema_version": 4,
                    "revision": self.revision,
                    "actor": 0,
                    "fingerprint": "runtime-frame-17",
                    "offers": offers,
                    "object_candidates": [],
                }
                if viewer == 0
                else None,
            }
        )

    def possible_world_space_json(self, viewer: int) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "identity": "runtime-space-17",
                "viewer": viewer,
                "opponent": 1 - viewer,
                "source_observation": {
                    "schema_version": 4,
                    "revision": self.revision,
                    "viewer": viewer,
                    "viewer_state_hash": self.viewer_hash,
                },
                "hand_size": 2,
                "pool": {
                    "Counterspell": 2,
                    "Lightning Bolt": 2,
                    "Mountain": 1,
                },
                "total_weight": "10",
                "worlds": [
                    {"index": index, "hand": hand, "weight": str(weight)}
                    for index, (hand, weight) in enumerate(self.worlds)
                ],
            }
        )


def _belief_agent() -> Agent:
    return Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=3,
            belief_card_vocab_size=3,
        ),
    )


def test_manabot_player_generates_belief_before_ordinary_action() -> None:
    engine = NativeContractEngine()
    player = ManabotPlayer(_belief_agent())

    player.start_game(engine, 0)
    action = player.act(engine, fixture_torch_observation())

    assert action in {0, 1, 2}
    assert player.last_step is not None
    assert player.last_step.belief_update.belief.normalization_error < 1e-12
    command = player.command_for_action(engine, action, command_id="runtime-test")
    assert command.command_id == "runtime-test"
    assert command.offer_id == action


def test_serialized_belief_checkpoint_binds_exact_runtime_schema(tmp_path) -> None:
    engine = NativeContractEngine()
    agent = _belief_agent()
    schema = belief_schema_from_engine(
        engine,
        PossibleWorldSpace.from_engine(engine, 0),
        count_buckets=agent.belief_count_buckets,
    )
    checkpoint_path = tmp_path / "belief.pt"
    save_bc_checkpoint(
        agent,
        agent.observation_space,
        checkpoint_path,
        belief_schema=schema,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["belief_schema_identity"] == schema.identity
    assert (
        payload["belief_content_manifest_identity"] == schema.content_manifest_identity
    )
    player, _ = make_player(
        {"kind": "checkpoint", "path": str(checkpoint_path)}, seed=1
    )
    assert isinstance(player, ManabotPlayer)
    player.start_game(engine, 0)
    assert player.manabot is not None

    with pytest.raises(BeliefError, match="belief schema"):
        player.start_game(NativeContractEngine(remap_card_ids=True), 0)
    assert player.manabot is None
    assert player.viewer is None
    assert player.history is None
    with pytest.raises(RuntimeError, match="start_game"):
        player.act(engine, fixture_torch_observation())


def test_belief_checkpoint_loader_requires_serialized_binding(tmp_path) -> None:
    agent = _belief_agent()
    checkpoint_path = tmp_path / "unbound-belief.pt"
    torch.save(
        {
            "hypers": {
                "observation_hypers": (
                    agent.observation_space.encoder.hypers.model_dump()
                ),
                "agent_hypers": agent.hypers.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(BeliefError, match="missing its schema binding"):
        load_checkpoint_agent(str(checkpoint_path))


def test_checkpoint_loader_rejects_removed_agent_fields(tmp_path) -> None:
    agent = Agent(ObservationSpace(), AgentHypers(hidden_dim=8, num_attention_heads=2))
    checkpoint_path = tmp_path / "removed-agent-field.pt"
    torch.save(
        {
            "hypers": {
                "observation_hypers": (
                    agent.observation_space.encoder.hypers.model_dump()
                ),
                "agent_hypers": {
                    **agent.hypers.model_dump(),
                    "max_conditions": 5,
                },
            },
            "model_state_dict": agent.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="max_conditions"):
        load_checkpoint_agent(str(checkpoint_path))
