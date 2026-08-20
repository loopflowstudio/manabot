"""Focused checks for the ordinary ManabotPlayer runtime boundary."""

import json

from manabot.belief import ManabotPlayer
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from tests.belief.support import fixture_torch_observation


class NativeContractEngine:
    """Small current-ABI engine double; the CLI demo remains the native proof."""

    def __init__(self) -> None:
        self.revision = 17
        self.viewer_hash = "runtime-viewer-17"
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
            "definitions": [
                {"card_def_id": 0, "registry_name": "Counterspell"},
                {"card_def_id": 1, "registry_name": "Lightning Bolt"},
                {"card_def_id": 2, "registry_name": "Mountain"},
            ],
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

def test_manabot_player_generates_belief_before_ordinary_action() -> None:
    engine = NativeContractEngine()
    agent = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=3,
            belief_card_vocab_size=3,
        ),
    )
    player = ManabotPlayer(agent)

    player.start_game(engine, 0)
    action = player.act(engine, fixture_torch_observation())

    assert action in {0, 1, 2}
    assert player.last_step is not None
    assert player.last_step.belief_update.belief.normalization_error < 1e-12
    command = player.command_for_action(engine, action, command_id="runtime-test")
    assert command.command_id == "runtime-test"
    assert command.offer_id == action
