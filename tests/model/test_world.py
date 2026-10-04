"""Ordinary checkpoint admission binds corrected rules and complete setup."""

from copy import deepcopy

import pytest
import torch

from etude.villain import CheckpointVillain
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentHypers, MatchHypers
from manabot.model import Agent
from manabot.model.world import checkpoint_world, validate_checkpoint_world
from manabot.sim.flat_mc import load_checkpoint_agent


def selected_match():
    return Match(MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies"))


def checkpoint():
    space = ObservationSpace()
    agent = Agent(space, AgentHypers(hidden_dim=8, num_attention_heads=2))
    return {
        "model_state_dict": agent.state_dict(),
        "hypers": {
            "agent_hypers": agent.hypers.model_dump(),
            "observation_hypers": space.encoder.hypers.model_dump(),
        },
        "world_binding": checkpoint_world(selected_match().to_rust(), space),
    }


def test_ordinary_loader_and_play_accept_full_setup_in_either_seat(tmp_path):
    path = tmp_path / "policy.pt"
    payload = checkpoint()
    torch.save(payload, path)
    agent, space = load_checkpoint_agent(str(path))
    assert agent.world_binding == payload["world_binding"]
    for match in (selected_match(), selected_match().swapped()):
        policy = CheckpointVillain(str(path), player_configs=match.to_rust())
        assert policy.agent.world_binding == agent.world_binding
    assert space.shapes["actions"][0] == 64


@pytest.mark.parametrize("change", ["missing", "world", "rules", "schema", "pack", "sideboard"])
def test_ordinary_loader_rejects_incompatible_binding_before_weights(tmp_path, change):
    payload = checkpoint()
    if change == "missing":
        del payload["world_binding"]
    else:
        binding = payload["world_binding"]
        if change == "world":
            binding["world"] = "w3"
        elif change == "rules":
            binding["rules"] = "discard-only"
        elif change == "schema":
            binding["input_schema"]["actions"]["shape"][0] = 32
        elif change == "pack":
            binding["content_manifest"]["content_digest"] = "old"
        else:
            binding["setups"][0]["sideboard"] = {}
    payload["model_state_dict"] = {}  # Admission must precede weight loading.
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="binding"):
        load_checkpoint_agent(str(path))


def test_same_shapes_do_not_admit_another_matchup():
    payload = checkpoint()
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world(payload, ObservationSpace(), Match().to_rust())
    changed = deepcopy(selected_match())
    changed.hero_sideboard.clear()
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world(payload, ObservationSpace(), changed.to_rust())
