"""Ordinary checkpoint admission binds corrected rules and complete setup."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from academy.villain import CheckpointVillain
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import (
    AUTHORED_PACK_BINDING,
    _setup_world,
    checkpoint_world,
    validate_agent_setup,
    validate_checkpoint_world,
)
import managym


def selected_match():
    return Match(
        MatchConfig.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    )


def checkpoint():
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )
    return {
        "model_state_dict": agent.state_dict(),
        "bc": {"value_semantic": "signed_outcome"},
        "hypers": {
            "agent_hypers": agent.config.model_dump(),
            "observation_hypers": space.encoder.config.model_dump(),
        },
        "world_binding": checkpoint_world(selected_match().to_rust(), space),
    }


def test_ordinary_loader_and_play_accept_full_setup_in_either_seat(tmp_path):
    path = tmp_path / "policy.pt"
    payload = checkpoint()
    torch.save(payload, path)
    agent, space = load_checkpoint(str(path))
    assert agent.world_binding == payload["world_binding"]
    for match in (selected_match(), selected_match().swapped()):
        policy = CheckpointVillain(str(path), player_configs=match.to_rust())
        assert policy.agent.world_binding == agent.world_binding
    assert space.shapes["actions"][0] == 64


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "world",
        "rules",
        "schema",
        "input_version",
        "action_width",
        "pack",
        "sideboard",
    ],
)
def test_ordinary_loader_rejects_incompatible_binding_before_weights(
    tmp_path: Path, change: str
) -> None:
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
        elif change == "input_version":
            binding["policy_input_version"] = 1
        elif change == "action_width":
            binding["input_schema"]["actions"]["shape"][1] = 17
        elif change == "pack":
            binding["content_manifest"]["content_digest"] = "old"
        else:
            binding["setups"][0]["sideboard"] = {}
    payload["model_state_dict"] = {}  # Admission must precede weight loading.
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="binding"):
        load_checkpoint(str(path))


def test_same_shapes_do_not_admit_another_matchup():
    payload = checkpoint()
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world(payload, ObservationSpace(), Match().to_rust())
    changed = deepcopy(selected_match())
    changed.hero_sideboard.clear()
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world(payload, ObservationSpace(), changed.to_rust())


@pytest.mark.parametrize("consumer", ["policy", "puct", "rollout", "value", "play"])
def test_loaded_consumers_reject_another_actual_setup(tmp_path, consumer):
    from manabot.env.env import Env
    from manabot.env.match import Reward
    from manabot.sim.rollout import BatchedSampler, PolicyRolloutMCPlayer
    from manabot.sim.search.flat_mc import AgentMatchupPlayer
    from manabot.sim.search.puct import AgentLeafEvaluator, DeterminizedPuctPlayer
    from manabot.sim.search.value import ValueScorer, VGreedyPlayer

    path = tmp_path / "policy.pt"
    torch.save(checkpoint(), path)
    if consumer == "play":
        with pytest.raises(ValueError, match="setup differs"):
            CheckpointVillain(str(path), player_configs=Match().to_rust())
        return
    agent, space = load_checkpoint(str(path))
    players = {
        "policy": lambda: AgentMatchupPlayer(agent),
        "puct": lambda: DeterminizedPuctPlayer(
            simulations=1, worlds=1, evaluator=AgentLeafEvaluator(agent, space)
        ),
        "rollout": lambda: PolicyRolloutMCPlayer(1, BatchedSampler(agent)),
        "value": lambda: VGreedyPlayer(ValueScorer(agent)),
    }
    env = Env(Match(), space, Reward(RewardConfig()))
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="setup differs"):
        players[consumer]().act(env, obs)


@pytest.mark.parametrize("deck", ["ur_lessons", "gw_allies"])
def test_mirror_checkpoint_binds_whole_authored_pack(tmp_path: Path, deck: str) -> None:
    space = ObservationSpace()
    match = Match(MatchConfig.authored("ur-lessons-vs-gw-allies", deck, deck))
    payload = checkpoint()
    payload["world_binding"] = checkpoint_world(match.to_rust(), space)
    path = tmp_path / "mirror.pt"
    torch.save(payload, path)
    agent, _ = load_checkpoint(str(path))
    assert agent.world_binding == payload["world_binding"]
    assert agent.world_binding["setup_binding"] == AUTHORED_PACK_BINDING
    validate_checkpoint_world(payload, space, match.to_rust())
    validate_checkpoint_world(payload, space, selected_match().to_rust())


COMPLEXITY = "mtg122-complexity-v1"


def _complexity(left: str, right: str) -> list[managym.PlayerConfig]:
    return Match(MatchConfig.authored(COMPLEXITY, left, right)).to_rust()


def _historical_pair(configs: list[managym.PlayerConfig], space) -> dict:
    """The unversioned pair binding every authored-pack checkpoint carried before."""
    return _setup_world(configs, space, roster=False)


def test_pair_trained_pack_checkpoint_runs_any_authored_pair() -> None:
    space = ObservationSpace()
    binding = checkpoint_world(_complexity("l1_swift", "l1_heavy"), space)
    assert binding["setup_binding"] == AUTHORED_PACK_BINDING
    assert len(binding["setups"]) == len(managym.authored_deck_keys(COMPLEXITY))
    agent = SimpleNamespace(world_binding=binding)
    for left, right in [("gw_allies", "l2_heavy"), ("ur_lessons", "ur_lessons")]:
        configs = _complexity(left, right)
        validate_agent_setup(agent, configs)
        assert validate_checkpoint_world({"world_binding": binding}, space, configs)
    assert validate_checkpoint_world({"world_binding": binding}, space) == binding

    unauthored = _complexity("gw_allies", "l2_heavy")
    unauthored[1] = Match().to_rust()[0]
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(agent, unauthored)
    # The same lists under another pack's selection are another world.
    with pytest.raises(ValueError, match="content pack"):
        validate_agent_setup(agent, selected_match().to_rust())
    with pytest.raises(ValueError, match="world/setup/input|invalid setup"):
        validate_checkpoint_world(
            {"world_binding": binding}, space, selected_match().to_rust()
        )
    unmarked = deepcopy(binding)
    del unmarked["setup_binding"]
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world({"world_binding": unmarked}, space)


def test_historical_pair_binding_keeps_its_pair() -> None:
    space = ObservationSpace()
    trained = _complexity("l1_swift", "l1_heavy")
    binding = _historical_pair(trained, space)
    assert "setup_binding" not in binding
    assert validate_checkpoint_world({"world_binding": binding}, space) == binding
    validate_checkpoint_world({"world_binding": binding}, space, trained)
    agent = SimpleNamespace(world_binding=binding)
    validate_agent_setup(agent, list(reversed(trained)))
    other = _complexity("gw_allies", "l2_heavy")
    with pytest.raises(ValueError, match="setup differs"):
        validate_agent_setup(agent, other)
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world({"world_binding": binding}, space, other)
    mirror = _complexity("l1_swift", "l1_swift")
    with pytest.raises(ValueError, match="setup differs"):
        validate_agent_setup(agent, mirror)
    validate_agent_setup(agent, mirror, allow_deck_repetition=True)
