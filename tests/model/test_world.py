"""Ordinary checkpoint admission binds corrected rules and complete setup."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from academy.villain import CheckpointVillain
from manabot.env import Match, ObservationSpace
<<<<<<< HEAD
from manabot.infra.hypers import AgentSpec, MatchHypers, RewardHypers
from manabot.model import Agent
from manabot.model.world import checkpoint_world, validate_checkpoint_world
from manabot.sim.flat_mc import load_checkpoint_agent
=======
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import checkpoint_world, validate_checkpoint_world
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


def selected_match():
    return Match(
<<<<<<< HEAD
        MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
=======
        MatchConfig.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    )


def checkpoint():
    space = ObservationSpace()
<<<<<<< HEAD
    agent = Agent(
        space,
        AgentSpec(
=======
    agent = ActorCritic(
        space,
        ModelConfig(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )
    return {
        "model_state_dict": agent.state_dict(),
        "bc": {"value_semantic": "signed_outcome"},
        "hypers": {
<<<<<<< HEAD
            "agent_hypers": agent.hypers.model_dump(),
            "observation_hypers": space.encoder.hypers.model_dump(),
=======
            "agent_hypers": agent.config.model_dump(),
            "observation_hypers": space.encoder.config.model_dump(),
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        },
        "world_binding": checkpoint_world(selected_match().to_rust(), space),
    }


def test_ordinary_loader_and_play_accept_full_setup_in_either_seat(tmp_path):
    path = tmp_path / "policy.pt"
    payload = checkpoint()
    torch.save(payload, path)
<<<<<<< HEAD
    agent, space = load_checkpoint_agent(str(path))
=======
    agent, space = load_checkpoint(str(path))
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
            binding["policy_input_version"] = 2
=======
            binding["policy_input_version"] = 1
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
        load_checkpoint_agent(str(path))
=======
        load_checkpoint(str(path))
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


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
    from manabot.env import Env, Reward
<<<<<<< HEAD
    from manabot.sim.flat_mc import AgentMatchupPlayer
    from manabot.sim.mcts import AgentLeafEvaluator, DeterminizedPuctPlayer
    from manabot.sim.rollout import BatchedSampler, PolicyRolloutMCPlayer
    from manabot.sim.value import ValueScorer, VGreedyPlayer
=======
    from manabot.sim.rollout import BatchedSampler, PolicyRolloutMCPlayer
    from manabot.sim.search.flat_mc import AgentMatchupPlayer
    from manabot.sim.search.puct import AgentLeafEvaluator, DeterminizedPuctPlayer
    from manabot.sim.search.value import ValueScorer, VGreedyPlayer
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

    path = tmp_path / "policy.pt"
    torch.save(checkpoint(), path)
    if consumer == "play":
        with pytest.raises(ValueError, match="setup differs"):
            CheckpointVillain(str(path), player_configs=Match().to_rust())
        return
<<<<<<< HEAD
    agent, space = load_checkpoint_agent(str(path))
=======
    agent, space = load_checkpoint(str(path))
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    players = {
        "policy": lambda: AgentMatchupPlayer(agent),
        "puct": lambda: DeterminizedPuctPlayer(
            simulations=1, worlds=1, evaluator=AgentLeafEvaluator(agent, space)
        ),
        "rollout": lambda: PolicyRolloutMCPlayer(1, BatchedSampler(agent)),
        "value": lambda: VGreedyPlayer(ValueScorer(agent)),
    }
<<<<<<< HEAD
    env = Env(Match(), space, Reward(RewardHypers()))
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="setup differs"):
        players[consumer]().act(env, obs)
=======
    env = Env(Match(), space, Reward(RewardConfig()))
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="setup differs"):
        players[consumer]().act(env, obs)


@pytest.mark.parametrize("deck", ["ur_lessons", "gw_allies"])
def test_mirror_checkpoint_reload_preserves_explicit_native_pack(
    tmp_path: Path, deck: str
) -> None:
    space = ObservationSpace()
    match = Match(MatchConfig.authored("ur-lessons-vs-gw-allies", deck, deck))
    payload = checkpoint()
    payload["world_binding"] = checkpoint_world(match.to_rust(), space)
    path = tmp_path / "mirror.pt"
    torch.save(payload, path)
    agent, _ = load_checkpoint(str(path))
    assert agent.world_binding == payload["world_binding"]
    validate_checkpoint_world(payload, space, match.to_rust())
    with pytest.raises(ValueError, match="world/setup/input"):
        validate_checkpoint_world(payload, space, selected_match().to_rust())
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
