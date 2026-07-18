"""Tests for the determinized PUCT Teacher-1 reference."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import (
    ROOT_VALUE_KEY,
    VISIT_COUNT_KEY,
    generate_selfplay_shard,
    load_shards,
    save_bc_checkpoint,
)
from manabot.sim.flat_mc import make_player
from manabot.sim.mcts import (
    AgentLeafEvaluator,
    DeterminizedPuctPlayer,
    determinized_puct,
)
from manabot.verify.util import INTERACTIVE_DECK
import managym


def _fresh_engine(seed: int = 0) -> managym.Env:
    env, _ = _fresh_engine_and_observation(seed)
    return env


def _fresh_engine_and_observation(
    seed: int = 0,
) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=seed, skip_trivial=True)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("hero", dict(INTERACTIVE_DECK)),
            managym.PlayerConfig("villain", dict(INTERACTIVE_DECK)),
        ]
    )
    return env, observation


def test_puct_is_seeded_budgeted_and_does_not_mutate_root() -> None:
    first_root = _fresh_engine(seed=31)
    second_root = _fresh_engine(seed=31)
    action_count = first_root.action_count()
    acting = first_root.current_agent_index()

    first = determinized_puct(
        first_root, simulations=12, worlds=3, seed=99, max_steps=2000
    )
    second = determinized_puct(
        second_root, simulations=12, worlds=3, seed=99, max_steps=2000
    )

    assert first.simulations == 12
    assert first.worlds == 3
    assert int(first.visit_counts.sum()) == 12
    assert len(first.visit_counts) == len(first.q_values) == action_count
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert np.array_equal(first.q_values, second.q_values)
    assert first.root_value == second.root_value
    assert 0.0 <= first.root_value <= 1.0
    assert np.all((first.q_values >= 0.0) & (first.q_values <= 1.0))
    assert first.tree_nodes > first.worlds
    assert first.max_depth >= 1
    assert first.world_visit_counts.shape == (3, action_count)
    assert first.world_q_values.shape == (3, action_count)
    assert first.world_root_values.shape == (3,)
    assert np.array_equal(first.world_visit_counts.sum(axis=0), first.visit_counts)
    assert np.array_equal(first.world_visit_counts, second.world_visit_counts)
    assert np.array_equal(first.world_q_values, second.world_q_values)
    assert np.array_equal(first.world_root_values, second.world_root_values)
    assert first.root_value == pytest.approx(float(first.world_root_values.mean()))

    assert first_root.action_count() == action_count
    assert first_root.current_agent_index() == acting
    assert not first_root.is_game_over()


def test_puct_rejects_more_worlds_than_simulations() -> None:
    with pytest.raises(ValueError, match="worlds"):
        determinized_puct(_fresh_engine(), simulations=2, worlds=3, seed=0)
    with pytest.raises(ValueError, match="worlds"):
        DeterminizedPuctPlayer(2, worlds=3)


def test_agent_leaf_evaluator_uses_masked_priors_and_root_perspective() -> None:
    root, observation = _fresh_engine_and_observation(seed=35)
    observation_space = ObservationSpace()
    agent = Agent(observation_space, AgentHypers())
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.value_head[-1].bias.fill_(torch.logit(torch.tensor(0.8)))
    evaluator = AgentLeafEvaluator(agent, observation_space)
    action_count = root.action_count()
    actor = int(root.current_agent_index())

    priors = evaluator.root_priors(observation, action_count=action_count)
    same_actor = evaluator.evaluate(
        root,
        observation,
        root_player=actor,
        node_player=actor,
        seed=1,
        max_steps=10,
    )
    other_actor = evaluator.evaluate(
        root,
        observation,
        root_player=actor,
        node_player=1 - actor,
        seed=1,
        max_steps=10,
    )

    assert priors.shape == (action_count,)
    assert np.all(priors > 0)
    assert float(priors.sum()) == pytest.approx(1.0)
    assert same_actor.root_value == pytest.approx(0.8)
    assert other_actor.root_value == pytest.approx(0.2)

    result = determinized_puct(
        root,
        simulations=4,
        worlds=2,
        seed=39,
        evaluator=evaluator,
        root_observation=observation,
    )
    assert int(result.visit_counts.sum()) == 4
    assert result.cap_hits == 0


def test_agent_leaf_evaluator_neutral_mode_keeps_forward_and_priors() -> None:
    root, observation = _fresh_engine_and_observation(seed=36)
    observation_space = ObservationSpace()
    agent = Agent(observation_space, AgentHypers())
    learned = AgentLeafEvaluator(agent, observation_space, value_mode="learned")
    neutral = AgentLeafEvaluator(agent, observation_space, value_mode="neutral")
    action_count = int(root.action_count())
    actor = int(root.current_agent_index())

    learned_priors = learned.root_priors(observation, action_count=action_count)
    neutral_priors = neutral.root_priors(observation, action_count=action_count)
    evaluation = neutral.evaluate(
        root,
        observation,
        root_player=actor,
        node_player=actor,
        seed=3,
        max_steps=2000,
    )

    np.testing.assert_allclose(neutral_priors, learned_priors)
    assert evaluation.root_value == 0.5
    assert learned.forward_calls == 1
    assert neutral.forward_calls == 2


def test_agent_puct_player_spec_loads_a_frozen_cpu_checkpoint(
    tmp_path: Path,
) -> None:
    observation_space = ObservationSpace()
    checkpoint = tmp_path / "student.pt"
    save_bc_checkpoint(
        Agent(observation_space, AgentHypers()), observation_space, checkpoint
    )

    player, loaded_space = make_player(
        {
            "kind": "agent_puct",
            "checkpoint": str(checkpoint),
            "sims": 4,
            "worlds": 2,
            "device": "cpu",
        },
        seed=43,
    )

    assert isinstance(player, DeterminizedPuctPlayer)
    assert isinstance(player.evaluator, AgentLeafEvaluator)
    assert loaded_space == observation_space
    with pytest.raises(ValueError, match="CPU-only"):
        make_player(
            {
                "kind": "agent_puct",
                "checkpoint": str(checkpoint),
                "sims": 4,
                "worlds": 2,
                "device": "mps",
            },
            seed=43,
        )


def test_player_publishes_visit_and_value_targets() -> None:
    root = _fresh_engine(seed=37)
    player = DeterminizedPuctPlayer(8, worlds=2, seed=41)

    class _Wrapper:
        _engine = root

    action = player.act(_Wrapper(), {})
    assert 0 <= action < root.action_count()
    assert player.last_visit_counts is not None
    assert player.last_scores is not None
    assert player.last_root_value is not None
    assert int(player.last_visit_counts.sum()) == 8
    assert player.last_visit_counts[action] == player.last_visit_counts.max()
    assert player.stats.simulations == 8
    assert player.stats.decisions == 1
    assert player.stats.tree_nodes > player.worlds
    assert player.stats.worlds_sampled == player.worlds
    assert player.stats.max_depth_sum >= 1
    assert player.stats.max_depth_max >= 1


def test_selfplay_shard_records_real_tree_targets(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00.npz"
    summary = generate_selfplay_shard(
        num_games=1,
        teacher_spec={
            "kind": "determinized_puct",
            "sims": 4,
            "worlds": 1,
            "c_puct": 1.5,
        },
        seed=43,
        out_path=shard,
    )
    dataset = load_shards([shard])

    assert summary["policy_target_kind"] == "visit_distribution"
    assert summary["value_target_kind"] == "root_value"
    assert VISIT_COUNT_KEY in dataset
    assert ROOT_VALUE_KEY in dataset
    assert np.all(dataset[VISIT_COUNT_KEY].sum(axis=1) == 4)
    assert np.all((dataset[ROOT_VALUE_KEY] >= 0) & (dataset[ROOT_VALUE_KEY] <= 1))
    assert np.all(dataset[VISIT_COUNT_KEY][dataset["actions_valid"] <= 0] == 0)
    chosen = dataset[VISIT_COUNT_KEY][
        np.arange(len(dataset["action"])), dataset["action"].astype(np.int64)
    ]
    assert np.all(chosen == dataset[VISIT_COUNT_KEY].max(axis=1))
    assert summary["search"]["tree_nodes"] > summary["search"]["worlds_sampled"]
    assert summary["search"]["max_depth_max"] >= 1
