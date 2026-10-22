"""Real frozen-opponent PPO on a deliberately passive creature-only toy world.

This control tests attack sensitivity, not selected-world strength: the toy
checkpoint always passes priority, so a learner that deploys creatures can
win before decking. No private features or shaped rewards enter learning.
"""

from pathlib import Path
import time

import numpy as np
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.env.observation import ActionEnum
from manabot.infra.experiment import seed_random
from manabot.infra.hypers import (
    AgentSpec,
    MatchHypers,
    RewardHypers,
)
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.sim.rollout import BatchedSampler, run_vector_games
from manabot.training.learner import Learner
from manabot.training.models import Learning, Schedule
from manabot.training.objectives import update_iteration
from manabot.verify.learnability.schema import PPOResult

_DECK = {"Mountain": 20, "Gray Ogre": 20}


def _pass_biased_agent(space: ObservationSpace, bias: float) -> Agent:
    """Ordinary serializable Agent whose initial logit is bias for passing.

    One active path carries the public action-type bit through the usual
    network. PPO can change every parameter; this is a deliberately small
    starting policy, not an alternate learner or a reward override.
    """
    agent = Agent(
        space, AgentSpec(hidden_dim=8, num_attention_heads=2, attention_on=False)
    )
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.action_embedding.projection[0].weight[
            0, ActionEnum.PRIORITY_PASS_PRIORITY
        ] = 1
        agent.action_embedding.projection[2].weight[0, 0] = 1
        agent.action_layer.projection[0].weight[0, 0] = 1
        agent.action_layer.projection[2].weight[0, 0] = 1
        agent.policy_head[0].weight[0, 0] = 1
        agent.policy_head[2].weight[0, 0] = bias
    return agent


def _score(agent: Agent, opponent: Agent, seed: int) -> float:
    torch.manual_seed(seed)
    games, _ = run_vector_games(
        BatchedSampler(agent),
        BatchedSampler(opponent),
        num_games=64,
        num_streams=8,
        seed=seed,
        deck=_DECK,
        max_env_steps=100_000,
    )
    assert len(games) == 64
    assert all(game.winner is not None for game in games)
    return sum(game.hero_won for game in games) / len(games)


def run_positive_control(tmp_path: Path, *, deadline: float) -> PPOResult:
    start = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        space = ObservationSpace()
        match = Match(MatchHypers(hero_deck=_DECK, villain_deck=_DECK))
        path = tmp_path / "pass-policy.pt"
        save_bc_checkpoint(
            _pass_biased_agent(space, 20), space, path, player_configs=match.to_rust()
        )
        opponent, _ = load_checkpoint_agent(str(path))
        torch.manual_seed(197)
        agent = _pass_biased_agent(space, 4)
        before = _score(agent, opponent, 100_197)
        collector = SeatRoutedCollector(
            space,
            match,
            Reward(RewardHypers()),
            num_envs=8,
            seed=197,
            opponent_mode="frozen",
            opponent_agent=opponent,
        )
        try:
            # The baseline score consumes policy RNG. Keep the historical
            # training boundary independent of evaluation and tracking setup.
            seed_random(197)
            learner = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
            learning = Learning(
                policy_lambda=1,
                retained_fraction=1,
                min_advantage=0,
                learning_rate=Schedule(initial=0.01),
                tau=Schedule(initial=0),
                collection_kl=0,
                epochs=4,
                minibatches=4,
                clip=0.2,
            )
            rng = np.random.default_rng(197)
            train_started = time.perf_counter()
            for _ in range(64):
                batch = collector.collect(agent, 128, deadline_monotonic=deadline)
                update_iteration(learner, batch, learning, 0, rng)
            training_seconds = time.perf_counter() - train_started
            checkpoint = tmp_path / "ppo-policy.pt"
            save_bc_checkpoint(agent, space, checkpoint, player_configs=match.to_rust())
            restored, _ = load_checkpoint_agent(str(checkpoint))
            after = _score(restored, opponent, 200_197)
            no_update = _score(_pass_biased_agent(space, 4), opponent, 200_197)
            return PPOResult(
                before=before,
                after=after,
                no_update=no_update,
                seconds=time.perf_counter() - start,
                transitions=collector.stats.learner_transitions,
                training_seconds=training_seconds,
                games=collector.stats.games,
                checkpoint=checkpoint.name,
                passed=after >= max(before, no_update) + 0.2 and after >= 0.8,
            )
        finally:
            collector.close()
    finally:
        torch.set_num_threads(previous_threads)
