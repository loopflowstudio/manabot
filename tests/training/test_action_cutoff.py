"""The explicit policy prefix survives collection and evaluation encodings."""

import json
from pathlib import Path

from managym.decision import Command, DecisionFrame
import numpy as np
import pytest
import torch

from manabot.env.env import Env
from manabot.env.failures import CaptureScope, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.vector_env import VectorEnv
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.sim.heuristic import HeuristicPlayer
from manabot.training.rollout.collection import SeatRoutedCollector


def test_heuristic_scores_only_retained_legal_offers(tmp_path: Path) -> None:
    deck = {"Mountain": 10, "Gray Ogre": 64, "Hill Giant": 1, "Shivan Dragon": 1}
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128, max_actions=64, action_overflow="first-legal"
        )
    )
    with capture_scope(CaptureScope(tmp_path)):
        env = Env(
            Match(MatchConfig(hero_deck=deck, villain_deck=deck)),
            space,
            Reward(RewardConfig()),
            seed=213,
        )
        try:
            env.reset()
            actor = env.last_raw_obs.agent.player_index
            engine = env._engine
            engine.scenario_clear_hand(actor)
            engine.scenario_force_card_in_hand(actor, "Hill Giant")
            for _ in range(64):
                engine.scenario_force_card_in_hand(actor, "Gray Ogre")
            engine.scenario_force_card_in_hand(actor, "Shivan Dragon")
            for _ in range(6):
                engine.scenario_force_battlefield(actor, "Mountain")
            raw = env._last_obs = engine.scenario_refresh()
            obs = space.encoder.encode(raw)
            assert len(raw.action_space.actions) == 67
            assert obs["actions_valid"].sum() == 64
            frame = json.loads(engine.semantic_decision_frame_json())
            before = engine.state_digest()
            unrestricted = HeuristicPlayer().act(env, obs)
            retained = HeuristicPlayer(max_actions=64).act(env, obs)
            assert unrestricted == 65 and retained == 0
            cards = {card.id: card.name for card in raw.agent_cards}
            assert (
                cards[raw.action_space.actions[unrestricted].focus[0]]
                == "Shivan Dragon"
            )
            assert cards[raw.action_space.actions[retained].focus[0]] == "Hill Giant"
            assert engine.state_digest() == before
            assert json.loads(engine.semantic_decision_frame_json()) == frame
            assert len(env.last_raw_obs.action_space.actions) == 67
            # Execute the exact retained offer, without clamping or remapping.
            decision = DecisionFrame.from_json(engine.semantic_decision_frame_json())
            command = Command(
                "matched-prefix",
                decision.revision,
                int(decision.offers[retained]["id"]),
            )
            env.step_semantic(command)
            assert engine.state_digest() != before
        finally:
            env.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["counts_by_kind"] == {
        "Priority": {"decisions": 1, "overflows": 1, "dropped_actions": 3}
    }
    assert report["samples"][0]["game"]["actor"] == actor


@pytest.mark.parametrize("workers", [1, 2])
def test_collector_uses_configured_prefix_and_one_masked_distribution(
    workers: int, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, attention_on=False))
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with capture_scope(CaptureScope(tmp_path)):
        collector = SeatRoutedCollector(
            space,
            match,
            Reward(RewardConfig()),
            num_envs=2,
            seed=213,
            workers=workers,
            opponent_mode="self",
            learning_seats="both",
        )
        try:
            batch = collector.collect(agent, 2)
        finally:
            collector.close()
    valid = batch.obs["actions_valid"].astype(bool)
    assert batch.probabilities.shape == valid.shape
    assert np.all(batch.probabilities[~valid] == 0)
    assert np.all(batch.probabilities[valid] > 0)
    np.testing.assert_allclose(batch.probabilities.sum(-1), 1, atol=1e-6)
    assert np.all(np.take_along_axis(valid, batch.actions[..., None], -1))
    reports = [
        json.loads(path.read_bytes()) for path in tmp_path.glob("action-cutoff-*.json")
    ]
    assert len(reports) == workers
    assert (
        sum(
            sum(counts["overflows"] for counts in report["counts_by_kind"].values())
            for report in reports
        )
        > 0
    )


def test_scalar_evaluation_cutoff_is_diagnostic_not_game_truncation(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with capture_scope(CaptureScope(tmp_path)):
        env = Env(match, space, Reward(RewardConfig()), seed=213)
        try:
            obs, _ = env.reset()
            assert obs["actions_valid"].sum() == 2
            assert len(env.last_raw_obs.action_space.actions) > 2
            actor = env.last_raw_obs.agent.player_index
            _, _, _, truncated, info = env.step(0)
            assert not truncated and not info["action_space_truncated"]
        finally:
            env.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["samples"][0]["game"]["seed"] == 213
    assert report["samples"][0]["retained_count"] == 2
    sample = report["samples"][0]
    assert sample["decision_kind"] == "Priority"
    assert sample["game"]["actor"] == actor
    assert json.loads(sample["game"]["frame"])["actor"] == actor
    assert sample["game"]["actions"] == []


def test_vector_reseed_retains_consumed_decision_counts(tmp_path: Path) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=64, action_overflow="first-legal")
    )
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with capture_scope(CaptureScope(tmp_path)):
        env = VectorEnv(1, match, space, Reward(RewardConfig()), "cpu", seed=213)
        try:
            for seed in (213, 214):
                env.reset(seed=seed)
                env.step(torch.tensor([0]))
        finally:
            env.close()
    reports = [
        json.loads(path.read_bytes()) for path in tmp_path.glob("action-cutoff-*.json")
    ]
    assert len(reports) == 2
    for report in reports:
        assert report["counts_by_kind"] == {
            "Priority": {"decisions": 1, "overflows": 0, "dropped_actions": 0}
        }
