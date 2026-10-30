"""Frozen value inference, leaf perspective and value-guided player legality."""

from pathlib import Path

import numpy as np
import pytest

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.rollout import OBS_KEYS, _allocate_buffers
from manabot.sim.search.flat_mc import make_player, spec_name
from manabot.sim.search.value import (
    ValueScorer,
    ValueSearchPlayer,
    VGreedyPlayer,
)
from manabot.verify.games import INTERACTIVE_DECK
import managym


def make_env(seed: int = 3) -> tuple[Env, dict]:
    obs_space = ObservationSpace()
    match = Match(
        MatchConfig(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardConfig()), seed=seed, auto_reset=False)
    obs, _ = env.reset(seed=seed)
    return env, obs


class TestValueScorer:
    def test_scores_are_probs(self):
        env, obs = make_env()
        scorer = ValueScorer(ActorCritic(ObservationSpace(), ModelConfig()))
        batch = {
            key: np.asarray(obs[key])[None].astype(
                np.int32 if key == "action_focus" else np.float32
            )
            for key in OBS_KEYS
        }
        probs = scorer.score(batch, np.array([0]))
        assert probs.shape == (1,)
        assert 0.0 <= probs[0] <= 1.0
        assert scorer.forward_calls == 1 and scorer.obs_scored == 1


class TestPoolPerspective:
    def test_acting_players_and_roots_match_layout(self):
        env, _ = make_env(seed=5)
        num_actions = env._engine.action_count()
        pool = env._engine.rollout_pool(3, 2, 42, 2000)
        assert pool.hero_index in (0, 1)
        roots = pool.root_actions()
        assert len(roots) == pool.num_slots == 3 * 2 * num_actions
        # (world, action, rollout) lexicographic layout
        expect = [a for _ in range(3) for a in range(num_actions) for _ in range(2)]
        assert roots == expect
        buffers = _allocate_buffers(ObservationSpace(), pool.num_slots)
        pool.set_buffers(buffers, pool.num_slots)
        active = pool.encode_active()
        acting = pool.acting_players()
        for slot, player in enumerate(acting):
            if slot in active:
                assert player in (0, 1)
            else:
                assert player == -1


class TestPlayers:
    @pytest.fixture
    def value_checkpoint(
        self, tmp_path: Path, interactive_player_configs: list[managym.PlayerConfig]
    ) -> str:
        obs_space = ObservationSpace()
        agent = ActorCritic(obs_space, ModelConfig())
        path = tmp_path / "value.pt"
        save_checkpoint(
            agent,
            obs_space,
            path,
            player_configs=interactive_player_configs,
            extra={"value": {}},
        )
        return str(path)

    def test_vgreedy_plays_valid_actions(self, value_checkpoint):
        env, obs = make_env(seed=7)
        player, _ = make_player(
            {
                "kind": "value_greedy",
                "checkpoint": value_checkpoint,
            },
            seed=1,
        )
        assert isinstance(player, VGreedyPlayer)
        for _ in range(20):
            valid = np.flatnonzero(obs["actions_valid"] > 0)
            action = player.act(env, obs)
            assert action in valid
            assert len(player.last_scores) == len(valid)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert player.stats.decisions > 0

    def test_value_search_scores_stay_in_range(self, value_checkpoint):
        env, obs = make_env(seed=9)
        player, _ = make_player(
            {
                "kind": "value_search",
                "sims": 8,
                "depth": 2,
                "checkpoint": value_checkpoint,
            },
            seed=2,
        )
        assert isinstance(player, ValueSearchPlayer)
        for _ in range(10):
            valid = np.flatnonzero(obs["actions_valid"] > 0)
            action = player.act(env, obs)
            assert action in valid
            scores = player.last_scores
            assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert player.stats.net_forwards > 0

    def test_spec_names(self):
        assert spec_name({"kind": "value_greedy"}) == "vgreedy"
        assert (
            spec_name({"kind": "value_search", "sims": 64, "depth": 0})
            == "vsearch-64-d0"
        )
