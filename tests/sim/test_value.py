"""Tests for the value head, V-greedy, and value-at-leaf search (exp-10).

Covers outcome-label construction, value training (loss decreases, policy
head untouched, checkpoint roundtrip), the Spearman helper, hero-perspective
leaf scoring against the RolloutPool bindings, and both players' action
validity.
"""

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.distill import OBS_KEYS
from manabot.sim.flat_mc import load_checkpoint_agent, make_player, spec_name
from manabot.sim.rollout import _allocate_buffers
from manabot.sim.value import (
    ValueScorer,
    VGreedyPlayer,
    ValueSearchPlayer,
    outcome_labels,
    save_value_checkpoint,
    spearman,
    train_value,
)
from manabot.verify.util import INTERACTIVE_DECK


def make_env(seed: int = 3) -> tuple[Env, dict]:
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed, auto_reset=False)
    obs, _ = env.reset(seed=seed)
    return env, obs


def tiny_dataset(num_rows: int = 96, seed: int = 0) -> dict[str, np.ndarray]:
    """Real encoded observations + synthetic winner/seat columns."""

    env, obs = make_env(seed=seed)
    rng = np.random.default_rng(seed)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    while len(rows["actions_valid"]) < num_rows:
        for key in OBS_KEYS:
            rows[key].append(np.asarray(obs[key], dtype=np.float32))
        valid = np.flatnonzero(obs["actions_valid"] > 0)
        obs, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + len(rows["actions_valid"]))
    dataset = {key: np.stack(values) for key, values in rows.items()}
    n = num_rows
    dataset["game_index"] = (np.arange(n) // 12).astype(np.int32)
    dataset["seat"] = (np.arange(n) % 2).astype(np.int8)
    winner_per_game = rng.integers(0, 2, size=n // 12 + 1)
    dataset["winner"] = winner_per_game[dataset["game_index"]].astype(np.int8)
    dataset["winner"][:5] = -1  # a few no-winner rows
    dataset["num_valid"] = np.full(n, 2, dtype=np.int16)
    dataset["action"] = np.zeros(n, dtype=np.int16)
    return dataset


class TestOutcomeLabels:
    def test_labels_follow_winner_and_seat(self):
        dataset = tiny_dataset()
        usable, labels = outcome_labels(dataset)
        assert (dataset["winner"][usable] >= 0).all()
        expect = (
            dataset["winner"][usable].astype(int)
            == dataset["seat"][usable].astype(int)
        ).astype(np.float32)
        assert np.array_equal(labels, expect)

    def test_no_winner_rows_dropped(self):
        dataset = tiny_dataset()
        usable, _ = outcome_labels(dataset)
        assert 0 not in usable[:0].tolist()
        assert len(usable) == int((dataset["winner"] >= 0).sum())


class TestSpearman:
    def test_perfect_and_inverse(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert spearman(x, x) == 1.0
        assert spearman(x, -x) == -1.0

    def test_ties_average(self):
        x = np.array([1.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 2.0, 3.0])
        assert abs(spearman(x, y) - 1.0) < 1e-12

    def test_known_value(self):
        # One transposition among four: rho = 1 - 6*2/(4*15) = 0.8
        assert abs(spearman(np.arange(4.0), np.array([0.0, 2, 1, 3])) - 0.8) < 1e-12


class TestTrainValue:
    def test_loss_decreases_and_policy_head_untouched(self):
        dataset = tiny_dataset()
        init = Agent(ObservationSpace(), AgentHypers())
        init_state = {k: v.clone() for k, v in init.state_dict().items()}
        agent, _, history = train_value(
            dataset, init_state=init_state, epochs=3, lr=1e-3, batch_size=32
        )
        assert history[-1].train_loss < history[0].train_loss
        for name, param in agent.named_parameters():
            if name.startswith("policy_head"):
                assert torch.equal(param.detach().cpu(), init_state[name])

    def test_freeze_encoder_only_moves_value_head(self):
        dataset = tiny_dataset()
        init = Agent(ObservationSpace(), AgentHypers())
        init_state = {k: v.clone() for k, v in init.state_dict().items()}
        agent, _, _ = train_value(
            dataset,
            init_state=init_state,
            freeze_encoder=True,
            epochs=2,
            batch_size=32,
        )
        for name, param in agent.named_parameters():
            same = torch.equal(param.detach().cpu(), init_state[name])
            assert same != name.startswith("value_head"), name

    def test_checkpoint_roundtrip(self, tmp_path):
        dataset = tiny_dataset()
        agent, obs_space, _ = train_value(dataset, epochs=1, batch_size=32)
        path = tmp_path / "value.pt"
        save_value_checkpoint(agent, obs_space, path)
        loaded, _ = load_checkpoint_agent(str(path))
        for (name, a), (_, b) in zip(
            agent.state_dict().items(), loaded.state_dict().items()
        ):
            assert torch.equal(a.cpu(), b.cpu()), name


class TestValueScorer:
    def test_scores_are_probs(self):
        env, obs = make_env()
        scorer = ValueScorer(Agent(ObservationSpace(), AgentHypers()))
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
        expect = [
            a for _ in range(3) for a in range(num_actions) for _ in range(2)
        ]
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
    def _value_checkpoint(self, tmp_path) -> str:
        obs_space = ObservationSpace()
        agent = Agent(obs_space, AgentHypers())
        path = tmp_path / "value.pt"
        save_value_checkpoint(agent, obs_space, path)
        return str(path)

    def test_vgreedy_plays_valid_actions(self, tmp_path):
        env, obs = make_env(seed=7)
        player, _ = make_player(
            {"kind": "value_greedy", "checkpoint": self._value_checkpoint(tmp_path)},
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

    def test_value_search_scores_stay_in_range(self, tmp_path):
        env, obs = make_env(seed=9)
        player, _ = make_player(
            {
                "kind": "value_search",
                "sims": 8,
                "depth": 2,
                "checkpoint": self._value_checkpoint(tmp_path),
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
