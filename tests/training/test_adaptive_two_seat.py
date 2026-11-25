"""Native two-seat curriculum credit, post-update reassignment and exact recovery."""

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.failures import CaptureScope, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    RewardConfig,
)
from manabot.infra.run_settings import RunSettings
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.rollout.adaptive import (
    AdaptiveConfig,
    AdaptiveSampler,
    ErrorChunk,
    GameResult,
)
from manabot.training.rollout.collection import (
    Collector,
    close_collector,
    collect,
    commit_adaptive,
    open_collector,
)
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    settle,
)
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from managym.recording import read_records
from managym.statistics import read_statistics


def _config(signal: Literal["U", "M", "E", "ME"] = "ME") -> AdaptiveConfig:
    decks = ({"Mountain": 8}, {"Forest": 8})
    return AdaptiveConfig(
        signal=signal,
        seed=17501,
        cells=tuple(
            MatchConfig(hero_deck=first, villain_deck=second)
            for first in decks
            for second in decks
        ),
    )


def _collector(config: AdaptiveConfig, agent: ActorCritic) -> Collector:
    return open_collector(
        agent.observation_space,
        Match(config.cells[0]),
        Reward(RewardConfig()),
        num_envs=2,
        seed=197,
        opponent_mode="self",
        learning_seats="both",
        adaptive=config,
        numerics=TorchCollectionSession(device="cpu", seed=197, precision="float32"),
    )


@pytest.mark.parametrize("gradient", ["ppo", "damped_policy"])
def test_both_seat_targets_preserve_physical_games(
    gradient: Literal["ppo", "damped_policy"],
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17501)
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    config = _config()
    collector = _collector(config, agent)
    batch = collect(collector, agent, 8)
    assert batch.assignments is not None
    labels = batch.assignments.reshape(-1, 4)
    assert set(labels[:, 1]) == {0, 1}
    # Per-game physical identities cluster the two separately credited seats.
    games = {(int(row[2]), int(row[3])) for row in labels}
    assert len(games) == collector.stats.games
    for stream, ordinal in games:
        rows = labels[(labels[:, 2] == stream) & (labels[:, 3] == ordinal)]
        assert set(rows[:, 1]) == {0, 1}
    with pytest.raises(ValueError, match="commit adaptive targets"):
        collect(collector, agent, 8)
    with pytest.raises(ValueError, match="actual target"):
        commit_adaptive(collector, batch)
    learning = (
        DampedPolicyLearning(gradient="damped_policy", advantage_quantile=0.75)
        if gradient == "damped_policy"
        else Learning()
    )
    expected = np.zeros_like(batch.values)
    gamma = learning.gamma if isinstance(learning, Learning) else 1.0
    for index in reversed(range(len(expected))):
        if batch.dones[index, 0]:
            expected[index, 0] = batch.rewards[index, 0]
        else:
            expected[index, 0] = batch.rewards[index, 0] + gamma * (
                (1 - learning.value_lambda) * batch.values[index + 1, 0]
                + learning.value_lambda * expected[index + 1, 0]
            )
    expected -= batch.values
    learner = Learner(agent, "cpu", minibatch_rows=2)
    diagnostic = (
        update_damped_iteration(torch_backend, learner, batch, learning, 1)
        if isinstance(learning, DampedPolicyLearning)
        else update_iteration(
            torch_backend, learner, batch, learning, 0.0, np.random.default_rng(17)
        )
    )
    assert batch.target_residuals is not None
    assert batch.target_residuals.shape == batch.values.shape
    np.testing.assert_allclose(batch.target_residuals, expected, atol=1e-6)
    scored = commit_adaptive(collector, batch)
    assert scored is not None
    assert scored["residual_count"] == batch.actions.size
    assert scored["residual_total"] == pytest.approx(
        float(np.abs(batch.target_residuals).sum())
    )
    assert scored["completed_games"] == len(games)
    assert scored["completed_roles"] == 2 * len(games)
    assert diagnostic["retained"] < batch.actions.size
    assert collector._sampler is not None
    assert collector._sampler.state.generation == 1


def test_mirror_roles_share_one_supporting_game_and_duplicates_are_rejected() -> None:
    sampler = AdaptiveSampler(_config("E"))
    results = [GameResult(0, seat, (0, 0), float(seat == 0)) for seat in (0, 1)]
    sampler.commit(
        results,
        [
            ErrorChunk(0, 21.0, 1, frozenset({(0, 0)})),
            ErrorChunk(1, 0.0, 1, frozenset()),
        ],
    )
    # Pool prior median=10.5; one physical completed game gives weight 1/21.
    assert sampler.state.error_scores[0] == pytest.approx(11.0)
    state = sampler.snapshot()
    with pytest.raises(ValueError, match="duplicate completed adaptive role"):
        sampler.commit(results, [])
    assert sampler.snapshot() == state


@pytest.mark.parametrize("adaptive", [False, True])
def test_refill_preserves_game_records_and_statistics(
    adaptive: bool,
    tmp_path: Path,
) -> None:
    config = _config("U")
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    scope = CaptureScope(
        tmp_path / "failures", settings=RunSettings(record_games=1, statistics=True)
    )
    with (
        capture_scope(scope),
        patch.object(
            torch,
            "multinomial",
            side_effect=lambda probs, *args, **kwargs: torch.zeros(
                (len(probs), 1), dtype=torch.long
            ),
        ),
    ):
        collector = open_collector(
            space,
            Match(config.cells[0]),
            Reward(RewardConfig()),
            num_envs=2,
            seed=197,
            opponent_mode="self",
            games_per_stream=2,
            adaptive=config if adaptive else None,
            numerics=TorchCollectionSession(device="cpu", seed=197),
        )
        try:
            batch = collect(collector, agent, 1)
            assert collector.stats.games == 4 and batch.actions.size == 20
            assert collector._env.current_agent_indices() == [-1, -1]
            assert collector._env.current_game_seeds() == [199, 200]
            assert collector._env.next_fresh_seeds() == [201, 202]
            if adaptive:
                batch.target_residuals = np.zeros_like(batch.values)
                diagnostics = commit_adaptive(collector, batch)
                assert diagnostics is not None and diagnostics["starts"] == 6
                assert collector._env.current_game_seeds() == [201, 202]
        finally:
            close_collector(collector)
    expected = [(0, 197), (0, 199), (1, 198), (1, 200)]
    games = [
        game
        for record in read_records(tmp_path / "games")
        for game in record.games
        if game.actions
    ]
    assert sorted((game.stream, game.seed) for game in games) == expected
    for game in games:
        assert game.actions == [0] * 5
        assert game.end is not None and (game.end.status, game.end.winner) == (
            "complete",
            0,
        )
    stats = [row for row in read_statistics(tmp_path / "statistics") if row.terminal]
    assert sorted((row.stream, row.seed) for row in stats) == expected
    for row in stats:
        assert row.winner == 0
        # Eight-land decks keep seven, draw one, and play every offered land.
        assert sorted(
            (
                card.seat,
                card.opening_copies,
                card.drawn_copies,
                card.opportunities,
                card.selections,
            )
            for card in row.cards
        ) == [(0, 7, 1, 2, 2), (1, 7, 1, 1, 1)]


def test_adaptive_action_limit_reassigns_live_slot() -> None:
    config = _config()
    error_policy = GameErrorPolicy(max_game_actions=1, window=2, warmup=2)
    attempts: list[GameAttempt] = []

    def record(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
        attempts.append(attempt)
        return settle(state, attempt, error_policy)

    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    collector = open_collector(
        space,
        Match(config.cells[0]),
        Reward(RewardConfig()),
        num_envs=1,
        seed=197,
        opponent_mode="self",
        adaptive=config,
        game_errors=error_policy,
        record_attempt=record,
        numerics=TorchCollectionSession(device="cpu", seed=197),
    )
    try:
        with pytest.raises(GameErrorLimit):
            collect(collector, agent, 1)
        assert [(a.seed, a.outcome, a.discarded_samples) for a in attempts] == [
            (197, "action_limit", 1),
            (198, "action_limit", 1),
        ]
        assert collector.stats.games == collector.stats.learner_transitions == 0
        assert collector._env.next_fresh_seeds() == [199]
    finally:
        close_collector(collector)
