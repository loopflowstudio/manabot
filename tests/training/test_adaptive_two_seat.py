"""Native two-seat curriculum credit, post-update reassignment and exact recovery."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import DampedPolicyLearning, Learning, TrainSelfPlay
from manabot.training.rollout.adaptive import (
    AdaptiveConfig,
    AdaptiveSampler,
    ErrorChunk,
    GameResult,
)
from manabot.training.rollout.collection import SeatRoutedCollector
import managym


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


def _collector(config: AdaptiveConfig, agent: ActorCritic) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        agent.observation_space,
        Match(config.cells[0]),
        Reward(RewardConfig()),
        num_envs=2,
        seed=197,
        opponent_mode="self",
        learning_seats="both",
        adaptive=config,
        recovery_max_game_microsteps=40000,
    )


@pytest.mark.parametrize("gradient", ["ppo", "damped_policy"])
def test_both_seat_targets_and_recovery_preserve_physical_games(
    gradient: Literal["ppo", "damped_policy"],
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17501)
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    config = _config()
    collector = _collector(config, agent)
    batch = collector.collect(agent, 8)
    assert batch.assignments is not None
    labels = batch.assignments.reshape(-1, 4)
    assert set(labels[:, 1]) == {0, 1}
    # Per-game physical identities cluster the two separately credited seats.
    games = {(int(row[2]), int(row[3])) for row in labels}
    assert len(games) == collector.stats.games
    for stream, ordinal in games:
        rows = labels[(labels[:, 2] == stream) & (labels[:, 3] == ordinal)]
        assert set(rows[:, 1]) == {0, 1}
    with pytest.raises(ValueError, match="committed adaptive"):
        collector.snapshot()
    with pytest.raises(ValueError, match="actual target"):
        collector.commit_adaptive(batch)
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
    diagnostic = update_iteration(
        Learner(agent, "cpu", minibatch_rows=2),
        batch,
        learning,
        0.0,
        np.random.default_rng(17),
        iteration=1,
    )
    assert batch.target_residuals is not None
    assert batch.target_residuals.shape == batch.values.shape
    np.testing.assert_allclose(batch.target_residuals, expected, atol=1e-6)
    scored = collector.commit_adaptive(batch)
    assert scored is not None
    assert scored["residual_count"] == batch.actions.size
    assert scored["residual_total"] == pytest.approx(
        float(np.abs(batch.target_residuals).sum())
    )
    assert scored["completed_games"] == len(games)
    assert scored["completed_roles"] == 2 * len(games)
    assert diagnostic["retained"] < batch.actions.size
    snapshot = collector.snapshot()
    assert snapshot.adaptive is not None
    assert snapshot.adaptive.sampler.generation == 1
    restored = _collector(config, agent)
    restored.restore(snapshot, lambda: None)
    left, right = collector.collect(agent, 8), restored.collect(agent, 8)
    np.testing.assert_array_equal(left.assignments, right.assignments)
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.values, right.values)
    for key in left.obs:
        np.testing.assert_array_equal(left.obs[key], right.obs[key])


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


def test_adaptive_recipe_excludes_one_seat_and_static_curriculum() -> None:
    config = _config()
    stage = TrainSelfPlay(id="train", operation="train_self_play", adaptive=config)
    assert stage.learning_seats == "both"
    assert (
        "adaptive"
        not in TrainSelfPlay(id="old", operation="train_self_play").model_dump()
    )
    for options in (
        {"learning_seats": "one"},
        {"matchup_curriculum": "elemental-uniform"},
        {"behavior": "ema-self"},
    ):
        with pytest.raises(ValueError, match="adaptive sampling requires"):
            TrainSelfPlay(
                id="train", operation="train_self_play", adaptive=config, **options
            )


def test_current_elemental_training_exports_scored_admitted_checkpoint(
    tmp_path: Path,
) -> None:
    from experiments.elemental_long import recipe
    from manabot.model.checkpoint import load_checkpoint
    from manabot.model.world import validate_agent_setup
    from manabot.training.execution import execute_regime
    from manabot.verify.store import VerifyStore
    from tests.training.test_recovery import tracked_updates

    regime = recipe(1, hours=0.05, setup_seconds=0, finalization_seconds=5)
    regime.id = "curriculum-current-world-proof"
    # This proof trains fresh weights on today's contract. The historical
    # campaign recipe and its retained checkpoints keep their original identity.
    regime.world = managym.WORLD_VERSION
    regime.observation = ObservationConfig()
    regime.agent.hidden_dim = 8
    regime.agent.num_attention_heads = 2
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.execution.device = "cpu"
    stage.streams = 2
    stage.transitions = 2
    stage.matchup_curriculum = "fixed"
    keys = managym.authored_deck_keys("elemental-world-v2")
    stage.adaptive = AdaptiveConfig(
        signal="ME",
        seed=17501,
        cells=tuple(
            MatchConfig.authored("elemental-world-v2", a, b) for a in keys for b in keys
        ),
    )
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(regime, 17501, tmp_path / "run", store)
    assert run.status == "completed"
    updates = tracked_updates(run, tmp_path / "run")
    assert len(updates) == 1
    diagnostic = updates[0]
    assert diagnostic["adaptive/completed_games"] > 0
    assert (
        diagnostic["adaptive/completed_roles"]
        == 2 * diagnostic["adaptive/completed_games"]
    )
    assert (
        len(
            [
                key
                for key in diagnostic
                if key.startswith("adaptive/priority_probabilities/")
            ]
        )
        == 100
    )
    assert diagnostic["adaptive/residual_count"] == sum(
        value
        for key, value in diagnostic.items()
        if key.startswith("adaptive/exposures/") and key.endswith("/transitions")
    )
    final, _ = load_checkpoint(run.stages[0].artifacts["raw"]["path"])
    for cell in stage.adaptive.cells:
        validate_agent_setup(final, Match(cell).to_rust())
