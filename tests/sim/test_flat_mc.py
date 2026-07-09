"""Tests for the flat determinized Monte Carlo player (exp-02 / wave C3).

Covers the engine search primitives exposed through PyO3 (clone independence,
determinization invariants, playout termination) and the Python matchup loop
(seat balancing, aggregation, argmax action selection).
"""

import numpy as np
import pytest

from manabot.sim.flat_mc import (
    FlatMCPlayer,
    GameRecord,
    RandomMatchupPlayer,
    aggregate_records,
    play_games,
    wilson_interval,
)
from manabot.verify.util import INTERACTIVE_DECK, wilson_lower_bound
import managym

SEARCH_4 = {"kind": "search", "sims": 4}
RANDOM = {"kind": "random"}


def _fresh_engine(seed: int = 0) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True)
    configs = [
        managym.PlayerConfig("hero", dict(INTERACTIVE_DECK)),
        managym.PlayerConfig("villain", dict(INTERACTIVE_DECK)),
    ]
    env.reset(configs)
    return env


# -----------------------------------------------------------------------------
# Engine primitives (PyO3 surface)
# -----------------------------------------------------------------------------


def test_clone_env_is_independent():
    env = _fresh_engine(seed=3)
    clone = env.clone_env()
    winner = clone.random_playout(seed=7, max_steps=100_000)
    assert clone.is_game_over()
    assert winner in (0, 1)
    assert not env.is_game_over()
    # Original still has its pre-clone action space.
    assert env.action_count() >= 1


def test_determinize_preserves_action_space_and_is_seeded():
    env = _fresh_engine(seed=5)
    count_before = env.action_count()
    agent_before = env.current_agent_index()
    env.determinize(seed=11)
    assert env.action_count() == count_before
    assert env.current_agent_index() == agent_before

    # Same seed on identical clones gives identical playout results.
    a = _fresh_engine(seed=5)
    b = _fresh_engine(seed=5)
    a.determinize(seed=11)
    b.determinize(seed=11)
    assert a.random_playout(seed=13) == b.random_playout(seed=13)


def test_random_playout_step_cap_returns_none():
    env = _fresh_engine(seed=9)
    assert env.random_playout(seed=1, max_steps=1) is None
    assert not env.is_game_over()


def test_flat_mc_scores_shape_and_bounds():
    env = _fresh_engine(seed=1)
    count = env.action_count()
    scores, sims, caps = env.flat_mc_scores(2, 2, 42, 2000)
    assert len(scores) == count
    assert sims == count * 4
    assert caps <= sims
    assert all(0.0 <= s <= 1.0 for s in scores)


# -----------------------------------------------------------------------------
# Python player + matchup loop
# -----------------------------------------------------------------------------


def test_flat_mc_player_sims_split():
    player = FlatMCPlayer(64)
    assert player.worlds * player.rollouts == 64
    tiny = FlatMCPlayer(1)
    assert tiny.worlds == 1 and tiny.rollouts == 1


def test_flat_mc_player_rejects_bad_sims():
    with pytest.raises(ValueError):
        FlatMCPlayer(0)


def test_random_matchup_player_uses_valid_mask():
    player = RandomMatchupPlayer(seed=0)
    obs = {"actions_valid": np.array([0.0, 1.0, 0.0, 1.0])}
    for _ in range(10):
        assert player.act(None, obs) in (1, 3)


def test_play_games_seat_balances_and_records():
    result = play_games(SEARCH_4, RANDOM, num_games=4, seed=0)
    assert len(result.records) == 4
    assert [r.hero_seat for r in result.records] == [0, 1, 0, 1]
    assert result.hero_search is not None
    assert result.hero_search.decisions > 0
    assert result.hero_search.simulations > 0
    assert result.villain_search is None
    for record in result.records:
        assert record.steps > 0


def test_play_games_game_offset_continues_seat_alternation():
    result = play_games(SEARCH_4, RANDOM, num_games=2, seed=0, game_offset=1)
    assert [r.hero_seat for r in result.records] == [1, 0]


def test_aggregate_records_per_seat_and_ci():
    records = [
        GameRecord(game_index=i, hero_seat=i % 2, hero_won=(i % 2 == 0), winner=0, steps=10)
        for i in range(10)
    ]
    metrics = aggregate_records(records)
    assert metrics["num_games"] == 10.0
    assert metrics["win_rate"] == 0.5
    assert metrics["win_rate_on_play"] == 1.0
    assert metrics["win_rate_on_draw"] == 0.0
    assert 0.0 <= metrics["win_ci_lower"] < 0.5 < metrics["win_ci_upper"] <= 1.0


def test_wilson_interval_matches_verify_lower_bound():
    lo, hi = wilson_interval(30, 100)
    assert lo == pytest.approx(wilson_lower_bound(30, 100))
    assert lo < 0.3 < hi
