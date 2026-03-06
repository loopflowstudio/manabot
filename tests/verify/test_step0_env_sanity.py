"""Integration coverage for step0 environment sanity checks."""

import numpy as np
import pytest

from manabot.env import (
    Env,
    Match,
    ObservationSpace,
    PassivePolicy,
    RandomPolicy,
    Reward,
)
from manabot.infra.hypers import MatchHypers, ObservationSpaceHypers, RewardHypers
from manabot.verify.step0_env_sanity import _run_matchup
from manabot.verify.util import (
    TRUNCATION_INFO_KEYS,
    build_hypers,
    winner_from_info_or_obs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DECKS = {
    "elves": {"Forest": 24, "Llanowar Elves": 36},
    "ogres": {"Mountain": 24, "Grey Ogre": 36},
    "mixed": {"Mountain": 12, "Forest": 12, "Llanowar Elves": 18, "Grey Ogre": 18},
    "lands": {"Mountain": 30, "Forest": 30},
}

NUM_GAMES = 200


def _obs_space() -> ObservationSpace:
    return ObservationSpace(
        ObservationSpaceHypers(
            max_cards_per_player=60,
            max_permanents_per_player=30,
            max_actions=20,
        )
    )


def _run_random_matchup(
    hero_deck: dict[str, int],
    villain_deck: dict[str, int],
    *,
    num_games: int = NUM_GAMES,
    seed: int = 42,
) -> dict:
    """Run a random-vs-random matchup and return detailed results."""
    hypers = MatchHypers(hero_deck=hero_deck, villain_deck=villain_deck)
    match = Match(hypers)
    obs_space = _obs_space()
    reward = Reward(RewardHypers())
    env = Env(match, obs_space, reward, seed=seed, auto_reset=False)

    hero_wins = 0
    aborts = 0
    truncation_counts = {key: 0 for key in TRUNCATION_INFO_KEYS}
    loss_reasons: list[dict[str, str]] = []

    try:
        for game in range(num_games):
            obs, _ = env.reset(seed=seed + game)
            done = False
            info: dict = {}
            try:
                while not done:
                    valid = np.flatnonzero(obs["actions_valid"] > 0)
                    action = int(np.random.choice(valid))
                    obs, _, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    for key in TRUNCATION_INFO_KEYS:
                        truncation_counts[key] += int(bool(info.get(key, False)))
            except Exception:
                aborts += 1
                continue

            if winner_from_info_or_obs(info, env.last_raw_obs) == 0:
                hero_wins += 1

            game_reasons = {}
            for key in ("p0_loss_reason", "p1_loss_reason"):
                if key in info:
                    game_reasons[key] = info[key]
            if game_reasons:
                loss_reasons.append(game_reasons)
    finally:
        env.close()

    played = num_games - aborts
    return {
        "hero_win_rate": hero_wins / played if played > 0 else 0.0,
        "aborts": aborts,
        "loss_reasons": loss_reasons,
        **truncation_counts,
    }


# ---------------------------------------------------------------------------
# Original step0 tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_step0_matchups_have_no_aborts_or_truncations():
    np.random.seed(1234)

    hypers = build_hypers()
    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)

    num_games = 200
    random_vs_random = _run_matchup(
        RandomPolicy(),
        RandomPolicy(),
        obs_space,
        match,
        reward,
        num_games=num_games,
        seed=1,
    )
    random_vs_passive = _run_matchup(
        RandomPolicy(),
        PassivePolicy(),
        obs_space,
        match,
        reward,
        num_games=num_games,
        seed=10_001,
    )

    total_truncations = sum(
        random_vs_random[key] + random_vs_passive[key] for key in TRUNCATION_INFO_KEYS
    )
    total_aborts = random_vs_random["aborts"] + random_vs_passive["aborts"]

    assert random_vs_random["hero_win_rate"] > 0.15
    assert random_vs_passive["hero_win_rate"] > 0.90
    assert total_truncations == 0
    assert total_aborts == 0


# ---------------------------------------------------------------------------
# Matchup integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_creatures_beat_lands():
    """Any creature deck should reliably beat a lands-only deck."""
    np.random.seed(100)
    ogres = _run_random_matchup(DECKS["ogres"], DECKS["lands"])
    assert ogres["aborts"] == 0
    assert ogres["hero_win_rate"] > 0.95

    np.random.seed(200)
    elves = _run_random_matchup(DECKS["elves"], DECKS["lands"])
    assert elves["aborts"] == 0
    assert elves["hero_win_rate"] > 0.95


@pytest.mark.slow
def test_ogres_beat_elves():
    """2/2 bodies should dominate 1/1 bodies."""
    np.random.seed(300)
    result = _run_random_matchup(DECKS["ogres"], DECKS["elves"])
    assert result["aborts"] == 0
    assert result["hero_win_rate"] > 0.90


@pytest.mark.slow
def test_mirror_is_competitive():
    """Mirror match should be winnable by both sides."""
    np.random.seed(400)
    result = _run_random_matchup(DECKS["mixed"], DECKS["mixed"])
    assert result["aborts"] == 0
    assert 0.05 < result["hero_win_rate"] < 0.95


# Decks with creatures — used for stability/truncation tests.
# Lands-only is excluded: no creatures means no kills, games go to decking,
# and 60 land permanents overflow the observation space (by design).
_CREATURE_DECKS = {k: v for k, v in DECKS.items() if k != "lands"}


@pytest.mark.slow
def test_zero_aborts_all_matchups():
    """Engine should never crash across all deck combinations."""
    np.random.seed(500)
    deck_names = list(_CREATURE_DECKS.keys())
    for i, hero_name in enumerate(deck_names):
        for j, villain_name in enumerate(deck_names):
            result = _run_random_matchup(
                _CREATURE_DECKS[hero_name],
                _CREATURE_DECKS[villain_name],
                num_games=50,
                seed=500 + i * 100 + j,
            )
            assert result["aborts"] == 0, f"{hero_name} vs {villain_name} had aborts"


@pytest.mark.slow
def test_zero_truncations_all_matchups():
    """Observation space should fit all creature deck combinations."""
    np.random.seed(600)
    deck_names = list(_CREATURE_DECKS.keys())
    for i, hero_name in enumerate(deck_names):
        for j, villain_name in enumerate(deck_names):
            result = _run_random_matchup(
                _CREATURE_DECKS[hero_name],
                _CREATURE_DECKS[villain_name],
                num_games=50,
                seed=600 + i * 100 + j,
            )
            total = sum(result[key] for key in TRUNCATION_INFO_KEYS)
            assert total == 0, f"{hero_name} vs {villain_name} had truncations"


@pytest.mark.slow
def test_loss_reasons_reported():
    """Info dict should contain loss reason for the losing player."""
    np.random.seed(700)
    result = _run_random_matchup(DECKS["ogres"], DECKS["lands"], num_games=50)
    assert len(result["loss_reasons"]) > 0
    for reasons in result["loss_reasons"]:
        values = list(reasons.values())
        assert all(v in ("life_total", "deck_empty") for v in values)


@pytest.mark.slow
def test_winner_index_matches_observation():
    """winner_index in info dict should agree with life-total inference."""
    np.random.seed(800)
    hypers = MatchHypers(hero_deck=DECKS["mixed"], villain_deck=DECKS["mixed"])
    match = Match(hypers)
    obs_space = _obs_space()
    reward = Reward(RewardHypers())
    env = Env(match, obs_space, reward, seed=800, auto_reset=False)

    try:
        for game in range(50):
            obs, _ = env.reset(seed=800 + game)
            done = False
            info: dict = {}
            while not done:
                valid = np.flatnonzero(obs["actions_valid"] > 0)
                action = int(np.random.choice(valid))
                obs, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

            if "winner_index" not in info:
                continue

            winner_from_info = int(info["winner_index"])
            raw = env.last_raw_obs
            agent_idx = int(raw.agent.player_index)
            opp_idx = int(raw.opponent.player_index)

            # Infer winner from life totals when possible.
            if raw.agent.life <= 0 and raw.opponent.life > 0:
                assert winner_from_info == opp_idx
            elif raw.opponent.life <= 0 and raw.agent.life > 0:
                assert winner_from_info == agent_idx
    finally:
        env.close()
