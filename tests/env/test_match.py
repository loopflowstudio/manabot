"""
test_match.py
"""

from types import SimpleNamespace
from typing import Dict

import pytest

# Local imports
from manabot.env.match import Match, Reward, parse_deck
from manabot.infra.hypers import RewardHypers

# Common Test Data
SAMPLE_DECKS = {
    "basic": {"Mountain": 12, "Forest": 12},
    "creature": {"Mountain": 24, "Lightning Bolt": 36},
    "default": {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Gray Ogre": 18,
    },
}


@pytest.fixture
def deck_formats() -> Dict[str, str]:
    """Provide sample deck strings in different formats for testing."""
    return {
        "json": '{"Mountain": 12, "Forest": 12}',
        "simple": "Mountain:12,Forest:12",
        "spaced": "Mountain: 12, Forest: 12",
        "complex": "Mountain:24,Lightning Bolt:36",
    }


@pytest.fixture
def sample_match() -> Match:
    """Create a match with a known configuration for testing."""
    return Match()


class TestDeckParsing:
    """Tests for parsing deck configurations from different formats."""

    @pytest.mark.parametrize("format_name", ["json", "simple", "spaced"])
    def test_valid_formats(self, deck_formats, format_name):
        """Test parsing of valid deck strings in different formats."""
        deck = parse_deck(deck_formats[format_name])
        assert deck == SAMPLE_DECKS["basic"], f"Failed to parse {format_name} format"

    @pytest.mark.parametrize(
        "invalid_input,expected_error",
        [
            ("{invalid json}", ValueError),
            ("Mountain,Forest", ValueError),  # Missing counts
            ("Mountain:twelve", ValueError),  # Non-integer count
            ("", ValueError),  # Empty string
            ("Mountain:12:extra", ValueError),  # Too many colons
        ],
    )
    def test_invalid_formats(self, invalid_input, expected_error):
        """Test that invalid deck strings raise appropriate errors."""
        with pytest.raises(expected_error):
            parse_deck(invalid_input)


class TestMatch:
    """Tests for match configuration and conversion."""

    def test_default_configuration(self):
        """Verify default match settings are correct."""
        match = Match()
        assert match.hero == "gaea"
        assert match.villain == "urza"
        assert match.hero_deck == SAMPLE_DECKS["default"]
        assert match.villain_deck == SAMPLE_DECKS["default"]

    def test_deck_independence(self):
        """Verify that deck modifications don't affect other instances."""
        match1 = Match()
        match2 = Match()
        match2.hero_deck["Mountain"] = 10
        match1.hero_deck["Mountain"] = 20
        assert match2.hero_deck["Mountain"] != 20

    def test_rust_conversion(self, sample_match):
        """Test conversion to managym PlayerConfig objects."""
        configs = sample_match.to_rust()
        assert len(configs) == 2

        hero_config, villain_config = configs
        assert hero_config.name == "gaea"
        assert dict(hero_config.decklist) == sample_match.hero_deck
        assert villain_config.name == "urza"
        assert dict(villain_config.decklist) == sample_match.villain_deck


@pytest.mark.parametrize("player_index", [0, 1])
def test_reward_correct_both_players(player_index):
    reward = Reward(RewardHypers(win_reward=7.0, lose_reward=-3.0))
    last_obs = SimpleNamespace(agent=SimpleNamespace(player_index=player_index))
    won_obs = SimpleNamespace(game_over=True, won=True)
    lost_obs = SimpleNamespace(game_over=True, won=False)

    assert reward.compute(0.0, last_obs, won_obs) == 7.0
    assert reward.compute(0.0, last_obs, lost_obs) == -3.0


def test_reward_progress_shaping():
    reward = Reward(
        RewardHypers(
            land_play_reward=1.0,
            creature_play_reward=1.0,
            opponent_life_loss_reward=1.0,
        )
    )
    battlefield = 2
    last_obs = SimpleNamespace(
        agent_cards=[
            SimpleNamespace(
                zone=battlefield,
                card_types=SimpleNamespace(is_land=True, is_creature=False),
            )
        ],
        opponent=SimpleNamespace(life=20),
        game_over=False,
        won=False,
    )
    new_obs = SimpleNamespace(
        agent_cards=[
            SimpleNamespace(
                zone=battlefield,
                card_types=SimpleNamespace(is_land=True, is_creature=False),
            ),
            SimpleNamespace(
                zone=battlefield,
                card_types=SimpleNamespace(is_land=True, is_creature=False),
            ),
            SimpleNamespace(
                zone=battlefield,
                card_types=SimpleNamespace(is_land=False, is_creature=True),
            ),
        ],
        opponent=SimpleNamespace(life=18),
        game_over=False,
        won=False,
    )

    assert reward.compute(0.0, last_obs, new_obs) == 4.0


def test_reward_progress_shaping_stacks_with_terminal_reward():
    reward = Reward(
        RewardHypers(
            win_reward=2.0,
            land_play_reward=1.0,
        )
    )
    battlefield = 2
    last_obs = SimpleNamespace(
        agent_cards=[],
        opponent=SimpleNamespace(life=20),
        game_over=False,
        won=False,
    )
    new_obs = SimpleNamespace(
        agent_cards=[
            SimpleNamespace(
                zone=battlefield,
                card_types=SimpleNamespace(is_land=True, is_creature=False),
            )
        ],
        opponent=SimpleNamespace(life=20),
        game_over=True,
        won=True,
    )

    assert reward.compute(0.0, last_obs, new_obs) == 3.0


def test_authored_setup_swap_and_hypers_roundtrip():
    from manabot.infra.hypers import MatchHypers
    import managym

    hypers = MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    match = Match(MatchHypers.model_validate_json(hypers.model_dump_json()))
    swapped = match.swapped()
    assert swapped.hero_sideboard == match.villain_sideboard
    assert swapped.villain_sideboard == match.hero_sideboard
    assert Match(swapped.hypers).to_rust_hero().sideboard == swapped.hero_sideboard
    swapped.villain_sideboard.clear()
    assert len(match.hero_sideboard) == 3
    assert len(hypers.hero_sideboard) == 3
    for first, second in ((match, match.swapped().swapped()),):
        engines = [managym.Env(seed=0) for _ in range(2)]
        engines[0].reset(first.to_rust())
        engines[1].reset(second.to_rust())
        assert engines[0].search_witness_json() == engines[1].search_witness_json()


def test_custom_setup_does_not_infer_sideboard():
    from manabot.infra.hypers import MatchHypers

    named = MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    custom = Match(
        MatchHypers(hero_deck=named.hero_deck, villain_deck=named.villain_deck)
    )
    assert custom.to_rust_hero().sideboard == {}
    assert custom.to_rust_villain().sideboard == {}


def test_reset_replacement_preserves_sideboard_on_auto_reset():
    from manabot.env import Env, ObservationSpace
    from manabot.infra.hypers import MatchHypers

    first = Match(MatchHypers(hero_deck={"Island": 8}, villain_deck={"Mountain": 8}))
    replacement = Match(
        MatchHypers(
            hero_deck={"Island": 8},
            villain_deck={"Mountain": 8},
            hero_sideboard={"Firebending Lesson": 1},
            villain_sideboard={"Island": 1},
        )
    ).swapped()
    env = Env(
        first, ObservationSpace(), Reward(RewardHypers()), seed=0, auto_reset=True
    )
    env.reset(options={"match": replacement})
    witness = env._engine.search_witness_json()
    for _ in range(100):
        _, _, _, _, info = env.step(0)
        if info["true_terminated"]:
            assert env._engine.search_witness_json() == witness
            return
    pytest.fail("bounded land-only match did not finish")


@pytest.mark.parametrize("reverse", [False, True])
def test_native_vector_reset_retains_learn_choices(reverse):
    import random

    from manabot.infra.hypers import MatchHypers
    import managym

    match = Match(
        MatchHypers.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    )
    if reverse:
        match = match.swapped()
    vector = managym.VectorEnv(1, seed=0, skip_trivial=False)
    scalar = managym.Env(seed=0, skip_trivial=False)
    # reset_all advances the lane seed; both resets must retain the same setup.
    for seed in (0, 1):
        scalar.set_seed(seed)
        obs, _ = scalar.reset(match.to_rust())
        vector_obs, _ = vector.reset_all(match.to_rust())[0]
        rng = random.Random(seed)
        for _ in range(3000):
            actions = obs.action_space.actions
            assert [
                (int(a.action_type), list(a.focus))
                for a in vector_obs.action_space.actions
            ] == [(int(a.action_type), list(a.focus)) for a in actions]
            retrievals = [
                i
                for i, action in enumerate(actions)
                if int(action.action_type) == int(managym.ActionEnum.LEARN_TAKE_LESSON)
            ]
            if retrievals:
                assert len(retrievals) == 3
                index = retrievals[-1]
                after, *_ = scalar.step(index)
                vector_after, *_ = vector.step([index])[0]
                assert vector_after.agent.known_hand == after.agent.known_hand
                assert vector_after.opponent.known_hand == after.opponent.known_hand
                assert (
                    sum(after.agent.known_hand.values())
                    + sum(after.opponent.known_hand.values())
                    == 1
                )
                break
            assert not obs.game_over, "match ended before Learn"
            index = rng.randrange(len(actions))
            obs, *_ = scalar.step(index)
            vector_obs, *_ = vector.step([index])[0]
        else:
            pytest.fail("no Learn in bounded vector prefix")


@pytest.mark.parametrize(
    "sideboard",
    [
        {"Island": 0},
        {"Island": -1},
        {"Island": True},
        {"Island": 1.0},
        {"Island": "1"},
        {"": 1},
        None,
    ],
)
def test_sideboard_counts_are_not_coerced(sideboard):
    from manabot.infra.hypers import MatchHypers

    with pytest.raises(ValueError, match="positive integer counts"):
        MatchHypers(hero_sideboard=sideboard)


if __name__ == "__main__":
    pytest.main([__file__])
