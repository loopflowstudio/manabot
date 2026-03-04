"""
test_observation.py
Tests for observation encoding in manabot using minimal test configurations.

This test suite verifies:
1. Parity between Python and C++ enums for game state representation
2. ObservationEncoder produces expected tensor shapes and keys
3. Validity masks accurately reflect game objects
4. Object-to-index mapping is consistent
5. Cross-observation independence
6. Focus indices validity
"""

from types import SimpleNamespace
from typing import Set, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

# Local imports
from manabot.env.observation import (
    ObservationEncoder,
    ObservationSpace,
    ObservationSpaceHypers,
    PhaseEnum,
    StepEnum,
    ZoneEnum,
)
import managym

# ─────────────────────────── Fixtures ───────────────────────────


@pytest.fixture(scope="session")
def hypers():
    """Create an encoder with minimal dimensions for testing."""
    return ObservationSpaceHypers(
        max_cards_per_player=3,
        max_permanents_per_player=2,
        max_actions=2,
        max_focus_objects=2,
    )


@pytest.fixture(scope="session")
def encoder(hypers):
    """Create an observation encoder for testing."""
    return ObservationEncoder(hypers)


@pytest.fixture(scope="session")
def observation_space(hypers):
    """Create an observation space using our test encoder."""
    return ObservationSpace(hypers)


@pytest.fixture(scope="session")
def player_configs():
    """Create consistent player configurations with minimal decks."""
    player_a = managym.PlayerConfig("Alice", {"Mountain": 10, "Grey Ogre": 20})
    player_b = managym.PlayerConfig("Bob", {"Forest": 10, "Llanowar Elves": 20})
    return [player_a, player_b]


@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    return managym.Env()


@pytest.fixture
def observation(env, player_configs) -> managym.Observation:
    """Get a fresh observation from a newly reset environment."""
    obs, _ = env.reset(player_configs)
    return obs


@pytest.fixture
def two_observations(
    env, player_configs
) -> Tuple[managym.Observation, managym.Observation]:
    """Get two consecutive observations for testing observation independence."""
    obs1, _ = env.reset(player_configs)
    # Take an action to get a different game state
    obs2, _, _, _, _ = env.step(0)
    return obs1, obs2


# ─────────────────────────── Tests ───────────────────────────


class TestEnumParity:
    """Verify that our Python enums match the C++ ones exactly."""

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (ZoneEnum.LIBRARY, managym.ZoneEnum.LIBRARY),
            (ZoneEnum.HAND, managym.ZoneEnum.HAND),
            (ZoneEnum.BATTLEFIELD, managym.ZoneEnum.BATTLEFIELD),
            (ZoneEnum.GRAVEYARD, managym.ZoneEnum.GRAVEYARD),
            (ZoneEnum.STACK, managym.ZoneEnum.STACK),
            (ZoneEnum.EXILE, managym.ZoneEnum.EXILE),
            (ZoneEnum.COMMAND, managym.ZoneEnum.COMMAND),
        ],
    )
    def test_zone_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (PhaseEnum.BEGINNING, managym.PhaseEnum.BEGINNING),
            (PhaseEnum.PRECOMBAT_MAIN, managym.PhaseEnum.PRECOMBAT_MAIN),
            (PhaseEnum.COMBAT, managym.PhaseEnum.COMBAT),
            (PhaseEnum.POSTCOMBAT_MAIN, managym.PhaseEnum.POSTCOMBAT_MAIN),
            (PhaseEnum.ENDING, managym.PhaseEnum.ENDING),
        ],
    )
    def test_phase_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize(
        "py_enum, cpp_enum",
        [
            (StepEnum.BEGINNING_UNTAP, managym.StepEnum.BEGINNING_UNTAP),
            (StepEnum.BEGINNING_UPKEEP, managym.StepEnum.BEGINNING_UPKEEP),
            (StepEnum.BEGINNING_DRAW, managym.StepEnum.BEGINNING_DRAW),
            (StepEnum.PRECOMBAT_MAIN_STEP, managym.StepEnum.PRECOMBAT_MAIN_STEP),
            (StepEnum.COMBAT_BEGIN, managym.StepEnum.COMBAT_BEGIN),
            (
                StepEnum.COMBAT_DECLARE_ATTACKERS,
                managym.StepEnum.COMBAT_DECLARE_ATTACKERS,
            ),
            (
                StepEnum.COMBAT_DECLARE_BLOCKERS,
                managym.StepEnum.COMBAT_DECLARE_BLOCKERS,
            ),
            (StepEnum.COMBAT_DAMAGE, managym.StepEnum.COMBAT_DAMAGE),
            (StepEnum.COMBAT_END, managym.StepEnum.COMBAT_END),
            (StepEnum.POSTCOMBAT_MAIN_STEP, managym.StepEnum.POSTCOMBAT_MAIN_STEP),
            (StepEnum.ENDING_END, managym.StepEnum.ENDING_END),
            (StepEnum.ENDING_CLEANUP, managym.StepEnum.ENDING_CLEANUP),
        ],
    )
    def test_step_enum(self, py_enum, cpp_enum):
        assert int(cpp_enum) == py_enum


class TestObservationEncoder:
    """Test the observation encoder's core functionality."""

    def test_default_padding_targets_72_objects(self):
        default_encoder = ObservationEncoder(ObservationSpaceHypers())
        total_objects = (
            2
            + default_encoder.cards_per_player * 2
            + default_encoder.perms_per_player * 2
        )
        assert total_objects == 72

    def get_expected_keys(self) -> Set[str]:
        """Get the complete set of expected keys in an encoded observation."""
        return {
            # Players
            "agent_player",
            "agent_player_valid",
            "opponent_player",
            "opponent_player_valid",
            # Cards
            "agent_cards",
            "agent_cards_valid",
            "opponent_cards",
            "opponent_cards_valid",
            # Permanents
            "agent_permanents",
            "agent_permanents_valid",
            "opponent_permanents",
            "opponent_permanents_valid",
            # Actions
            "actions",
            "actions_valid",
            "action_focus",
        }

    def test_encode_observation(self, observation_space, observation):
        """Test that encoding produces arrays with correct shapes and no invalid values."""
        encoded = observation_space.encode(observation)

        # Verify all expected keys are present
        expected_keys = self.get_expected_keys()
        assert set(encoded.keys()) == expected_keys, (
            f"Keys mismatch. Missing: {expected_keys - set(encoded.keys())}"
        )

        # Check shapes match specification
        for key in encoded:
            expected_shape = observation_space.shapes[key]
            assert encoded[key].shape == expected_shape, f"Shape mismatch for {key}"
            assert not np.isnan(encoded[key]).any(), f"NaN found in {key}"
            assert not np.isinf(encoded[key]).any(), f"Inf found in {key}"

    def test_validity_masks(self, observation_space, observation, hypers):
        """Test that validity masks correctly reflect the game state."""
        encoded = observation_space.encode(observation)

        # Check cards validity
        agent_cards_valid = encoded["agent_cards_valid"]
        actual_cards = len(observation.agent_cards)
        expected_valid = min(actual_cards, hypers.max_cards_per_player)
        valid_sum = int(agent_cards_valid.sum())  # Convert to Python int for comparison
        assert valid_sum == expected_valid, (
            f"Expected {expected_valid} valid cards, got {valid_sum}"
        )

        # Verify invalid slots are zeroed
        for key in [
            "agent_cards",
            "opponent_cards",
            "agent_permanents",
            "opponent_permanents",
        ]:
            valid_key = f"{key}_valid"
            invalid = ~encoded[valid_key].astype(bool)
            if np.any(invalid):
                array_slice = encoded[key][invalid]
                assert (array_slice == 0).all(), (
                    f"Non-zero values found in invalid {key} slots: {array_slice[array_slice != 0]}"
                )

    def test_player_features_include_turn_and_are_normalized(
        self, observation_space, observation
    ):
        encoded = observation_space.encode(observation)

        for key, player in (
            ("agent_player", observation.agent),
            ("opponent_player", observation.opponent),
        ):
            vec = encoded[key][0]

            assert vec[0] == pytest.approx(player.life / 20.0)
            assert vec[1] == pytest.approx(float(player.is_active))

            zone_slice = vec[2:9]
            expected_zones = np.array(player.zone_counts, dtype=np.float32) / 60.0
            np.testing.assert_allclose(zone_slice, expected_zones, atol=1e-6)

            phase_slice = vec[9:14]
            step_slice = vec[14:26]
            assert phase_slice.sum() == pytest.approx(1.0)
            assert step_slice.sum() == pytest.approx(1.0)
            assert phase_slice[int(observation.turn.phase)] == pytest.approx(1.0)
            assert step_slice[int(observation.turn.step)] == pytest.approx(1.0)

    def test_card_and_permanent_features_are_partitioned_and_normalized(
        self, observation_space, observation, hypers
    ):
        encoded = observation_space.encode(observation)

        for i, card in enumerate(
            observation.agent_cards[: hypers.max_cards_per_player]
        ):
            vec = encoded["agent_cards"][i]
            assert vec[7] == pytest.approx(1.0)
            assert vec[8] == pytest.approx(card.power / 10.0)
            assert vec[9] == pytest.approx(card.toughness / 10.0)
            assert vec[10] == pytest.approx(card.mana_cost.mana_value / 10.0)

        for i, card in enumerate(
            observation.opponent_cards[: hypers.max_cards_per_player]
        ):
            vec = encoded["opponent_cards"][i]
            assert vec[7] == pytest.approx(0.0)
            assert vec[8] == pytest.approx(card.power / 10.0)
            assert vec[9] == pytest.approx(card.toughness / 10.0)
            assert vec[10] == pytest.approx(card.mana_cost.mana_value / 10.0)

        for i, perm in enumerate(
            observation.agent_permanents[: hypers.max_permanents_per_player]
        ):
            vec = encoded["agent_permanents"][i]
            assert vec[0] == pytest.approx(1.0)
            assert vec[2] == pytest.approx(perm.damage / 10.0)

        for i, perm in enumerate(
            observation.opponent_permanents[: hypers.max_permanents_per_player]
        ):
            vec = encoded["opponent_permanents"][i]
            assert vec[0] == pytest.approx(0.0)
            assert vec[2] == pytest.approx(perm.damage / 10.0)

    def test_object_index_mapping(self, observation_space, observation, hypers):
        """Test that object IDs are mapped to consistent indices."""
        observation_space.encode(observation)
        mapping = observation_space.encoder.object_to_index

        # Find a sample card ID
        if observation.agent_cards:
            card_id = observation.agent_cards[0].id
            assert card_id in mapping, "Card ID not found in mapping"
            idx = mapping[card_id]
            # Index should be after players but before max objects
            total_objects = (
                2
                + hypers.max_cards_per_player * 2
                + hypers.max_permanents_per_player * 2
            )
            assert 2 <= idx < total_objects

    def test_focus_indices(self, observation_space, observation, hypers):
        """Test that action focus indices are valid."""
        encoded = observation_space.encode(observation)
        focus = encoded["action_focus"]

        # Calculate valid range for focus indices
        total_objects = (
            2 + hypers.max_cards_per_player * 2 + hypers.max_permanents_per_player * 2
        )

        # All indices should be either -1 (no focus) or within valid range
        valid = (focus == -1) | ((focus >= 0) & (focus < total_objects))
        assert valid.all(), f"Invalid focus indices found: {focus[~valid]}"

    def test_action_space_truncation_warning(self, monkeypatch):
        hypers = ObservationSpaceHypers(max_actions=2, max_focus_objects=2)
        encoder = ObservationEncoder(hypers)
        fake_actions = [
            SimpleNamespace(action_type=0, focus=[]),
            SimpleNamespace(action_type=1, focus=[]),
            SimpleNamespace(action_type=2, focus=[]),
        ]
        fake_obs = SimpleNamespace(action_space=SimpleNamespace(actions=fake_actions))
        fake_parent_logger = MagicMock()
        fake_logger = MagicMock()
        fake_parent_logger.getChild.return_value = fake_logger
        monkeypatch.setattr(
            "manabot.env.observation.getLogger", lambda *_: fake_parent_logger
        )

        actions, action_focus = encoder._encode_actions(fake_obs)

        assert actions.shape == (2, encoder.action_dim)
        assert action_focus.shape == (2, hypers.max_focus_objects)
        fake_logger.warning.assert_called_once_with("Action space truncated: 3 -> 2")

    def test_card_space_truncation_warning(self, monkeypatch):
        hypers = ObservationSpaceHypers(max_cards_per_player=1)
        encoder = ObservationEncoder(hypers)
        fake_cards = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        fake_parent_logger = MagicMock()
        fake_logger = MagicMock()
        fake_parent_logger.getChild.return_value = fake_logger
        monkeypatch.setattr(
            "manabot.env.observation.getLogger", lambda *_: fake_parent_logger
        )
        monkeypatch.setattr(
            encoder,
            "_encode_card_features",
            lambda *_: np.zeros(encoder.card_dim, dtype=np.float32),
        )

        encoded = encoder._encode_cards(fake_cards, is_mine=1.0)

        assert encoded.shape == (1, encoder.card_dim)
        fake_logger.warning.assert_called_once_with("Card list truncated: 2 -> 1")

    def test_permanent_space_truncation_warning(self, monkeypatch):
        hypers = ObservationSpaceHypers(max_permanents_per_player=1)
        encoder = ObservationEncoder(hypers)
        fake_perms = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        fake_parent_logger = MagicMock()
        fake_logger = MagicMock()
        fake_parent_logger.getChild.return_value = fake_logger
        monkeypatch.setattr(
            "manabot.env.observation.getLogger", lambda *_: fake_parent_logger
        )
        monkeypatch.setattr(
            encoder,
            "_encode_permanent_features",
            lambda *_: np.zeros(encoder.permanent_dim, dtype=np.float32),
        )

        encoded = encoder._encode_perms(fake_perms, is_mine=1.0)

        assert encoded.shape == (1, encoder.permanent_dim)
        fake_logger.warning.assert_called_once_with("Permanent list truncated: 2 -> 1")


if __name__ == "__main__":
    pytest.main([__file__])
