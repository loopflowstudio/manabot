"""
test_match.py
"""

import pytest
import argparse
from typing import Dict

from manabot.env.match import Match, parse_deck

# Common Test Data
SAMPLE_DECKS = {
    "basic": {"Mountain": 12, "Forest": 12},
    "creature": {"Mountain": 24, "Lightning Bolt": 36},
    "default": {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    }
}

@pytest.fixture
def deck_formats() -> Dict[str, str]:
    """Provide sample deck strings in different formats for testing."""
    return {
        "json": '{"Mountain": 12, "Forest": 12}',
        "simple": "Mountain:12,Forest:12",
        "spaced": "Mountain: 12, Forest: 12",
        "complex": "Mountain:24,Lightning Bolt:36"
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

    @pytest.mark.parametrize("invalid_input,expected_error", [
        ("{invalid json}", ValueError),
        ("Mountain,Forest", ValueError),  # Missing counts
        ("Mountain:twelve", ValueError),  # Non-integer count
        ("", ValueError),  # Empty string
        ("Mountain:12:extra", ValueError),  # Too many colons
    ])
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

    def test_cpp_conversion(self, sample_match):
        """Test conversion to C++ PlayerConfig objects."""
        cpp_configs = sample_match.to_cpp()
        assert len(cpp_configs) == 2
        
        hero_config, villain_config = cpp_configs
        assert hero_config.name == "gaea"
        assert dict(hero_config.decklist) == sample_match.hero_deck
        assert villain_config.name == "urza"
        assert dict(villain_config.decklist) == sample_match.villain_deck

if __name__ == "__main__":
    pytest.main([__file__])