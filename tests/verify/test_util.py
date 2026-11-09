"""Tests for verification helper utilities."""

from manabot.verify.util import build_verification_config


class TestVerificationConfig:
    def test_defaults_disable_attention(self) -> None:
        config = build_verification_config()
        assert config.agent.attention_on is False

    def test_nested_overrides_apply(self) -> None:
        config = build_verification_config(
            match={"hero_deck": {"Mountain": 20}},
        )
        assert config.match.hero_deck == {"Mountain": 20}
