"""`manabot.policy.checkpoint.require_supported` runs on every platform (NumPy only),
so this test guards the field names it reads against `ModelConfig` changes; the
backend tests that exercise it fully skip where no framework extra is installed."""

import pytest

from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import UnsupportedPolicy, require_supported


def test_require_supported_reads_only_current_model_fields() -> None:
    model = ModelConfig(policy_film=True)
    observation = ObservationConfig()
    with pytest.raises(UnsupportedPolicy) as refused:
        require_supported(model, observation)
    # Every field it reads exists: the refusal lists reasons, never an AttributeError.
    assert "policy_film" in str(refused.value)
