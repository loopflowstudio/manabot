"""Small model, input and match configuration shared by admission and inference checks."""

from dataclasses import dataclass

from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.recipes.builders import PACK
from manabot.sim.matchup import selected_match


@dataclass
class PolicyInputs:
    """Fresh, mutable configuration: each test adjusts its own copy."""

    observation: ObservationConfig
    agent: ModelConfig
    match: MatchConfig


def recipe() -> PolicyInputs:
    return PolicyInputs(
        observation=ObservationConfig(),
        agent=ModelConfig(world=PACK, hidden_dim=8, num_attention_heads=2),
        match=selected_match(),
    )
