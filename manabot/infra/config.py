"""
Configuration models shared across training, simulation and runtime setup.
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from manabot.decklist import ImportedMatch


def _default_deck() -> dict[str, int]:
    return {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Gray Ogre": 18,
    }


def _default_runs_dir() -> Path:
    runs_dir = os.getenv("MANABOT_RUNS_DIR")
    if runs_dir:
        return Path(runs_dir)
    return Path.cwd() / ".runs"


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationConfig(BaseConfig):
    policy_input_version: Literal[1, 2, 3, 4] = Field(
        default=4, exclude_if=lambda value: value == 1
    )

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "ObservationConfig":
        """Missing version means the frozen input-v1 contract, never today's default."""
        return cls(**{"policy_input_version": 1, **payload})

    max_cards_per_player: int = 60
    # 30 -> 40: token-heavy GW Allies games exceed 30 battlefield entries.
    max_permanents_per_player: int = 40
    # Fits all Learn choices in the selected setup. Larger action sets fail
    # encoding explicitly; this is not a universal priority/combat bound.
    max_actions: int = 64
    max_focus_objects: int = 2
    max_events: int = 32
    policy_history_version: Literal[0, 1] = Field(
        default=0, exclude_if=lambda value: value == 0
    )


class MatchConfig(BaseConfig):
    """Parameters passed to the match builder."""

    content_pack: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    imported_decks: ImportedMatch | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: dict[str, int] = Field(default_factory=_default_deck)
    villain_deck: dict[str, int] = Field(default_factory=_default_deck)
    hero_sideboard: dict[str, int] = Field(default_factory=dict)
    villain_sideboard: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_imported_decks(self) -> "MatchConfig":
        imported = self.imported_decks
        if imported is not None:
            imported.validate_setup(
                content_pack=self.content_pack,
                hero_deck=self.hero_deck,
                villain_deck=self.villain_deck,
                hero_sideboard=self.hero_sideboard,
                villain_sideboard=self.villain_sideboard,
            )
        return self

    @field_validator("hero_sideboard", "villain_sideboard", mode="before")
    @classmethod
    def validate_sideboard_counts(cls, value: Any) -> Any:
        if not isinstance(value, dict) or any(
            not isinstance(name, str)
            or not name
            or type(count) is not int
            or count <= 0
            for name, count in value.items()
        ):
            raise ValueError("sideboard must map card names to positive integer counts")
        return value

    @classmethod
    def authored(
        cls,
        pack_key: str,
        hero_deck: str,
        villain_deck: str,
        *,
        hero: str = "gaea",
        villain: str = "urza",
    ) -> "MatchConfig":
        """Resolve both complete setups from the compiled rules pack."""
        import managym

        first = managym.authored_deck_setup(pack_key, hero_deck)
        second = managym.authored_deck_setup(pack_key, villain_deck)
        return cls(
            content_pack=(
                pack_key
                if hero_deck == villain_deck
                or len(managym.authored_deck_keys(pack_key)) > 2
                else None
            ),
            hero=hero,
            villain=villain,
            hero_deck=first.decklist,
            villain_deck=second.decklist,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
        )


class RuntimeConfig(BaseConfig):
    """Configuration for experiment tracking and runtime setup."""

    name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    runs_dir: Path = Field(default_factory=_default_runs_dir)
    log_level: str = "INFO"


class ModelConfig(BaseConfig):
    """Model construction contract, independent of learning and execution settings.

    Checkpoints retain the serialized ``agent_hypers`` dictionary key; renaming
    this Python type changes neither field meanings nor ordinary reload admission.
    """

    compound_decisions: bool = Field(default=False, exclude_if=lambda value: not value)
    # Public identity/history v1; false preserves historical receipt bytes.
    recent_events: bool = Field(default=False, exclude_if=lambda value: not value)
    semantic_pack: str | None = None
    # Serialized architecture choice; categorical logits are loss/draw/win.
    value_kind: Literal["scalar", "categorical_wdl"] = "scalar"
    value_aggregation: Literal["historical_mean", "masked_mean", "value_token"] = (
        "historical_mean"
    )
    # Scale each action embedding by the pooled game state before the policy
    # head (FiLM). Omitted is the historical additive policy, so frozen
    # recipes and checkpoints keep their meaning.
    policy_film: bool = Field(default=False, exclude_if=lambda value: not value)
    attention_layers: int = Field(default=1, ge=1)
    # Omitted defaults preserve historical recipe and checkpoint identities.
    attention_feedforward_dim: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    # Shared embedding space for game objects and actions.
    hidden_dim: int = 64
    # Number of attention heads used in the GameObjectAttention layer.
    num_attention_heads: int = 4
    attention_on: bool = True
    # Canonical belief rows are an opt-in checkpoint ABI. A positive bucket
    # count requires explicit belief tensors on every forward pass; there is
    # no neutral or positional-condition fallback on this path.
    belief_count_buckets: int = 0
    belief_card_vocab_size: int = 0
    belief_owner_role_vocab_size: int = 2
    belief_hidden_zone_vocab_size: int = 7

    @model_validator(mode="after")
    def validate_value_architecture(self) -> "ModelConfig":
        if self.hidden_dim < 1 or self.num_attention_heads < 1:
            raise ValueError("embedding width and head count must be positive")
        if self.attention_on and self.hidden_dim % self.num_attention_heads:
            raise ValueError("embedding width must be divisible by attention heads")
        if not self.attention_on and (
            self.value_aggregation == "value_token"
            or self.attention_layers != 1
            or self.attention_feedforward_dim is not None
        ):
            raise ValueError("value token and stacked layers require attention")
        if self.compound_decisions and (
            self.value_aggregation != "historical_mean" or self.attention_layers != 1
        ):
            raise ValueError(
                "compound critic does not support value aggregation variants"
            )
        if self.compound_decisions and self.policy_film:
            raise ValueError(
                "compound decoder owns its own state conditioning; "
                "set agent.policy_film=false"
            )
        return self


class RewardConfig(BaseConfig):
    trivial: bool = False
    managym: bool = False
    win_reward: float = 1.0
    lose_reward: float = -1.0
    land_play_reward: float = 0.0
    creature_play_reward: float = 0.0
    opponent_life_loss_reward: float = 0.0
    # Potential-based shaping (Ng, Harada & Russell 1999): adds
    # gamma * Phi(s') - Phi(s) to every hero step reward, with Phi(terminal)
    # treated as 0 so the potential telescopes out over an episode. Phi is a
    # hero-perspective board-state potential:
    #   Phi(s) = potential_land_weight * (hero bf lands - villain bf lands)
    #          + potential_creature_weight * (hero bf creatures - villain bf creatures)
    #          + potential_life_weight * (hero life - villain life) / 20
    # potential_gamma must match the training discount (train.gamma) for
    # policy invariance to hold.
    potential_enabled: bool = False
    potential_gamma: float = 0.99
    potential_land_weight: float = 0.03
    potential_creature_weight: float = 0.06
    potential_life_weight: float = 0.2


class SimulationConfig(BaseConfig):
    """Hyperparameters for model simulation."""

    hero: str = "random"
    villain: str = "default"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchConfig = Field(default_factory=MatchConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="after")
    def validate_imported_players(self) -> "SimulationConfig":
        if self.match.imported_decks is not None and any(
            player not in {"random", "default"} for player in (self.hero, self.villain)
        ):
            raise ValueError(
                "Imported decks currently support random/default simulation players; checkpoint setup admission is not available"
            )
        return self
