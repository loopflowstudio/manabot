"""
hypers.py
Pydantic hyperparameter schemas shared across training and simulation.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _default_deck() -> dict[str, int]:
    return {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    }


def _default_runs_dir() -> Path:
    return Path(os.getenv("MANABOT_RUNS_DIR", str(Path.cwd() / ".manabot-runs")))


class BaseHypersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationSpaceHypers(BaseHypersModel):
    max_cards_per_player: int = 20
    max_permanents_per_player: int = 15
    max_actions: int = 10
    max_focus_objects: int = 2


class MatchHypers(BaseHypersModel):
    """Parameters passed to the match builder."""

    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: dict[str, int] = Field(default_factory=_default_deck)
    villain_deck: dict[str, int] = Field(default_factory=_default_deck)


class ExperimentHypers(BaseHypersModel):
    """Configuration for experiment tracking and runtime setup."""

    exp_name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    wandb: bool = True
    wandb_project_name: str = "manabot"
    runs_dir: Path = Field(default_factory=_default_runs_dir)
    log_level: str = "INFO"
    profiler_enabled: bool = False


class AgentHypers(BaseHypersModel):
    # Shared embedding space for game objects and actions.
    hidden_dim: int = 64
    # Number of attention heads used in the GameObjectAttention layer.
    num_attention_heads: int = 4
    attention_on: bool = True


class TrainHypers(BaseHypersModel):
    """Training-related hyperparameters."""

    total_timesteps: int = 20_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = float("inf")
    opponent_policy: str = "passive"

    @field_validator("target_kl", mode="before")
    @classmethod
    def _coerce_target_kl(cls, value: Any) -> Any:
        if isinstance(value, str) and value.lower() in {
            "inf",
            "+inf",
            "infinity",
            "+infinity",
        }:
            return float("inf")
        return value


class RewardHypers(BaseHypersModel):
    trivial: bool = False
    managym: bool = False
    win_reward: float = 1.0
    lose_reward: float = -1.0


class Hypers(BaseHypersModel):
    """Top-level training configuration."""

    observation: ObservationSpaceHypers = Field(default_factory=ObservationSpaceHypers)
    match: MatchHypers = Field(default_factory=MatchHypers)
    train: TrainHypers = Field(default_factory=TrainHypers)
    reward: RewardHypers = Field(default_factory=RewardHypers)
    agent: AgentHypers = Field(default_factory=AgentHypers)
    experiment: ExperimentHypers = Field(default_factory=ExperimentHypers)

    @model_validator(mode="after")
    def _validate_observation_limits(self) -> "Hypers":
        if self.observation.max_cards_per_player < 1:
            raise ValueError("max_cards_per_player must be positive")
        if self.observation.max_actions < 1:
            raise ValueError("max_actions must be positive")
        return self


class SimulationHypers(BaseHypersModel):
    """Hyperparameters for model simulation."""

    hero: str = "simple"
    villain: str = "default"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchHypers = Field(default_factory=MatchHypers)
    reward: RewardHypers = Field(default_factory=RewardHypers)
