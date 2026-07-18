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
        "Gray Ogre": 18,
    }


def _default_runs_dir() -> Path:
    runs_dir = os.getenv("MANABOT_RUNS_DIR")
    if runs_dir:
        return Path(runs_dir)
    return Path.cwd() / ".runs"


class BaseHypersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationSpaceHypers(BaseHypersModel):
    max_cards_per_player: int = 60
    # 30 -> 40: token-heavy GW Allies games exceed 30 battlefield entries.
    max_permanents_per_player: int = 40
    # 20 -> 32: the real Milestone-1 decks exceed 20 legal actions at some
    # priority windows (learn hands, wide waterbend boards) — the encoder
    # truncated and uniform-random-over-encoded policies never saw the tail.
    max_actions: int = 32
    max_focus_objects: int = 2
    max_events: int = 32


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
    # Belief-conditioning side-input channel (INT-14). 0 = no condition
    # channel (the plain Teacher-0/1 student; fully backward compatible).
    # >0 = the Agent appends one neutral object row to the attention sequence
    # built from a per-row condition_index (in [0, max_conditions)) and a
    # condition_weight scalar, provided as a side input by a conditional
    # shard. No belief head, range net, or per-hand value vector is added;
    # the policy and scalar value heads are unchanged. When the obs dict
    # omits the condition keys (arena inference), the Agent defaults to the
    # neutral True/uniform condition (index 0, weight 1.0).
    max_conditions: int = 0


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
    eval_interval: int = 100
    eval_num_games: int = 50

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
        if self.observation.max_events < 1:
            raise ValueError("max_events must be positive")
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
