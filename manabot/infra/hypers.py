"""
hypers.py
Dataclass hyperparameter schemas shared across training and simulation.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict


@dataclass
class ObservationSpaceHypers:
    max_cards_per_player: int = 20
    max_permanents_per_player: int = 15
    max_actions: int = 10
    max_focus_objects: int = 2


@dataclass
class MatchHypers:
    """Parameters passed to the match builder."""

    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: Dict[str, int] = field(
        default_factory=lambda: {
            "Mountain": 12,
            "Forest": 12,
            "Llanowar Elves": 18,
            "Grey Ogre": 18,
        }
    )
    villain_deck: Dict[str, int] = field(
        default_factory=lambda: {
            "Mountain": 12,
            "Forest": 12,
            "Llanowar Elves": 18,
            "Grey Ogre": 18,
        }
    )


@dataclass
class ExperimentHypers:
    """Configuration for experiment tracking and runtime setup."""

    exp_name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    wandb: bool = True
    wandb_project_name: str = "manabot"
    runs_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MANABOT_RUNS_DIR", str(Path.cwd() / ".manabot-runs"))
        )
    )
    log_level: str = "INFO"
    profiler_enabled: bool = False


@dataclass
class AgentHypers:
    # Shared embedding space for game objects and actions.
    hidden_dim: int = 64
    # Number of attention heads used in the GameObjectAttention layer.
    num_attention_heads: int = 4
    attention_on: bool = True


@dataclass
class TrainHypers:
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


@dataclass
class RewardHypers:
    trivial: bool = False
    managym: bool = False
    win_reward: float = 1.0
    lose_reward: float = -1.0


@dataclass
class Hypers:
    """Top-level training configuration."""

    observation: ObservationSpaceHypers = field(default_factory=ObservationSpaceHypers)
    match: MatchHypers = field(default_factory=MatchHypers)
    train: TrainHypers = field(default_factory=TrainHypers)
    reward: RewardHypers = field(default_factory=RewardHypers)
    agent: AgentHypers = field(default_factory=AgentHypers)
    experiment: ExperimentHypers = field(default_factory=ExperimentHypers)

    def __post_init__(self):
        if self.observation.max_cards_per_player < 1:
            raise ValueError("max_cards_per_player must be positive")
        if self.observation.max_actions < 1:
            raise ValueError("max_actions must be positive")


@dataclass
class SimulationHypers:
    """Hyperparameters for model simulation."""

    hero: str = "simple"
    villain: str = "default"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchHypers = field(default_factory=MatchHypers)
    reward: RewardHypers = field(default_factory=RewardHypers)


def initialize() -> None:
    """Backward-compatible no-op kept for callers from pre-Typer runtime."""

    return None
