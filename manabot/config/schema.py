"""Pydantic schemas for CLI/runtime configuration."""

import os
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from manabot.infra.hypers import (
    AgentHypers,
    ExperimentHypers,
    Hypers,
    MatchHypers,
    ObservationSpaceHypers,
    RewardHypers,
    SimulationHypers,
    TrainHypers,
)


class ObservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cards_per_player: int = 20
    max_permanents_per_player: int = 15
    max_actions: int = 10
    max_focus_objects: int = 2


class MatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: Dict[str, int] = Field(
        default_factory=lambda: {
            "Mountain": 12,
            "Forest": 12,
            "Llanowar Elves": 18,
            "Grey Ogre": 18,
        }
    )
    villain_deck: Dict[str, int] = Field(
        default_factory=lambda: {
            "Mountain": 12,
            "Forest": 12,
            "Llanowar Elves": 18,
            "Grey Ogre": 18,
        }
    )


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    def _coerce_target_kl(cls, value):
        if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
            return float("inf")
        return value


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trivial: bool = False
    managym: bool = False
    win_reward: float = 1.0
    lose_reward: float = -1.0


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = 64
    num_attention_heads: int = 4
    attention_on: bool = True


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    wandb: bool = True
    wandb_project_name: str = "manabot"
    runs_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MANABOT_RUNS_DIR", str(Path.cwd() / ".manabot-runs"))
        )
    )
    log_level: str = "INFO"
    profiler_enabled: bool = False


class TrainingConfig(BaseModel):
    """Top-level training runtime schema."""

    model_config = ConfigDict(extra="forbid")

    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    def to_hypers(self) -> Hypers:
        return Hypers(
            observation=ObservationSpaceHypers(**self.observation.model_dump()),
            match=MatchHypers(**self.match.model_dump()),
            train=TrainHypers(**self.train.model_dump()),
            reward=RewardHypers(**self.reward.model_dump()),
            agent=AgentHypers(**self.agent.model_dump()),
            experiment=ExperimentHypers(**self.experiment.model_dump()),
        )


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hero: str = "simple"
    villain: str = "default"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchConfig = Field(default_factory=MatchConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)


class SimulationConfig(BaseModel):
    """Top-level simulation runtime schema."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    sim: SimConfig = Field(default_factory=SimConfig)

    def to_hypers(self) -> tuple[SimulationHypers, ExperimentHypers]:
        sim_dump = self.sim.model_dump()
        match_cfg = MatchHypers(**sim_dump.pop("match"))
        reward_cfg = RewardHypers(**sim_dump.pop("reward"))

        sim_hypers = SimulationHypers(match=match_cfg, reward=reward_cfg, **sim_dump)
        experiment_hypers = ExperimentHypers(**self.experiment.model_dump())
        return sim_hypers, experiment_hypers
