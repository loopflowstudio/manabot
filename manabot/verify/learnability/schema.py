"""Frozen diagnostic budgets and serializable evidence owned by the runner."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class PPOResult:
    before: float
    after: float
    no_update: float
    seconds: float
    transitions: int
    training_seconds: float
    games: int
    checkpoint: str
    passed: bool


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # learnability-2 trains the FiLM-conditioned policy on case set
    # action-cases-2; learnability-1 used the additive policy, whose verdict
    # depended on the starting weights, on action-cases-1.
    version: str = "learnability-2"
    seeds: tuple[int, ...] = (17, 29, 43)
    updates: int = 240
    learning_rate: float = 0.003
    hidden_dim: int = 32
    probability: float = 0.9
    invariance_tolerance: float = 0.0001
    timeout_seconds: int = 480
    ppo_timeout_seconds: int = 300
    ppo: bool = True


class Score(BaseModel):
    seed: int
    family: str
    split: str
    variant: int
    preference: int
    native: bool
    initial: float
    no_update: float
    trained: float
    identity_removed: float
    ablation_gap: float
    offer_error: float
    object_error: float
    passed: bool


class Result(BaseModel):
    version: str = "learnability-2"
    status: Literal["running", "passed", "failed", "error", "timeout"] = "running"
    source: str
    source_digest: str
    world: str | None = None
    config: Config
    native_sha256: str | None = None
    torch_version: str | None = None
    fit_cases: int = 0
    held_out_cases: int = 0
    # `action_cases.CASE_SET`; None in reports that predate case-set versions.
    case_set: str | None = None
    cases_digest: str | None = None
    platform: str
    started_at: str
    wall_seconds: float = 0
    cpu_seconds: float = 0
    training_seconds: dict[str, float] = Field(default_factory=dict)
    scores: list[Score] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    ppo: PPOResult | None = None
    errors: list[str] = Field(default_factory=list)
    skips: list[str] = Field(
        default_factory=lambda: [
            "Ability: synthetic projection only; native legality/resolution remains in Rust tests.",
            "Semantic-program cohort not trained; this suite isolates generic input identity.",
            "Historical w4/w5 runtimes are a separate pinned qualification, not nightly ablations.",
            "No full-game strength or MTG-118 acceptance claim.",
        ]
    )
