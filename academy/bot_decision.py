"""Recorded policy outputs, captured at choice time rather than reconstructed."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyFactor:
    chosen: int
    probabilities: list[float]
    value: float


@dataclass(frozen=True)
class BeliefSummary:
    model: str
    digest: str
    condition_mass: float
    sampled_worlds: int


@dataclass(frozen=True)
class BotDecision:
    action: int
    probabilities: list[float] = field(default_factory=list)
    value: float | None = None
    value_semantic: str | None = None
    search_scores: list[float | None] = field(default_factory=list)
    unavailable: str | None = None
    factors: list[PolicyFactor] = field(default_factory=list)
    belief: BeliefSummary | None = None
