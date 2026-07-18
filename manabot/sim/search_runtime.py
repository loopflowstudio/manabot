"""Dependency-light constants and receipts shared by search implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_PLAYOUT_STEPS = 2000


@dataclass
class SearchStats:
    """Accumulated cost and behavior for one search player."""

    decisions: int = 0
    seconds: float = 0.0
    simulations: int = 0
    cap_hits: int = 0
    decision_seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "decisions": float(self.decisions),
            "seconds": self.seconds,
            "simulations": float(self.simulations),
            "cap_hits": float(self.cap_hits),
        }
