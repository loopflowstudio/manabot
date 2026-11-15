"""Seat-owned terminal-return update with a prior-completed-batch baseline."""

from dataclasses import dataclass
import math
from types import ModuleType

from manabot.policy.draft import DraftTrajectory


@dataclass
class RunningBaseline:
    total: float = 0
    count: int = 0

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0


def update_rotisserie(
    backend: ModuleType,
    learner: object,
    trajectories: list[DraftTrajectory],
    rewards: list[float | None],
    baseline: RunningBaseline,
) -> float | None:
    """Exclude whole incomplete bundles. Read baseline before observing returns."""
    if backend.__name__ != "manabot.backends.torch":
        raise ValueError("Rotisserie currently requires the torch backend")
    if len(trajectories) != len(rewards):
        raise ValueError("one reward bundle per learner trajectory required")
    valid = [
        (t, r) for t, r in zip(trajectories, rewards, strict=True) if r is not None
    ]
    if not valid:
        return None
    if any(not math.isfinite(r) for _, r in valid):
        raise ValueError("reward must be a finite signed estimate")
    loss = backend.draft.update(
        learner, [t for t, _ in valid], [r for _, r in valid], baseline.value
    )
    baseline.total += sum(r for _, r in valid)
    baseline.count += len(valid)
    return loss
