"""Bounded local production-path smoke recipe, without an experiment campaign."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from manabot.training.baseline import baseline
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.verify.store import VerifyStore

LOCAL_STEPS = 2
LOCAL_SECONDS = 120.0


def local_smoke(
    *,
    steps: int = LOCAL_STEPS,
    seconds: float = LOCAL_SECONDS,
    learning_seats: Literal["one", "both"] = "both",
) -> TrainingRegime:
    """Terminal-outcome self-play; complete games for both, exact quotas for one.

    This deliberately replaces, and is not equivalent to, the retired local
    fixed-seat PPO preset. The baseline model and horizon mapping are shared;
    this recipe omits independent outcome evaluation and makes no strength claim.
    """
    regime = baseline(steps, seconds=seconds, learning_seats=learning_seats).regime
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    regime.id = "local-smoke-v1"
    regime.wall_seconds = seconds
    stage.execution.wall_seconds = seconds
    stage.streams = 2
    stage.transitions = 16
    return regime


def run_local(
    *,
    steps: int = LOCAL_STEPS,
    seed: int = 197,
    out: Path | None = None,
    seconds: float = LOCAL_SECONDS,
    learning_seats: Literal["one", "both"] = "both",
) -> TrainingRun:
    plan = local_smoke(steps=steps, seconds=seconds, learning_seats=learning_seats)
    destination = out or Path(".runs") / (
        "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with VerifyStore(destination.parent / "training.sqlite") as store:
        return execute_regime(plan, seed, destination, store)
