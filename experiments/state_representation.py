"""Same-input state encoder comparison; declaration does not allocate compute.

All arms run the existing complete-game self-play executor and checkpoint path.
The protocol is in scratch/5da5f4-design-effective-state.md. Scientific launch
is blocked pending a current quote and a frozen allocation; the smoke recipe is
only an engineering check and must never be scored as a training replicate.
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from etude.launch.records import Run

from experiments.batched_baseline import batched_baseline
from manabot.experiment.job import TrainingJob
from manabot.recipes.local import local_smoke
from manabot.training.regime import TrainingRegime, TrainSelfPlay

Arm = Literal["rows", "typed_pool", "typed_gru"]
ARMS: tuple[Arm, ...] = ("rows", "typed_pool", "typed_gru")
SEEDS = (73201, 73202, 73203, 73204)


def recipe(arm: Arm, *, smoke: bool = False) -> TrainingRegime:
    """Same v5 observations and learner across arms; no recurrence or pilot edits."""
    regime = local_smoke(steps=1, seconds=120) if smoke else batched_baseline()
    regime.id = f"state-representation-{arm}" + ("-smoke" if smoke else "")
    regime.agent.state_encoder = arm
    regime.agent.policy_film = True
    regime.agent.recent_events = False
    regime.observation.policy_input_version = 5
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "both"
    if not smoke:
        stage.games_per_stream = 1
        stage.updates = 128
        # Setup and finalization must also fit the rental's one-hour window.
        stage.execution.wall_seconds = 2700
        regime.wall_seconds = 2700
    return TrainingRegime.model_validate(regime.model_dump())


def runs(revision: str, canary: bool = False) -> list[Run]:
    """Twelve declared runs, no submission; $1/run is only a reservation ceiling.

    Actual rental price and cohort controls must be inspected before launch.
    This module does not authorize a default machine or the existing pilot pool.
    """
    result: list[Run] = []
    regimes = {arm: recipe(arm) for arm in ARMS}
    if canary:
        for regime in regimes.values():
            regime.id += "-canary"
            stage = regime.stages[0]
            assert isinstance(stage, TrainSelfPlay)
            stage.updates = 1
            stage.execution.wall_seconds = regime.wall_seconds = 300
    jobs = {
        arm: TrainingJob(
            regime=regimes[arm],
            experiment="state_representation",
            milestones=() if canary else (32, 64, 96),
            finalization_seconds=60 if canary else 120,
        ).job(
            repo=Path(__file__).resolve().parents[1],
            revision=revision,
            max_hours=0.2 if canary else 1,
            max_usd=Decimal("0.30") if canary else Decimal("1"),
        )
        for arm in ARMS
    }
    # Interleave treatments within each seed to reduce machine/time confounds.
    for index, seed in enumerate((73200,) if canary else SEEDS):
        for arm in (*ARMS[index % 3 :], *ARMS[: index % 3]):
            result.extend(jobs[arm].runs(seed))
    return result
