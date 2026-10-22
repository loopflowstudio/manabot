"""The ordinary baseline recipe, above regimes and below experiment comparisons.

`baseline(steps)` resolves the same learning rule for CLI and nightly callers.
A recipe owns its horizon, checkpoints and independent evaluation; TrainingRegime
still owns the learner. No diagnostic labels or action-specific rewards enter
training. `run_baseline` uses the existing isolated experiment coordinator.
"""

from datetime import datetime, timezone
from pathlib import Path
import socket
from typing import Literal

from pydantic import Field

from manabot.arena.match import selected_match
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.training.builders import PACK
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import (
    ExperimentRun,
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import (
    DampedPolicyLearning,
    Execution,
    IterationSchedule,
    Strict,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import MonitorProtocol
from manabot.verify.learning_regression import report_baseline
import managym

BASELINE_STEPS = 128
# v3 combines the FiLM policy with learning from both seats by default.
BASELINE_VERSION = "baseline-v3"
# Separate from the existing scientific and monitoring cohorts. Repeated nightly
# use makes these regression/development deals, not a scientific holdout forever.
BASELINE_DEALS = tuple(range(1_915_100_000, 1_915_100_024))


class TrainingRecipe(Strict):
    """Resolved, serializable run intention; constructing one launches nothing."""

    name: str
    version: str
    steps: int = Field(ge=1)
    regime: TrainingRegime
    checkpoint_updates: int = Field(ge=1)
    evaluation: MonitorProtocol
    wall_seconds: float = Field(gt=0)

    def experiment(self, seed: int) -> Experiment:
        return Experiment(
            name=self.version,
            baseline=Baseline.capture(self.version, self.regime),
            schedule=ExperimentSchedule(
                seeds=(seed,),
                hardware="baseline-cpu",
                wall_seconds=self.wall_seconds,
                process_seconds=2 * self.wall_seconds,
                checkpoint_updates=self.checkpoint_updates,
                monitoring=MonitoringBudget(
                    seconds=self.wall_seconds,
                    attempt_seconds=self.wall_seconds / 2,
                    include_initial=True,
                    # Arena evaluation is expensive: initial and endpoint only.
                    # Intermediate exports are reused by cheap frozen probes.
                    updates=(0, self.steps),
                    protocol=self.evaluation,
                ),
                disk_reserve_bytes=1024**3,
            ),
        )


def baseline(
    steps: int = BASELINE_STEPS,
    *,
    seconds: float | None = None,
    learning_seats: Literal["one", "both"] = "both",
) -> TrainingRecipe:
    """Resolve horizon-scaled Ataraxos self-play on the current authored world.

    The scalar, width-64, depth-2 model follows the existing ordinary self-play
    baseline, with the FiLM policy that new runs train. The explicit schedule
    mapping is a new recipe choice: its first and last rates match iterations 1
    and 10,000 of the historical rule. Short runs are compressed schedules, not
    prefixes of a longer run. Reward remains the executor's terminal game
    outcome. Wall time is only a failure bound.
    """
    schedule = IterationSchedule(steps=steps)
    allowance = seconds if seconds is not None else max(1680.0, steps * 8.0 + 600)
    return TrainingRecipe(
        name="baseline",
        version=BASELINE_VERSION,
        steps=steps,
        checkpoint_updates=max(1, steps // 2),
        evaluation=MonitorProtocol(deal_seeds=BASELINE_DEALS, opponent="random"),
        wall_seconds=allowance,
        regime=TrainingRegime(
            id=BASELINE_VERSION,
            world=managym.WORLD_VERSION,
            match=selected_match(),
            observation=ObservationConfig(),
            agent=ModelConfig(
                hidden_dim=64,
                attention_layers=2,
                num_attention_heads=4,
                semantic_pack=PACK,
                value_kind="scalar",
                value_aggregation="masked_mean",
                policy_film=True,
            ),
            stages=[
                TrainSelfPlay(
                    id="self-play",
                    operation="train_self_play",
                    updates=steps,
                    streams=4,
                    transitions=64,
                    behavior="current-self",
                    learning_seats=learning_seats,
                    learning=DampedPolicyLearning(
                        gradient="damped_policy", iteration_schedule=schedule
                    ),
                    execution=Execution(wall_seconds=allowance * 0.85, threads=1),
                )
            ],
            wall_seconds=allowance * 0.85,
            schedule_clock="iteration_fraction",
        ),
    )


def run_baseline(
    *,
    steps: int = BASELINE_STEPS,
    seed: int = 197,
    out: Path | None = None,
    seconds: float | None = None,
    learning_seats: Literal["one", "both"] = "both",
) -> ExperimentRun:
    """Train, export and evaluate; retain every failure without automatic retry."""
    plan = baseline(steps, seconds=seconds, learning_seats=learning_seats)
    destination = out or Path(".runs") / (
        "baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    hardware = HardwareInventory(
        resources=(
            Hardware(
                name="baseline-cpu",
                host=socket.gethostname(),
                cpu_threads=2,
            ),
        )
    )
    try:
        result = run_experiment(plan.experiment(seed), hardware, destination)
    except Exception:
        if (destination / "experiment.json").exists():
            report_baseline(destination)
        raise
    report = report_baseline(destination)
    if report.software != "passed" and result.status == "completed":
        raise RuntimeError(
            f"baseline inspection failed: {report.error}; see {destination}"
        )
    return result
