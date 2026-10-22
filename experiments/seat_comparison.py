"""From-scratch one-seat versus two-seat self-play, matched per learner decision.

Jack Heart's question (MTG-165, 2026-10-22): is score against the heuristic at
N learner decisions the same whether the decisions come from one seat or both?
If so, two-seat at E environment decisions matches one-seat at 2E. Four
training seeds per arm, each run to about 64M learner decisions; the protocol is
experiments/protocols/seat-comparison-v1.md.

`seat_comparison` owns the matched recipe. This module binds it to seeds and
RunPod allocations: one job per (arm, seed), so the eight runs can proceed in
parallel as independent single-job cohorts.
"""

from typing import Literal

from experiments.batched_baseline import seat_comparison
from manabot.remote.plan import JobSpec, Machine
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import PlannedRun
from manabot.training.experiments import Baseline, Case, Experiment, Pipeline

SEEDS = (64001, 64002, 64003, 64004)
# Complete random-policy games run ~180 environment decisions and trained ones
# ~108, so 256 streams yield 27k-46k learner decisions per update: 2,400 updates
# passes 64M in both arms. Matched comparisons interpolate between exports.
UPDATES = 2_400
CHECKPOINT_UPDATES = 20
L4 = Machine(
    gpu_types=("NVIDIA L4",),
    thread_limit=4,
    vcpus=6,
    memory_gb=62,
    container_gb=60,
    volume_gb=1,
    image="runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853",
    hourly_ceiling=0.59,
)


def experiment(
    updates: int = UPDATES,
    seeds: tuple[int, ...] = SEEDS,
    lifetime_hours: float = 11.9,
    checkpoint_updates: int = CHECKPOINT_UPDATES,
) -> Experiment:
    """One job per (arm, seed); evaluation happens off the rental."""
    spec = JobSpec(
        machine=L4,
        lifetime_hours=lifetime_hours,
        spending_limit=round((0.59 + 0.02) * lifetime_hours + 0.01, 2),
        setup_seconds=900,
        checkpoint_seconds=180,
        upload_seconds=300,
        cleanup_seconds=120,
    )
    arms: tuple[Literal["one", "both"], ...] = ("one", "both")
    return Experiment(
        name="seat-comparison-v1",
        baseline=Baseline.capture(
            "seat-comparison-one", seat_comparison("one", updates)
        ),
        cases=(
            Case("one"),
            Case(
                "both",
                (Pipeline(tuple(seat_comparison("both", updates).stages)),),
                label="learn from both seats",
            ),
        ),
        jobs=tuple(
            PlannedRun(
                case=f"seat-comparison-v1-{arm}",
                seed=seed,
                spec=spec,
                monitoring=MonitoringBudget(
                    seconds=600, attempt_seconds=300, include_monitoring=False
                ),
                checkpoint_updates=checkpoint_updates,
            )
            for seed in seeds
            for arm in arms
        ),
    )
