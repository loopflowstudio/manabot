"""Thin CLI admission shared by the history and depth declarations.

Frozen plan admission stays with each existing screen specification. Experiment
owns new training and monitoring execution, not historical scientific re-scoring.
"""

import argparse
from pathlib import Path
from typing import Callable

from experiments.runners.run_history_input import verify_runtime
from experiments.runners.screen_spec import specification
from experiments.runners.training_protocol import ResolvedStudy
from manabot.training.experiment_execution import ExperimentSchedule, HardwareInventory
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Experiment


def admit_screen(plan: ResolvedStudy, experiment: Experiment) -> None:
    spec = specification(plan.protocol.study)
    schedule = experiment.schedule
    if schedule is None:
        raise ValueError("screen requires an explicit Experiment schedule")
    if experiment.resolve().digests != plan.protocol.regime_digests:
        raise ValueError("declaration differs from frozen scientific recipes")
    if schedule.seeds != spec.SEEDS or schedule.order != spec.ORDER:
        raise ValueError("screen seed/order differs from frozen plan")
    if set(schedule.scientific_deal_seeds) != set(spec.DEALS):
        raise ValueError("screen must reserve all scientific deals")
    if max(schedule.wall_seconds, schedule.process_seconds) > spec.TRAINING_SECONDS:
        raise ValueError(
            "training plus monitoring must fit the original training envelope"
        )


def screen_main(
    factory: Callable[[ResolvedStudy, ExperimentSchedule], Experiment],
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = ResolvedStudy.model_validate_json(args.plan.read_text())
    schedule = ExperimentSchedule.model_validate_json(args.schedule.read_text())
    experiment = factory(plan, schedule)
    admit_screen(plan, experiment)
    verify_runtime(plan)
    result = run_experiment(
        experiment,
        HardwareInventory.model_validate_json(args.hardware.read_text()),
        args.out,
        resume=args.resume,
    )
    print(f"{result.status}: {result.notebook}")
    if result.status != "completed":
        raise SystemExit(1)
