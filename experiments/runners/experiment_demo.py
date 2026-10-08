"""Tiny editable comparison demo: four short training runs and four milestones.

This is a software workflow fixture, not a strength experiment. It uses the same
public declaration and explicit runner as the history/depth comparisons.
"""

import argparse
from pathlib import Path

from manabot.infra.hypers import AgentSpec
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import ExperimentSchedule, HardwareInventory
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Baseline, Case, Experiment, Model
from manabot.training.monitor_evaluation import MonitorProtocol
from manabot.training.presets import ataraxos_mtg_v1


def declaration() -> Experiment:
    base = ataraxos_mtg_v1().regime()
    base.stages = base.stages[:1]
    base.wall_seconds = 20
    for stage in base.stages:
        stage.execution.wall_seconds = 8
        stage.streams = 2
        stage.transitions = 8
    base.agent.hidden_dim = 8
    base.agent.num_attention_heads = 2
    return Experiment(
        name="fixture",
        baseline=Baseline.capture("fixture", base),
        cases=(Case("small"), Case("wide", (Model(AgentSpec(hidden_dim=16)),))),
        schedule=ExperimentSchedule(
            seeds=(117, 118),
            hardware="fixture",
            wall_seconds=750,
            process_seconds=750,
            monitoring=MonitoringBudget(
                seconds=650,
                attempt_seconds=140,
                protocol=MonitorProtocol(
                    deal_seeds=(1910101000,), game_seconds=30, bootstrap_replicates=20
                ),
            ),
            checkpoint_updates=1000,
            disk_reserve_bytes=0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        required=True,
        help="Configured local resource named fixture",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        declaration(),
        HardwareInventory.model_validate_json(args.hardware.read_text()),
        args.out,
        resume=args.resume,
    )
    print(f"{result.status}: {result.notebook}")
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
