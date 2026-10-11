"""ETU-125 declaration, timing calibration and retained-checkpoint diagnostic.

Experiment/TrainingRegime own execution; this module freezes a bounded screen
from timing only. The external local service owns the one absolute deadline.
No paid compute, foreign execution, policy selection or automatic retry occurs.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import socket
import time

from nbclient import NotebookClient
import nbformat
import torch

from manabot.arena.models import file_sha256
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.comparison_notebook import write_comparison_notebook
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiment_execution import (
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
from manabot.training.experiment_report import RetainedRun
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Baseline, Case, Experiment, Pipeline
from manabot.training.models import TrainingCoordinates, TrainingRegime, TrainSelfPlay
from manabot.training.monitor_evaluation import MonitorProtocol, evaluate_checkpoint
from manabot.training.recovery import attempt_lock
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (12551, 12552, 12553)
ARMS = ("cross-balanced", "mirrors-balanced")
BASE = ROOT / "experiments/regimes/mirror-screen-baseline.json"


def recipe(arm: str, updates: int, allowance: float) -> TrainingRegime:
    value = TrainingRegime.model_validate_json(BASE.read_text())
    value.id = arm
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.matchup_curriculum = arm
    stage.updates = updates // 2
    stage.execution.wall_seconds = (allowance - 60) / 2
    second = deepcopy(stage)
    second.id, second.initial = "endpoint", stage.id
    value.stages = [stage, second]
    value.wall_seconds = allowance
    return TrainingRegime.model_validate(value.model_dump())


def long_recipe(arm: str, allowance: float) -> TrainingRegime:
    """Fresh 10k trajectory with global iteration schedules and fixed milestones."""
    value = recipe(arm, 10000, allowance)
    first = value.stages[0]
    assert isinstance(first, TrainSelfPlay)
    stages: list[TrainSelfPlay] = []
    for name, updates in (
        ("update-2500", 2500),
        ("midpoint", 2500),
        ("endpoint", 5000),
    ):
        stage = first.model_copy(deep=True)
        stage.id = name
        stage.initial = stages[-1].id if stages else None
        stage.updates = updates
        stage.execution.wall_seconds = (allowance - 600) * updates / 10000
        stages.append(stage)
    value.stages = list(stages)
    value.recovery_max_microsteps = 20_000_000
    value.recovery_every_updates = 250
    return TrainingRegime.model_validate(value.model_dump())


def diagnostic(out: Path) -> None:
    """Score the unchanged Mini 2,500-update bytes on new, labeled matrix deals."""
    torch.set_num_threads(1)
    producer = RetainedRun.model_validate_json(
        (out / "baseline/producer.json").read_text()
    )
    path = out / "baseline/update-2500-raw.pt"
    expected = "55d3cfb0420f900c7bfcbc7341ebd3924c71dd48b951bc0b31021c0c59ce1090"
    if file_sha256(path) != expected:
        raise ValueError("frozen Mini checkpoint digest differs")
    artifact = {
        "path": str(path.resolve()),
        "sha256": expected,
        "bytes": path.stat().st_size,
    }
    checkpoint = next(s for s in producer.stages if s.id == "update-2500")
    coordinate = TrainingCoordinates(
        stage_id=checkpoint.id,
        updates=2500,
        training_seconds=checkpoint.cumulative_seconds or 0,
    )
    for opponent, count in (("scripted_greedy", 5), ("random", 2)):
        result = evaluate_checkpoint(
            producer,
            artifact,
            coordinate,
            out / "diagnostic" / opponent,
            protocol=MonitorProtocol(
                purpose="supplemental-matchup-diagnostic",
                include_mirrors=True,
                opponent=opponent,
                deal_seeds=tuple(range(1_912_500_000, 1_912_500_000 + count)),
            ),
            concurrent_activity="one laptop evaluator; Mini training is on a different host",
        )
        if result.status != "completed":
            raise RuntimeError(f"diagnostic failed: {result.error}")
    report(out)


def calibrate(out: Path) -> None:
    torch.set_num_threads(1)
    timings: dict[str, float] = {}
    with VerifyStore(out / "calibration.sqlite") as store:
        for arm in ARMS:
            start = time.time()
            calibration_recipe = recipe(arm, 20, 600)
            calibration_recipe.id = "calibration-" + arm
            run = execute_regime(
                calibration_recipe,
                12550,
                out / "calibration" / arm,
                store,
                checkpoint_seconds=3600,
            )
            if run.status != "completed":
                raise RuntimeError(f"calibration failed: {run.error}")
            timings[arm] = time.time() - start
    atomic_json(
        out / "calibration.json",
        {
            "updates": 20,
            "seconds": timings,
            "source": file_sha256(Path(__file__)),
            "baseline": file_sha256(BASE),
        },
    )


def freeze(out: Path) -> None:
    """Counts depend on elapsed calibration only; outcomes cannot choose a plan."""
    allocation = json.loads((out / "allocation.json").read_text())
    remaining = allocation["deadline_unix"] - time.time() - 600
    calibration = json.loads((out / "calibration.json").read_text())
    diagnostic_result = json.loads(
        (out / "diagnostic/scripted_greedy/monitor.json").read_text()
    )
    random_result = json.loads((out / "diagnostic/random/monitor.json").read_text())
    if any(r["status"] != "completed" for r in (diagnostic_result, random_result)):
        raise ValueError("calibration requires both completed diagnostic anchors")
    attempt_seconds = max(
        600.0,
        max(r["evaluation_seconds"] for r in (diagnostic_result, random_result)) * 2.5,
    )
    # Prefer three paired seeds; admit a two-seed pilot only from timing.
    per_update = max(calibration["seconds"].values()) / 20 * 1.6
    for seed_count, greedy_count, random_count in (
        (3, 10, 4),
        (2, 10, 4),
        (3, 6, 2),
        (2, 6, 2),
    ):
        per_checkpoint = (
            diagnostic_result["evaluation_seconds"] * greedy_count / 10
            + random_result["evaluation_seconds"] * random_count / 4
        )
        evaluation_reserve = max(2400.0, 1.6 * per_checkpoint * 3 * 2 * seed_count)
        allowance = (remaining - evaluation_reserve) / (2 * seed_count)
        updates = min(400, int((allowance - 120) / per_update) // 20 * 20)
        if updates >= 100:
            break
    else:
        raise RuntimeError(
            "two paired seeds at 100 updates cannot fit remaining allocation"
        )
    allowance = min(allowance, updates * per_update + 120)
    plan = {
        "seeds": SEEDS[:seed_count],
        "scope": "three-seed-screen" if seed_count == 3 else "two-seed-pilot",
        "updates": updates,
        "run_allowance_seconds": allowance,
        "evaluation_reserve_seconds": evaluation_reserve,
        "attempt_seconds": attempt_seconds,
        "wall_seconds": remaining,
        "created_unix": time.time(),
        "calibration_sha256": file_sha256(out / "calibration.json"),
        "baseline_sha256": file_sha256(BASE),
        "source_sha256": file_sha256(Path(__file__)),
        "greedy_games_per_cell": greedy_count,
        "random_games_per_cell": random_count,
    }
    target = out / "plan.json"
    with target.open("x") as stream:
        json.dump(plan, stream, indent=2)


def declaration(out: Path) -> Experiment:
    plan = json.loads((out / "plan.json").read_text())
    regimes = [
        long_recipe(arm, plan["run_allowance_seconds"])
        if plan.get("scope") == "two-seed-10k"
        else recipe(arm, plan["updates"], plan["run_allowance_seconds"])
        for arm in ARMS
    ]
    greedy = MonitorProtocol(
        include_mirrors=True,
        deal_seeds=tuple(
            range(
                plan.get("greedy_deal_start", 1_912_510_000),
                plan.get("greedy_deal_start", 1_912_510_000)
                + plan["greedy_games_per_cell"] // 2,
            )
        ),
    )
    random = MonitorProtocol(
        include_mirrors=True,
        opponent="random",
        deal_seeds=tuple(
            range(
                plan.get("random_deal_start", 1_912_520_000),
                plan.get("random_deal_start", 1_912_520_000)
                + plan["random_games_per_cell"] // 2,
            )
        ),
    )
    return Experiment(
        name="etu125-mirrors",
        baseline=Baseline.capture("mirror-screen", regimes[0]),
        cases=(Case(ARMS[0]), Case(ARMS[1], (Pipeline(tuple(regimes[1].stages)),))),
        schedule=ExperimentSchedule(
            seeds=tuple(plan["seeds"]),
            hardware="etu125-laptop",
            wall_seconds=plan["wall_seconds"],
            process_seconds=plan.get("process_seconds", plan["wall_seconds"]),
            checkpoint_seconds=plan.get("checkpoint_seconds", 86400),
            disk_reserve_bytes=plan.get("disk_reserve_bytes", 4 * 1024**3),
            order=((0, 1), (1, 0), (0, 1))[: len(plan["seeds"])],
            monitoring=MonitoringBudget(
                seconds=plan["evaluation_reserve_seconds"],
                attempt_seconds=plan["attempt_seconds"],
                protocol=greedy,
                additional_protocols=(random,),
                include_initial=True,
            ),
        ),
    )


def report(out: Path) -> None:
    path = out / "comparison.ipynb"
    fresh = not path.exists()
    write_comparison_notebook(out, path)
    notebook = nbformat.read(path, as_version=4)
    if fresh:
        source = notebook.cells[1].source
        source = source.replace(
            "QUESTION = 'How do the declared variants progress at shared training milestones?'",
            "QUESTION = 'Does mirror-inclusive self-play improve Lessons learning and cross-deck strength?'",
        )
        source += "\nfrom experiments.runners.mirror_analysis import analyze\nNOTES = analyze(DATA_ROOT) + ' ETU-125 exploratory screen. Initialization is the frozen/no-update control. The Mini diagnostic and calibration are separate from randomized training. Per-cell gains and paired-seed effects are retained in contrasts.json.'"
        notebook.cells[1].source = source
        nbformat.write(notebook, path)
    # Execute the editable generator in memory: refresh HTML without overwriting
    # notebook edits or manufacturing new measurements.
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(out)}},
    ).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("diagnostic", "calibrate", "freeze", "run", "report")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover", type=int, nargs="*", default=[])
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "run":
        inventory = HardwareInventory(
            resources=(
                Hardware(
                    name="etu125-laptop", host=socket.gethostname(), cpu_threads=2
                ),
            )
        )
        run_experiment(
            declaration(out),
            inventory,
            out / "science",
            resume=args.resume,
            recover=tuple(args.recover),
        )
        report(out)
    elif args.phase == "diagnostic":
        with attempt_lock(Path.home() / ".cache/manabot/checkpoint-evaluator.lock"):
            diagnostic(out)
    else:
        {
            "diagnostic": diagnostic,
            "calibrate": calibrate,
            "freeze": freeze,
            "report": report,
        }[args.phase](out)


if __name__ == "__main__":
    main()
