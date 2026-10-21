"""CUDA companion to the corrected Mini baseline, through public Etude jobs.

Calibration measures the unchanged 256-transition learner and real CPU arena.
Scientific jobs require its frozen cost receipt; no final cohort is scored here.
Etude owns rental deadlines, publication and deletion; Mini owns final scoring.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time

from etude.track.records import Step
from etude.track.run import init
import torch

from experiments.runners.baseline_monitoring import DiagnosticMonitor
from experiments.runners.corrected_baseline import MATCHUP, recipe
from manabot.training.etude_evaluation import publish_saved_evaluation
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRegime
from manabot.training.monitor_evaluation import (
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore

MILESTONES = (1000, 2000, 3000, 4000, 5000, *range(7500, 25001, 2500))
SEEDS = (12261, 12262, 12263)


def gpu_recipe(updates: int, seconds: float) -> TrainingRegime:
    result = recipe(updates, seconds)
    result.stages[0].execution.device = "cuda"
    return TrainingRegime.model_validate(result.model_dump())


def run() -> None:
    if not os.environ.get("ETUDE_RUN") or not torch.cuda.is_available():
        raise ValueError("GPU baseline requires Etude and CUDA; no CPU fallback")
    config = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
    calibration = config.get("mode") == "calibration"
    seed = int(os.environ["ETUDE_SEED"])
    if seed not in ((12260,) if calibration else SEEDS):
        raise ValueError("unregistered baseline seed")
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    if calibration:
        updates, seconds = 100, 1200.0
    else:
        admission = config["calibration"]
        if not admission["admitted"]:
            raise ValueError("GPU timing did not admit the fixed study")
        updates, seconds = 25000, float(admission["learner_seconds"])
        if (
            deadline - time.time()
            < seconds + float(admission["diagnostic_seconds"]) + 600
        ):
            raise ValueError("remaining rental cannot admit the full fixed target")
    monitor = (
        None
        if calibration
        else DiagnosticMonitor(
            output / "training/run.json",
            output / "diagnostics",
            updates=MILESTONES,
            seconds=float(config["calibration"]["diagnostic_seconds"]),
        )
    )
    if monitor is not None:
        monitor.start()
    try:
        with VerifyStore(output / "training.sqlite") as store:
            trained = execute_regime(
                gpu_recipe(updates, seconds),
                seed,
                output / "training",
                store,
                checkpoint_updates=50 if calibration else 500,
                stop_path=Path(os.environ["ETUDE_STOP_FILE"]),
            )
    finally:
        if monitor is not None:
            monitor.finish()
    if trained.status != "completed":
        raise RuntimeError(f"GPU fit failed: {trained.error}")
    if calibration:
        point = stage_checkpoint(trained, trained.stages[0].id)
        if point is None:
            raise ValueError("calibration checkpoint unavailable")
        costs: list[float] = []
        for anchor in ("scripted_greedy", "random"):
            result = evaluate_checkpoint(
                trained,
                point.artifact,
                point.coordinates,
                output / anchor,
                protocol=MonitorProtocol(
                    authored_matchup=MATCHUP,
                    include_mirrors=True,
                    deal_seeds=(1226000100,),
                    opponent=anchor,
                    bootstrap_replicates=100,
                ),
            )
            result.validate_cohort()
            if result.status != "completed" or len(result.rows) != 8:
                raise ValueError("incomplete calibration arena")
            costs.append(float(result.evaluation_seconds) / 8)
        clocks = [0.0] + [
            float(d["coordinates"]["active_training_seconds"])
            for d in trained.stages[0].diagnostics
        ]
        slowest = max((clocks[i] - clocks[i - 20]) / 20 for i in range(20, 101, 20))
        learner = 25000 * 1.5 * slowest + 120
        diagnostic = max(1800.0, len(MILESTONES) * 104 * sum(costs) * 1.5)
        atomic_json(
            output / "calibration.json",
            {
                "admitted": learner + diagnostic + 600 <= 8.5 * 3600,
                "learner_seconds": learner,
                "diagnostic_seconds": diagnostic,
                "slowest_seconds_per_update": slowest,
                "anchor_seconds_per_game": costs,
                "updates": 25000,
                "seeds": SEEDS,
                "milestones": MILESTONES,
                "calibration_run": trained.id,
            },
        )
    else:
        tracker = init(
            experiment="mtg122-gpu-evaluation",
            clocks=[Step("evaluations", "Diagnostic cohorts")],
            run=os.environ["ETUDE_RUN"] + "-diagnostics",
            directory=Path(os.environ["ETUDE_TRACK_DIR"]) / "diagnostics",
            strict=True,
        )
        try:
            paths = sorted(
                (output / "diagnostics").glob("attempt-*/evaluation/monitor.json")
            )
            for path in paths:
                publish_saved_evaluation(tracker, path)
            tracker.log(evaluations=len(paths))
        finally:
            tracker.finish()
    atomic_json(output / "etude-result.json", {"clocks": {"completed_runs": 1}})


if __name__ == "__main__":
    run()
