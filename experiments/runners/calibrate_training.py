"""Bounded CPU collect/update/export/arena/replay calibration workflow.

The study runner owns execution and attempts; VerifyStore owns TrainingRuns.
CalibrationReport is a derived view of those records, never a resume authority.
One fixed seed and concurrent smoke timings cannot project scientific throughput.
"""

import argparse
import os
from pathlib import Path
import platform
import time
from typing import Literal

import psutil
from pydantic import BaseModel, Field
import torch

from experiments.runners.run_training_regimes import run_study, smoke_recipe
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256
from manabot.training.clock import watchdog_seconds
from manabot.training.execution import atomic_json
from manabot.training.models import StageRecord, TrainingRun, TrainSelfPlay
from manabot.verify.store import VerifyStore


class HostSample(BaseModel):
    """Endpoint RSS/load observations; neither is a peak or attribution model."""

    load_average: tuple[float, float, float]
    process_rss_bytes: int
    available_memory_bytes: int
    torch_threads: int


def _host_sample() -> HostSample:
    return HostSample(
        load_average=os.getloadavg(),
        process_rss_bytes=psutil.Process().memory_info().rss,
        available_memory_bytes=psutil.virtual_memory().available,
        torch_threads=torch.get_num_threads(),
    )


class _Replay(BaseModel):
    passed: bool
    seconds: float | None = None


class _Game(BaseModel):
    failure: str | None
    terminated: bool
    truncated: bool
    replay_passed: bool


class _Cell(BaseModel):
    evaluation_seconds: float
    scheduled_games: int
    replay: _Replay
    rows: list[_Game]


class _Study(BaseModel):
    status: str
    comparisons: list[_Cell]


class CalibrationRun(BaseModel):
    """Stage counters/costs are copied without reinterpreting their denominators."""

    run_id: str
    seed: int
    status: str
    run_seconds: float
    watchdog_seconds: float
    setup_seconds: float
    stages: list[StageRecord]


class CalibrationReport(BaseModel):
    status: str
    error: str | None = None
    platform: str = Field(default_factory=platform.platform)
    device: Literal["cpu"] = "cpu"
    evaluation_threads: Literal[1] = 1
    elapsed_seconds: float = 0
    sleep_inclusive_seconds: float = 0
    process_cpu_seconds: float = 0
    host_before: HostSample
    host_after: HostSample | None = None
    runs: list[CalibrationRun] = Field(default_factory=list)
    evaluation_including_replay_seconds: float = 0
    replay_seconds: float | None = 0
    scheduled_evaluation_games: int = 0
    complete_replayed_games: int = 0
    limits: tuple[str, ...] = (
        "Workflow smoke only; no hardware ranking, strength claim or cohort projection.",
        "CPU only: TrainingRegime and study inference do not support MPS/CUDA here.",
        "Training native microsteps are StageRecord.environment_decisions; forced internal engine ticks are not counted.",
        "Learner transitions and optimizer sample exposures are distinct; exposures count repeated minibatch samples, not optimizer steps.",
        "Collection includes inference; device transfer time, cloning and hidden-world sampling are not separately measured.",
        "Arena time includes worker startup, play, trace export and replay; replay_seconds is a subset, not an added cost.",
        "Stage RSS is sampled process-tree RSS; endpoint RSS is parent-only. Neither measures exact peaks or GPU memory.",
        "Host load cannot identify contention or thermal throttling; both remain uncontrolled and unmeasured.",
        "Phase perf_counter clocks may exclude sleep; outer continuous clock includes sleep. Process CPU time excludes arena children.",
        "The study uses its existing signal deadline; this is not a new cross-phase sleep-inclusive watchdog or training resume contract.",
        "Checkpoint cumulative_seconds records observed training availability; comparison is unavailable outside overlapping observed ranges.",
        "The 168-hour campaign cap is unchanged; this smoke allocates no scientific or paid compute.",
    )


def calibration_plan(device: str = "cpu") -> ResolvedStudy:
    """Fix all work before execution; reject devices without an integrated path."""
    if device != "cpu":
        raise ValueError(
            "complete-loop calibration supports CPU only; MPS/CUDA unsupported"
        )
    recipe = smoke_recipe("direct-self-play")
    recipe.id = "full-loop-calibration"
    recipe.wall_seconds = 60
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.execution.wall_seconds = 30
        stage.execution.threads = 1
        stage.streams = 4
        stage.transitions = 64
        stage.updates = 1
    protocol = EvaluationProtocol(
        study="training-calibration",
        regime_digests=(canonical_sha256(recipe.model_dump(mode="json")),),
        process_seconds=180,
        game_seconds=10,
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=(recipe.model_dump(mode="json"),),
        allocation_seconds=180,
        prior_campaign_seconds=0,
        calibration_evidence="Fixed bounded workflow proof; concurrent host timings are not scientific calibration",
    )


def _collect_records(out: Path, report: CalibrationReport) -> None:
    study_path = out / "study.json"
    if not study_path.exists():
        return
    study = _Study.model_validate_json(study_path.read_text())
    report.status = study.status
    database = out / "training.sqlite"
    if database.exists():
        with VerifyStore(database) as store:
            # Include failed canonical attempts even when their JSON export failed.
            ids = store.con.execute(
                "SELECT id FROM training_runs ORDER BY rowid"
            ).fetchall()
            for (run_id,) in ids:
                run: TrainingRun = store.training_run(run_id)
                report.runs.append(
                    CalibrationRun(
                        run_id=run.id,
                        seed=run.seed,
                        status=run.status,
                        run_seconds=run.seconds,
                        watchdog_seconds=run.watchdog_seconds,
                        setup_seconds=run.setup_seconds,
                        stages=run.stages,
                    )
                )
    for cell in study.comparisons:
        report.evaluation_including_replay_seconds += cell.evaluation_seconds
        if report.replay_seconds is not None:
            report.replay_seconds = (
                report.replay_seconds + cell.replay.seconds
                if cell.replay.seconds is not None
                else None
            )
        report.scheduled_evaluation_games += cell.scheduled_games
        report.complete_replayed_games += sum(
            row.failure is None
            and row.terminated
            and not row.truncated
            and row.replay_passed
            for row in cell.rows
        )


def calibrate(out: Path, device: str = "cpu") -> CalibrationReport:
    """Run once in a new directory; preserve failures and never retry a seed."""
    plan = calibration_plan(device)
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"calibration output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = CalibrationReport(status="running", host_before=_host_sample())
    start, continuous, cpu = (
        time.perf_counter(),
        watchdog_seconds(),
        time.process_time(),
    )
    try:
        run_study("training-calibration", out, plan, render_report=False)
        report.status = "completed"
    except BaseException as error:
        report.status = "failed"
        report.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.elapsed_seconds = time.perf_counter() - start
        report.sleep_inclusive_seconds = watchdog_seconds() - continuous
        report.process_cpu_seconds = time.process_time() - cpu
        report.host_after = _host_sample()
        if out.exists():
            _collect_records(out, report)
            atomic_json(out / "calibration.json", report.model_dump(mode="json"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    calibrate(args.out, args.device)


if __name__ == "__main__":
    main()
