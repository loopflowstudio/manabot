"""Bounded CPU collect/update/export/arena/replay calibration workflow.

The study runner owns execution and attempts; VerifyStore owns TrainingRuns.
CalibrationReport is a derived view of those records, never a resume authority.
One fixed seed and concurrent smoke timings cannot project scientific throughput.
"""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import signal
import time
from typing import Literal

import psutil
from pydantic import BaseModel, Field
import torch

from experiments.runners.model_capacity import regimes as capacity_regimes
from experiments.runners.run_training_regimes import run_study, smoke_recipe
from experiments.runners.scale_probe import scale
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.architecture import ArchitectureReceipt, architecture_receipt
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.clock import watchdog_seconds
from manabot.training.execution import atomic_json
from manabot.training.models import (
    StageRecord,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
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


class InferenceProbe(BaseModel):
    """CPU checkpoint load and fixed native-observation batch; no optimizer work."""

    run_id: str
    checkpoint_sha256: str
    observation_sha256: str
    architecture: ArchitectureReceipt
    initialization_seconds: float
    cold_load_seconds: float
    collection_seconds: float
    first_forward_seconds: float
    warmup_forwards: int
    measured_forwards: int
    batch_size: int
    steady_forward_seconds: float
    sampled_peak_rss_bytes: int


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
    inference: list[InferenceProbe] = Field(default_factory=list)
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
        game_seconds=30,
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=(recipe.model_dump(mode="json"),),
        allocation_seconds=180,
        prior_campaign_seconds=0,
        calibration_evidence="Fixed bounded workflow proof; concurrent host timings are not scientific calibration",
    )


def capacity_plan(device: str = "cpu") -> ResolvedStudy:
    """Reuse the delivered ladder with a fixed tiny workload and shared arena."""
    baseline = calibration_plan(device)
    recipe = TrainingRegime.model_validate(baseline.recipes[0])
    recipe.wall_seconds = 120
    for stage in recipe.stages:
        stage.execution.wall_seconds = 60
    recipes = tuple(capacity_regimes(recipe).values())
    protocol = EvaluationProtocol(
        study="capacity-calibration",
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in recipes
        ),
        process_seconds=780,
        game_seconds=30,
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=tuple(r.model_dump(mode="json") for r in recipes),
        allocation_seconds=900,
        prior_campaign_seconds=0,
        calibration_evidence="Bounded capacity software proof; concurrent timings do not rank hardware or models",
    )


def _inference_probe(run: TrainingRun) -> InferenceProbe:
    """Measure reload/first call separately from repeated batch inference.

    The native collector supplies real viewer-safe tensors. Its cost is a probe
    overhead, not additional training. No action/value sampling occurs inside the
    warmed timing loop. RSS samples are lower bounds, not exact allocator peaks.
    """
    checkpoint = Path(run.stages[-1].artifacts["raw"]["path"])
    torch.set_num_threads(1)
    space = ObservationSpace(run.regime.observation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(run.seed_streams["initialization"])
        tick = time.perf_counter()
        fresh = Agent(space, run.regime.agent)
        initialization = time.perf_counter() - tick
        del fresh
    tick = time.perf_counter()
    agent, space = load_checkpoint_agent(str(checkpoint))
    cold = time.perf_counter() - tick
    peak = psutil.Process().memory_info().rss
    tick = time.perf_counter()
    collector = SeatRoutedCollector(
        space,
        Match(run.regime.match),
        Reward(RewardHypers()),
        num_envs=4,
        seed=run.seed_streams["collection"],
    )
    batch = collector.collect(agent, 1)
    collection = time.perf_counter() - tick
    obs = {key: torch.from_numpy(value[0]) for key, value in batch.obs.items()}
    digest = hashlib.sha256()
    for key, tensor in sorted(obs.items()):
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    # The collector has already called the model. Reload for a first-forward
    # measurement on these same identified tensors; this is not cold OS cache.
    agent, _ = load_checkpoint_agent(str(checkpoint))
    warmup_forwards, measured_forwards = 3, 10
    with torch.inference_mode():
        tick = time.perf_counter()
        agent(obs)
        first = time.perf_counter() - tick
        for _ in range(warmup_forwards):
            agent(obs)
        tick = time.perf_counter()
        for _ in range(measured_forwards):
            agent(obs)
        steady = time.perf_counter() - tick
    peak = max(peak, psutil.Process().memory_info().rss)
    return InferenceProbe(
        run_id=run.id,
        checkpoint_sha256=file_sha256(checkpoint),
        observation_sha256=digest.hexdigest(),
        architecture=architecture_receipt(agent),
        initialization_seconds=initialization,
        cold_load_seconds=cold,
        collection_seconds=collection,
        first_forward_seconds=first,
        warmup_forwards=warmup_forwards,
        measured_forwards=measured_forwards,
        batch_size=collector.num_envs,
        steady_forward_seconds=steady,
        sampled_peak_rss_bytes=peak,
    )


def _recorded_runs(out: Path) -> list[TrainingRun]:
    """Include failed canonical attempts even when their JSON export failed."""
    database = out / "training.sqlite"
    if not database.exists():
        return []
    with VerifyStore(database) as store:
        ids = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid"
        ).fetchall()
        return [store.training_run(run_id) for (run_id,) in ids]


def _collect_records(out: Path, report: CalibrationReport) -> None:
    study_path = out / "study.json"
    if not study_path.exists():
        return
    study = _Study.model_validate_json(study_path.read_text())
    if report.error is None:
        report.status = study.status
    report.runs = [
        CalibrationRun(
            run_id=run.id,
            seed=run.seed,
            status=run.status,
            run_seconds=run.seconds,
            watchdog_seconds=run.watchdog_seconds,
            setup_seconds=run.setup_seconds,
            stages=run.stages,
        )
        for run in _recorded_runs(out)
    ]
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


def calibrate(
    out: Path, device: str = "cpu", *, capacity: bool = False
) -> CalibrationReport:
    """Run once in a new directory; preserve failures and never retry a seed."""
    plan = capacity_plan(device) if capacity else calibration_plan(device)
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
    previous_threads = torch.get_num_threads()
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        run_study(plan.protocol.study, out, plan, render_report=False)
        if capacity:

            def deadline(*_: object) -> None:
                raise TimeoutError(
                    f"capacity calibration exceeded {plan.allocation_seconds:g} seconds"
                )

            remaining = plan.allocation_seconds - (time.perf_counter() - start)
            if remaining <= 0:
                deadline()
            signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, remaining)
            for run in _recorded_runs(out):
                if run.status != "completed":
                    raise ValueError("capacity probe requires completed training")
                report.inference.append(_inference_probe(run))
                atomic_json(out / "calibration.json", report.model_dump(mode="json"))
        report.status = "completed"
    except BaseException as error:
        report.status = "failed"
        report.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        torch.set_num_threads(previous_threads)
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
    parser.add_argument(
        "--capacity",
        action="store_true",
        help="Measure the three delivered capacities within 900 seconds",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Measure four sizes on CPU/MPS and ordinary CPU training within 900 seconds",
    )
    args = parser.parse_args()
    if args.scale:
        if args.capacity or args.device != "cpu":
            parser.error("--scale cannot be combined with --capacity or --device")
        scale(args.out)
        return
    calibrate(args.out, args.device, capacity=args.capacity)


if __name__ == "__main__":
    main()
