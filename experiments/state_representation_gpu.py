"""Sequential GPU admission and prospectively budgeted representation learning.

The same complete-game executor owns both excluded throughput admission and
eight independent scientific learners. A frozen allocation binds measured
throughput and a rental quote before scientific execution; one child owns the
GPU at a time. Monitoring exports and evaluation retain every scheduled point.
"""

import argparse
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Literal

from etude.launch.records import Job, Run
from etude.report import Report
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from experiments.batched_baseline import batched_baseline
from experiments.state_representation_v7 import (
    ARMS,
    Arm,
    ArmEvaluation,
    PairedEstimate,
    compare_cohort,
    paired_interval,
)
from manabot.env.observation import ObservationConfig
from manabot.eval.monitor import MonitorProtocol, MonitorResult
from manabot.experiment.checkpoint_evaluation import (
    Checkpoint,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.experiment.numerics import check_numerics
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.recipes.builders import move_learning
from manabot.sim.execution import EvalExecution
from manabot.training.execution import execute_regime
from manabot.training.panels import ML, PERF
from manabot.training.records import TrainingCoordinates, TrainingRun
from manabot.training.regime import IterationSchedule, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

EXPERIMENT = "state_representation_gpu"
UV_EXTRAS = " --extra dev --extra artifacts --extra evaluation"
SEEDS = (74211, 74212, 74213, 74214)
STREAMS = 256
UPDATES = 256
CADENCE = 32
POINTS = tuple(range(0, UPDATES + 1, CADENCE))
ADMISSION_UPDATES = 4
ADMISSION_SEED = 74200
# Four ordinary updates must fit alongside the executor's finalization reserve.
# The outer child deadline also leaves time for export and process shutdown.
ADMISSION_ARM_SECONDS = 1800
ADMISSION_WORK_HOURS = 1.25
SETUP_SECONDS = 900
EVALUATION_SECONDS = 5400
TEARDOWN_SECONDS = 120
RENTAL_OVERHEAD_SECONDS = SETUP_SECONDS + EVALUATION_SECONDS + TEARDOWN_SECONDS
CURVE_DEALS = tuple(range(85001, 85033))
HELDOUT_DEALS = tuple(range(86001, 86065))
SCORING = {
    "name": "state-representation-gpu-batched-5da5f4",
    "seeds": SEEDS,
    "arms": ARMS,
    "points": POINTS,
    "curve_deals": CURVE_DEALS,
    "heldout_deals": HELDOUT_DEALS,
    "plateau": "last three successive 32-update gains: each arm's paired-seed 95% t interval entirely within +/-0.05; fixed final window",
    "primary": "GRU minus pooled at common final update 256 on untouched held-out deals",
    "failure": "missing seeds/checkpoints/games prevent a qualified result; no retries or selection",
}


def recipe(arm: Arm) -> TrainingRegime:
    """Fresh CUDA learning with schedules indexed by equivalent game exposure."""
    if arm not in ARMS:
        raise ValueError("unknown study arm")
    regime = batched_baseline()
    regime.id = f"state-representation-gpu-batched-{arm}"
    regime.agent.state_encoder = arm
    regime.agent.policy_film = True
    regime.agent.recent_events = False
    regime.observation = ObservationConfig(
        policy_input_version=7,
        max_facts=128,
        max_cards_per_player=60,
        max_permanents_per_player=40,
        max_actions=512,
        max_focus_objects=2,
        max_events=32,
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "both"
    stage.games_per_stream = 1
    stage.streams = STREAMS
    stage.updates = UPDATES
    # Keep schedule progress aligned to complete-game exposure. Larger learner
    # minibatches still change optimization; this is a prospective recipe.
    stage.learning = move_learning(
        IterationSchedule(steps=UPDATES, reference_steps=2048)
    )
    stage.execution.device = "cuda"
    stage.execution.memory_bytes = 24 * 1024**3
    stage.execution.wall_seconds = 66000
    regime.wall_seconds = 66000
    return TrainingRegime.model_validate(regime.model_dump())


def excluded_recipe(
    arm: Arm,
    regime_id: str,
    updates: int,
    seconds: int,
    device: Literal["cpu", "cuda"] = "cuda",
) -> TrainingRegime:
    """The study recipe under a short excluded horizon; never a scored learner."""
    regime = recipe(arm)
    regime.id = regime_id
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    stage.execution.device = device
    stage.execution.wall_seconds = regime.wall_seconds = seconds
    return TrainingRegime.model_validate(regime.model_dump())


def recipe_digests() -> dict[Arm, str]:
    return {arm: canonical_sha256(recipe(arm).model_dump(mode="json")) for arm in ARMS}


def evaluation_protocol(*, heldout: bool = False) -> MonitorProtocol:
    return MonitorProtocol(
        purpose="frozen-study-evaluation",
        study_protocol_sha256=canonical_sha256(SCORING),
        deal_seeds=HELDOUT_DEALS if heldout else CURVE_DEALS,
        opponent="heuristic",
        game_seconds=120,
        max_commands=10_000,
        bootstrap_seed=85000,
        bootstrap_replicates=2000,
    )


class ArmThroughput(BaseModel):
    """Measured costs for one ordinary learner, in host wall seconds.

    Collection, learning and unclassified in-stage bookkeeping recur per update.
    Child imports/setup/close and initial/final exports occur once per learner.
    Admission's entire export bundle bounds each later raw checkpoint export;
    this is conservative, not a measurement of that checkpoint's exact cost.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    arm: Arm
    updates: int = Field(ge=1)
    collection_seconds: float = Field(ge=0)
    learning_seconds: float = Field(ge=0)
    bookkeeping_seconds: float = Field(ge=0)
    fixed_seconds: float = Field(ge=0)
    export_seconds: float = Field(ge=0)

    @property
    def recurring_seconds(self) -> float:
        return (
            self.collection_seconds + self.learning_seconds + self.bookkeeping_seconds
        )

    @property
    def elapsed_seconds(self) -> float:
        return self.recurring_seconds + self.fixed_seconds + self.export_seconds

    @classmethod
    def from_run(cls, arm: Arm, run: TrainingRun, elapsed: float) -> "ArmThroughput":
        if len(run.stages) != 1:
            raise ValueError("throughput needs one training stage")
        stage = run.stages[0]
        if stage.actual_device != "cuda:0" or stage.actual_threads != 1:
            raise ValueError("throughput requires the declared CUDA/thread placement")
        if stage.updates < 1 or stage.games != stage.updates * STREAMS:
            raise ValueError("throughput needs complete ordinary game batches")
        stage_seconds = (
            max(0, run.seconds - run.setup_seconds)
            if stage.status == "running"
            else stage.seconds
        )
        if elapsed < stage_seconds:
            raise ValueError("child elapsed time cannot be shorter than its stage")
        active = stage.collection_seconds + stage.learning_seconds
        if active <= 0 or active + stage.export_seconds > stage_seconds + 0.01:
            raise ValueError("inconsistent stage timings")
        return cls(
            arm=arm,
            updates=stage.updates,
            collection_seconds=stage.collection_seconds,
            learning_seconds=stage.learning_seconds,
            bookkeeping_seconds=max(0, stage_seconds - active - stage.export_seconds),
            fixed_seconds=elapsed - stage_seconds,
            export_seconds=stage.export_seconds,
        )

    def cohort_seconds(self) -> float:
        recurring = self.recurring_seconds * UPDATES / self.updates
        # Seven intermediate raw exports; initial/final exports count once.
        checkpoints = len(POINTS[1:-1]) * self.export_seconds
        return len(SEEDS) * (
            1.5 * recurring + self.fixed_seconds + self.export_seconds + checkpoints
        )


class ThroughputAdmission(BaseModel):
    """Both arms' phase costs; fixed costs never scale with the update horizon."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    concurrency: Literal[1]
    elapsed_seconds: float = Field(gt=0)
    arms: tuple[ArmThroughput, ArmThroughput]

    @model_validator(mode="after")
    def complete(self) -> "ThroughputAdmission":
        if {item.arm for item in self.arms} != set(ARMS):
            raise ValueError("throughput requires both distinct arms")
        if any(item.updates != ADMISSION_UPDATES for item in self.arms):
            raise ValueError("throughput requires every admission update")
        measured = sum(arm.elapsed_seconds for arm in self.arms)
        if measured > self.elapsed_seconds + 0.01:
            raise ValueError("arm costs exceed admission elapsed time")
        return self

    def cohort_hours(self) -> float:
        measured = sum(arm.elapsed_seconds for arm in self.arms)
        orchestration = max(0, self.elapsed_seconds - measured) * len(SEEDS)
        return (
            sum(arm.cohort_seconds() for arm in self.arms)
            + orchestration
            + RENTAL_OVERHEAD_SECONDS
        ) / 3600

    def projected_usd(self, hourly_usd: Decimal) -> Decimal:
        if not hourly_usd.is_finite() or hourly_usd <= 0:
            raise ValueError("hourly price must be finite and positive")
        return Decimal(str(self.cohort_hours())) * hourly_usd


class AdmissionLimits(BaseModel):
    """Prospective execution policy, independent of model scores.

    The quote ceiling and remaining scientific allowance determine the affordable
    mean update time. Four times that pace permits variation within one complete
    game batch. Two completed updates are required before a phase-cost stop;
    startup and no-completed-update stops retain distinct reasons.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scientific_usd: Decimal = Field(default=Decimal("16"), gt=0, le=16)
    hourly_usd: Decimal = Field(default=Decimal("0.26"), gt=0)
    startup_seconds: float = Field(default=180, gt=0)
    grace_seconds: float = Field(default=120, gt=0)

    @property
    def progress_seconds(self) -> float:
        seconds = float(self.scientific_usd / self.hourly_usd) * 3600
        per_update = max(0, seconds - RENTAL_OVERHEAD_SECONDS) / (
            len(SEEDS) * len(ARMS) * UPDATES
        )
        return max(300, 4 * per_update)


@dataclass
class ProgressWatch:
    """Observe durable update boundaries using an injected monotonic clock."""

    arm: Arm
    path: Path
    # Touching this file asks the child to flush its evidence and exit.
    stop: Path
    limits: AdmissionLimits
    started: float
    # None means no ready CUDA record has been observed yet.
    last_progress: float | None = None
    updates: int = 0

    def observe(self, run: TrainingRun | None, now: float) -> str | None:
        if run is not None and run.stages:
            stage = run.stages[0]
            if (stage.actual_device is not None or stage.updates > 0) and (
                stage.actual_device != "cuda:0" or stage.actual_threads != 1
            ):
                return "training differs from declared CUDA/thread placement"
            if self.last_progress is None and stage.actual_device is not None:
                self.last_progress = now
            if stage.updates < self.updates:
                raise ValueError("training progress regressed")
            if stage.updates > self.updates:
                self.updates = stage.updates
                self.last_progress = now
                if stage.updates >= 2:
                    costs = ArmThroughput.from_run(
                        self.arm, run, max(now - self.started, stage.seconds)
                    )
                    # Only this arm, with the other arm optimistically free. This
                    # is a predeclared capacity policy, not a bad-host diagnosis.
                    projected = (
                        costs.cohort_seconds() + RENTAL_OVERHEAD_SECONDS
                    ) / 3600
                    if (
                        Decimal(str(projected)) * self.limits.hourly_usd
                        > self.limits.scientific_usd
                    ):
                        return "measured arm exceeds scientific allocation"
        if self.last_progress is None:
            if now - self.started >= self.limits.startup_seconds:
                return "startup produced no ready training record"
        elif now - self.last_progress >= self.limits.progress_seconds:
            return "no completed update within workload allowance"
        return None


class ScientificAllocation(BaseModel):
    """Measured full-cohort allocation; setup and teardown count against the cap.

    The retained admission digest identifies evidence, not a model score. The
    operator checks the matching hardware/source and remaining Task allowance
    before freezing this file. Etude separately enforces the live rental quote.
    """

    model_config = ConfigDict(extra="forbid")
    admission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission: ThroughputAdmission
    hourly_usd: Decimal = Field(gt=0, allow_inf_nan=False)
    max_usd: Decimal = Field(gt=0, le=17, allow_inf_nan=False)
    hardware: str = Field(min_length=1)

    @property
    def work_hours(self) -> float:
        # The projection includes 15 minutes setup; Etude adds that separately.
        return self.admission.cohort_hours() - SETUP_SECONDS / 3600

    @model_validator(mode="after")
    def affordable(self) -> "ScientificAllocation":
        if self.admission.projected_usd(self.hourly_usd) > self.max_usd:
            raise ValueError("projected complete cohort exceeds allocation")
        return self


class StudyJob(BaseModel):
    """Source identity and, only after admission, the complete scientific budget."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle
    allocation: ScientificAllocation | None = None
    admission_limits: AdmissionLimits = Field(default_factory=AdmissionLimits)


# Job declaration and independent learner/evaluator entry points.


def runs(revision: str, allocation: str = "", admission_limits: str = "") -> list[Run]:
    """Default to admission; science requires an explicit frozen allocation file."""
    root = Path(__file__).resolve().parents[1]
    admitted = (
        ScientificAllocation.model_validate_json(Path(allocation).read_bytes())
        if allocation
        else None
    )
    limits = (
        AdmissionLimits.model_validate_json(Path(admission_limits).read_bytes())
        if admission_limits
        else AdmissionLimits()
    )
    spec = StudyJob(
        source_bundle=bundle_at(root, revision),
        allocation=admitted,
        admission_limits=limits,
    )
    job = Job(
        name=EXPERIMENT,
        experiment=EXPERIMENT,
        setup=machine_setup(UV_EXTRAS),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked{UV_EXTRAS} python -m experiments.{EXPERIMENT}",
        repo=str(root),
        revision=revision,
        config=spec.model_dump_json(),
        until=f"updates={UPDATES if admitted else ADMISSION_UPDATES}",
        max_hours=admitted.work_hours if admitted else ADMISSION_WORK_HOURS,
        max_usd=admitted.max_usd if admitted else Decimal("0.50"),
        max_attempts=1,
    )
    return job.runs(0)


def checkpoints(run: TrainingRun) -> list[Checkpoint]:
    """Admit exactly initialization, all curve exports and the completed endpoint."""
    if run.status != "completed" or run.updates_through() != UPDATES:
        raise ValueError("curve requires completed training")
    stage = run.stages[0]
    points = [
        Checkpoint(
            artifact=stage.artifacts["initial_raw"],
            coordinates=TrainingCoordinates(
                stage_id=stage.id, updates=0, training_seconds=0
            ),
        )
    ]
    for saved in run.monitoring_checkpoints:
        if saved.error is not None or saved.artifact is None:
            raise ValueError("declared checkpoint export failed")
        points.append(
            Checkpoint(
                artifact=saved.artifact,
                coordinates=TrainingCoordinates.model_validate(
                    saved.model_dump(include=set(TrainingCoordinates.model_fields))
                ),
            )
        )
    final = stage_checkpoint(run, stage.id)
    if final is not None:
        points.append(final)
    points.sort(key=lambda point: point.coordinates.updates)
    if tuple(point.coordinates.updates for point in points) != POINTS:
        raise ValueError("missing, duplicate or undeclared curve checkpoint")
    return points


def train_arm(
    arm: Arm, seed: int, out: Path, stop: Path, *, admission: bool = False
) -> None:
    """One fresh learner owns CUDA, execution and receipts in either mode.

    Admission uses an excluded seed and short horizon without curve milestones;
    science requires every declared export before evaluation can begin.
    """
    if seed not in ((ADMISSION_SEED,) if admission else SEEDS):
        raise ValueError("undeclared training seed")
    if not torch.cuda.is_available():
        raise RuntimeError("this study requires CUDA")
    # One learner owns the device; leave room for contexts and workspaces.
    torch.cuda.set_per_process_memory_fraction(0.80)
    regime = (
        excluded_recipe(
            arm,
            f"representation-gpu-admission-{arm}",
            ADMISSION_UPDATES,
            ADMISSION_ARM_SECONDS,
        )
        if admission
        else recipe(arm)
    )
    out.mkdir(parents=True, exist_ok=False)
    with VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(
            regime,
            seed,
            out / "training",
            store,
            experiment=f"{EXPERIMENT}_admission" if admission else EXPERIMENT,
            checkpoint_updates=() if admission else POINTS[1:-1],
            stop_path=stop,
            stop_reserve_seconds=120,
        )
    atomic_json(
        out / "device.json",
        {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )
    if admission:
        if run.status != "completed" or run.updates_through() != ADMISSION_UPDATES:
            raise RuntimeError(f"incomplete admission: {run.status}: {run.error}")
    else:
        checkpoints(run)


def evaluate_arm(arm: Arm, seed: int, out: Path, deadline: float) -> None:
    """Score all frozen curve points and a separate final held-out cohort."""
    run = TrainingRun.model_validate_json((out / "training/run.json").read_bytes())
    if (
        seed not in SEEDS
        or run.seed != seed
        or run.regime_digest != recipe_digests()[arm]
    ):
        raise ValueError("run is outside the declared GPU cohort")
    for point in checkpoints(run):
        update = point.coordinates.updates
        for heldout in (False, True) if update == UPDATES else (False,):
            if time.time() >= deadline:
                raise RuntimeError("evaluation allocation expired")
            result = evaluate_checkpoint(
                run,
                point.artifact,
                point.coordinates,
                out / ("heldout" if heldout else f"curve-{update:04d}"),
                protocol=evaluation_protocol(heldout=heldout),
                concurrent_activity="one sequential evaluator; all study training finished",
                execution=EvalExecution(
                    device="cpu",
                    max_active_games=4,
                    max_batch_rows=32,
                    resident_model_bytes=1024**3,
                    deadline_unix=deadline,
                ),
            )
            if result.status != "completed":
                raise RuntimeError(
                    "incomplete evaluation retained in study denominator"
                )


def compare_heldout(results: list[ArmEvaluation]) -> PairedEstimate:
    return compare_cohort(
        results, SEEDS, UPDATES, evaluation_protocol(heldout=True), recipe_digests()
    )


# Sequential execution and child lifetime ownership.

_CHILD = (sys.executable, "-m", f"experiments.{EXPERIMENT}")


def scientific_cohort(out: Path, stop: Path, deadline: float) -> None:
    """Sequential balanced arm order; stop and retain the denominator on failure."""
    assignments = [
        (arm, seed)
        for index, seed in enumerate(SEEDS)
        for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
    ]
    atomic_json(
        out / "cohort.json",
        {
            "scoring": SCORING,
            "assignments": assignments,
            "recipes": {arm: recipe(arm).model_dump(mode="json") for arm in ARMS},
            "concurrency": 1,
        },
    )
    for mode in ("train", "evaluate"):
        phase = out / mode
        phase.mkdir()
        for arm, seed in assignments:
            logs = phase / f"{arm}-{seed}"
            logs.mkdir()
            command = [
                *_CHILD,
                f"--{mode}-arm",
                arm,
                "--seed",
                str(seed),
                "--out",
                str(out / f"{arm}-{seed}"),
            ]
            bound = deadline - (
                EVALUATION_SECONDS if mode == "train" else TEARDOWN_SECONDS
            )
            command += (
                ["--stop", str(stop)] if mode == "train" else ["--deadline", str(bound)]
            )
            print(f"{mode}: {arm} seed {seed}", flush=True)
            _run_child(command, logs, stop, bound)

    def retained(arm: Arm, path: Path) -> ArmEvaluation:
        return ArmEvaluation(
            arm=arm, result=MonitorResult.model_validate_json(path.read_bytes())
        )

    curves = [
        retained(arm, out / f"{arm}-{seed}/curve-{update:04d}/monitor.json")
        for arm, seed in assignments
        for update in POINTS
    ]
    final = [
        retained(arm, out / f"{arm}-{seed}/heldout/monitor.json")
        for arm, seed in assignments
    ]
    atomic_json(
        out / "comparison.json",
        {
            "plateau": asdict(plateau(curves)),
            "heldout": asdict(compare_heldout(final)),
        },
    )
    atomic_json(out / "etude-result.json", {"clocks": {"updates": UPDATES}})


def admit_throughput(
    out: Path, stop: Path, deadline: float, limits: AdmissionLimits
) -> None:
    """Measure a complete two-arm sequential batch without scoring."""
    phase = out / "admission-1"
    phase.mkdir()
    started = time.monotonic()
    costs: list[ArmThroughput] = []
    atomic_json(out / "admission-limits.json", limits.model_dump(mode="json"))
    for index, arm in enumerate(ARMS):
        batch = phase / f"batch-{index}"
        batch.mkdir()
        print(
            f"admission: {arm}, {STREAMS} streams, {ADMISSION_UPDATES} updates",
            flush=True,
        )
        child_started = time.monotonic()
        watch = ProgressWatch(
            arm,
            phase / arm / "training/run.json",
            batch / "stop",
            limits,
            child_started,
        )
        command = [
            *_CHILD,
            "--admit-arm",
            arm,
            "--out",
            str(phase / arm),
            "--stop",
            str(watch.stop),
        ]
        _run_child(
            command,
            batch,
            stop,
            min(
                deadline - TEARDOWN_SECONDS,
                time.time() + ADMISSION_ARM_SECONDS + limits.grace_seconds,
            ),
            watch=watch,
        )
        run = TrainingRun.model_validate_json(watch.path.read_bytes())
        if run.status != "completed":
            raise RuntimeError("admission learner did not complete")
        costs.append(ArmThroughput.from_run(arm, run, time.monotonic() - child_started))
        print(f"admission: {arm} completed", flush=True)
    result = ThroughputAdmission(
        concurrency=1,
        elapsed_seconds=time.monotonic() - started,
        arms=(costs[0], costs[1]),
    )
    atomic_json(out / "throughput.json", [result.model_dump()])
    atomic_json(out / "etude-result.json", {"clocks": {"updates": ADMISSION_UPDATES}})


def _run_child(
    command: list[str],
    out: Path,
    stop: Path,
    deadline: float,
    *,
    watch: ProgressWatch | None = None,
) -> None:
    """Bound one worker and its descendants by the rental on every exit.

    A watched child gets a graceful stop and its grace period; any other child
    is terminated directly.
    """
    # The learner owns local tracking; it must not attach to the launcher run.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("ETUDE_")
    }
    with (out / "child-0.log").open("ab") as log:
        try:
            child = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            atomic_json(out / "processes.json", {"exit_codes": []})
            raise
    grace = watch.limits.grace_seconds if watch else 0
    try:
        while child.poll() is None:
            reason = None
            if stop.exists() or time.time() >= deadline - grace:
                reason = "worker cancelled or allocation expired"
            elif watch is not None:
                run = (
                    TrainingRun.model_validate_json(watch.path.read_bytes())
                    if watch.path.exists()
                    else None
                )
                previous = (watch.last_progress, watch.updates)
                reason = watch.observe(run, time.monotonic())
                if run is not None and previous != (watch.last_progress, watch.updates):
                    atomic_json(out / "progress.json", run.model_dump(mode="json"))
                    stage = run.stages[0]
                    print(
                        f"admission: {watch.arm} updates={stage.updates} "
                        f"games={stage.games} collection={stage.collection_seconds:.2f}s "
                        f"learning={stage.learning_seconds:.2f}s "
                        f"exports={stage.export_seconds:.2f}s",
                        flush=True,
                    )
            if reason is not None:
                atomic_json(
                    out / "supervision.json",
                    {
                        "reason": reason,
                        "grace_seconds": grace,
                        "updates": watch.updates if watch else None,
                    },
                )
                if watch is not None:
                    watch.stop.touch()
                    # Once stopped, only evidence flush and exit remain. Waiting
                    # here avoids a second polling state and cannot extend the lease.
                    with suppress(subprocess.TimeoutExpired):
                        child.wait(timeout=max(0, min(grace, deadline - time.time())))
                raise RuntimeError(reason)
            time.sleep(1)
        if child.returncode != 0:
            raise RuntimeError(f"worker child failed: exit {child.returncode}")
    finally:
        forced = child.poll() is None
        if forced:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        atomic_json(
            out / "processes.json",
            {
                "exit_codes": [child.returncode],
                "forced_termination": forced,
            },
        )


def worker() -> None:
    """Etude owns the capped allocation and every learner/evaluator child."""
    bind_etude_source()
    spec = StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    work_hours = spec.allocation.work_hours if spec.allocation else ADMISSION_WORK_HOURS
    required = work_hours * 3600 - 180
    if deadline - time.time() < required:
        raise ValueError("remaining allocation cannot cover declared work")
    out.mkdir(parents=True, exist_ok=True)
    gate = check_numerics(out, min(time.time() + 300, deadline))
    if gate.status != "completed":
        raise RuntimeError("CUDA numerical admission failed")
    if spec.allocation:
        if torch.cuda.get_device_name(0) != spec.allocation.hardware:
            raise ValueError("scientific hardware differs from throughput admission")
        atomic_json(out / "allocation.json", spec.allocation.model_dump(mode="json"))
        scientific_cohort(out, stop, deadline)
    else:
        admit_throughput(out, stop, deadline, spec.admission_limits)


# Complete-cohort analysis; no partial successes or checkpoint selection.


@dataclass(frozen=True)
class PlateauInterval:
    arm: Arm
    before: int
    after: int
    estimate: PairedEstimate

    @property
    def bounded(self) -> bool:
        return self.estimate.lower > -0.05 and self.estimate.upper < 0.05


@dataclass(frozen=True)
class PlateauResult:
    reached: bool
    observed: int
    expected: int
    intervals: tuple[PlateauInterval, ...] = ()


def plateau(points: list[ArmEvaluation]) -> PlateauResult:
    """Positive bounded-change evidence across seeds, not a nonsignificant slope.

    The fixed final window avoids choosing a favorable flat patch after seeing
    the curve. This conditional small-sample diagnostic is not equivalence of
    architectures or proof that no later improvement is possible.
    """
    expected = {
        (arm, seed, update) for arm in ARMS for seed in SEEDS for update in POINTS
    }
    protocol = evaluation_protocol()
    digests = recipe_digests()
    scores: dict[tuple[Arm, int, int], float] = {}
    for item in points:
        result = item.result
        key = item.arm, result.training_seed, result.coordinates.updates
        if key in scores or key not in expected:
            raise ValueError("duplicate or undeclared curve point")
        scores[key] = item.score_for(protocol, digests[item.arm])
    if set(scores) != expected:
        return PlateauResult(False, len(scores), len(expected))
    intervals: list[PlateauInterval] = []
    for arm in ARMS:
        for before, after in zip(POINTS[-4:-1], POINTS[-3:], strict=True):
            estimate = paired_interval(
                tuple(
                    scores[arm, seed, after] - scores[arm, seed, before]
                    for seed in SEEDS
                )
            )
            intervals.append(PlateauInterval(arm, before, after, estimate))
    return PlateauResult(
        all(item.bounded for item in intervals),
        len(scores),
        len(expected),
        tuple(intervals),
    )


def report() -> Report:
    return Report(clock="learner_transitions", perf=PERF, ml=ML)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--admit-arm", choices=ARMS)
    mode.add_argument("--train-arm", choices=ARMS)
    mode.add_argument("--evaluate-arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop", type=Path)
    args = parser.parse_args()
    if args.admit_arm:
        if args.out is None or args.stop is None:
            parser.error("admission needs output and stop path")
        train_arm(args.admit_arm, ADMISSION_SEED, args.out, args.stop, admission=True)
    elif args.train_arm or args.evaluate_arm:
        if args.seed is None or args.out is None:
            parser.error("scientific worker needs seed and output")
        if args.train_arm:
            if args.stop is None:
                parser.error("training requires stop path")
            train_arm(args.train_arm, args.seed, args.out, args.stop)
        else:
            if args.deadline is None:
                parser.error("evaluation requires deadline")
            evaluate_arm(args.evaluate_arm, args.seed, args.out, args.deadline)
    else:
        worker()


if __name__ == "__main__":
    main()
