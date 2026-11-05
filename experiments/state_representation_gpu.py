"""GPU admission and the recovery cohort of the representation plateau study.

The same complete-game executor owns excluded throughput admission and each
scientific learner. After Run `e51f61b92931` was lost, `RECOVERY` freezes the
amendment that lets three retained learners be evaluated while the other five
train. `SCORING` is untouched: it stays the evaluator's identity.

Every recovery unit is its own Etude Run, because outputs are delivered only when
an attempt ends. `LearnerUnit` trains one fresh learner on one GPU;
`EvaluationUnit` scores endpoints of one learner on CPU, from checkpoints given
as verified inputs. The ledger (`pending_endpoints`, `verified_result`) reads
delivered attempts: a complete hash-verified `monitor.json` is never repeated,
and an unscored attempt stays in the denominator and repeats in a fresh directory.
"""

import argparse
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Annotated, Literal

from etude.launch.records import Input, Job, Run
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
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.recipes.builders import move_learning
from manabot.sim.execution import EvalExecution
from manabot.training.execution import execute_regime
from manabot.training.panels import ML, PERF
from manabot.training.records import TrainingCoordinates, TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    IterationSchedule,
    TrainingRegime,
    TrainSelfPlay,
)
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

# The executed cohort's scheduled order, preserved for the five missing learners.
ASSIGNMENTS: tuple[tuple[Arm, int], ...] = tuple(
    (arm, seed)
    for index, seed in enumerate(SEEDS)
    for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
)
GPU_CLASS = "NVIDIA GeForce RTX 3090"
# SHA256 of each retained run.json, from the 13:00:43 UTC receipt of `e51f61b92931`.
RETAINED_RUNS: dict[tuple[Arm, int], str] = {
    ("typed_pool", 74211): (
        "12fd45fadfdd3202f38d6da63dad8d28e2801a70a72fb7e0411fb6d862da6d82"
    ),
    ("typed_gru", 74211): (
        "704c9f8bcf940da09868d6c68fcdbb2f53679c8320c84eef2589c2f2ca9b8e5c"
    ),
    ("typed_gru", 74212): (
        "154776ff30546249f6eac9ac18e3f913a53da0596137d48fe9f5d52c9f1a678a"
    ),
}
MISSING_LEARNERS = tuple(item for item in ASSIGNMENTS if item not in RETAINED_RUNS)
# The interrupted learner restarts from initialization on its seed. This digest is
# compared with the restart and recorded; it does not gate training.
INTERRUPTED_INITIAL_RAW: dict[tuple[Arm, int], str] = {
    ("typed_pool", 74212): (
        "db81a9eacee1bf475a2c90d08f4afb43608f17b59e7e799aa6e51f036d5e7f7f"
    ),
}
RECOVERY = {
    "name": "state-representation-gpu-recovery-20261105",
    "scoring_sha256": canonical_sha256(SCORING),
    "failed_run": "e51f61b92931",
    "retained_runs": sorted(
        (arm, seed, digest) for (arm, seed), digest in RETAINED_RUNS.items()
    ),
    "missing_learners": MISSING_LEARNERS,
    "interrupted_initial_raw": sorted(
        (arm, seed, digest) for (arm, seed), digest in INTERRUPTED_INITIAL_RAW.items()
    ),
    "hardware": f"one learner per {GPU_CLASS}; each learner's host recorded",
    "evaluation": "CPU, four active games, batch 32, 1GiB residency, per-unit Runs",
    "retry": (
        "amends SCORING's no-retries clause for this cohort only: a learner lost "
        "before completion restarts from initialization on its seed; an unscored "
        "endpoint repeats in a fresh attempt; a scored endpoint never repeats; "
        "every attempt stays in the denominator"
    ),
    "interim": (
        "partial curves are descriptive; interim scores select no treatment, "
        "retry, horizon or checkpoint"
    ),
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


class Endpoint(BaseModel):
    """One attempt at a scheduled curve point or the final held-out cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    update: int
    heldout: bool = False
    attempt: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def scheduled(self) -> "Endpoint":
        if self.update not in POINTS or (self.heldout and self.update != UPDATES):
            raise ValueError("undeclared evaluation endpoint")
        return self

    @property
    def name(self) -> str:
        return "heldout" if self.heldout else f"curve-{self.update:04d}"

    def directory(self, arm: Arm, seed: int) -> Path:
        """This attempt's own place below the ledger root, in delivery and at rest."""
        return Path(f"{arm}-{seed}", self.name, f"attempt-{self.attempt:04d}")


# Curve points in update order, then the held-out cohort: 72 + 8 across the study.
ENDPOINTS = (
    *(Endpoint(update=update) for update in POINTS),
    Endpoint(update=UPDATES, heldout=True),
)


class LearnerUnit(BaseModel):
    """One learner trained from initialization on one GPU; `attempt` counts restarts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["train"] = "train"
    arm: Arm
    seed: int
    attempt: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def missing(self) -> "LearnerUnit":
        if (self.arm, self.seed) not in MISSING_LEARNERS:
            raise ValueError("retained learners are immutable and never retrained")
        return self


class EvaluationUnit(BaseModel):
    """Endpoints of one learner, scored in order; `run_sha256` names its run.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["evaluate"] = "evaluate"
    arm: Arm
    seed: int
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoints: tuple[Endpoint, ...] = Field(min_length=1)
    # What actually shares this job's window, stated at dispatch. A scored result
    # retains it, so it must not claim anything the operator cannot observe.
    concurrent_activity: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct(self) -> "EvaluationUnit":
        if (self.arm, self.seed) not in ASSIGNMENTS:
            raise ValueError("undeclared learner")
        names = [endpoint.name for endpoint in self.endpoints]
        if len(names) != len(set(names)):
            raise ValueError("an evaluation Run scores each endpoint once")
        return self


RecoveryUnit = Annotated[LearnerUnit | EvaluationUnit, Field(discriminator="kind")]


def unit_key(unit: LearnerUnit | EvaluationUnit) -> str:
    """Stable Etude Run id: declaring the same work again finds the same Run.

    Concurrent activity describes a dispatch, not what the unit computes.
    """
    identity = unit.model_dump(mode="json", exclude={"concurrent_activity"})
    return canonical_sha256({"recovery": RECOVERY, "unit": identity})[:12]


class RecoveryAllocation(BaseModel):
    """One unit's bounds and inputs; the operator checks them against a fresh quote."""

    model_config = ConfigDict(extra="forbid")
    unit: RecoveryUnit
    max_hours: float = Field(gt=0)
    max_usd: Decimal = Field(gt=0, allow_inf_nan=False)
    inputs: tuple[Input, ...] = ()

    @model_validator(mode="after")
    def supplied(self) -> "RecoveryAllocation":
        if isinstance(self.unit, EvaluationUnit) != bool(self.inputs):
            raise ValueError("evaluation alone reads delivered checkpoints")
        return self


class StudyJob(BaseModel):
    """Source identity and, for a recovery unit, its frozen allocation."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle
    recovery: RecoveryAllocation | None = None
    admission_limits: AdmissionLimits = Field(default_factory=AdmissionLimits)


# Job declaration and independent learner/evaluator entry points.


def runs(revision: str, unit: str = "", admission_limits: str = "") -> list[Run]:
    """Default to admission; a recovery unit requires its frozen allocation file."""
    root = Path(__file__).resolve().parents[1]
    allocated = (
        RecoveryAllocation.model_validate_json(Path(unit).read_bytes())
        if unit
        else None
    )
    limits = (
        AdmissionLimits.model_validate_json(Path(admission_limits).read_bytes())
        if admission_limits
        else AdmissionLimits()
    )
    spec = StudyJob(
        source_bundle=bundle_at(root, revision),
        recovery=allocated,
        admission_limits=limits,
    )
    name, until, hours, usd, inputs = (
        EXPERIMENT,
        f"updates={ADMISSION_UPDATES}",
        ADMISSION_WORK_HOURS,
        Decimal("0.50"),
        (),
    )
    if allocated is not None:
        work = allocated.unit
        hours, usd, inputs = allocated.max_hours, allocated.max_usd, allocated.inputs
        name = f"{EXPERIMENT}-{work.kind}-{work.arm}-{work.seed}"
        until = (
            f"updates={UPDATES}"
            if isinstance(work, LearnerUnit)
            else f"endpoints={len(work.endpoints)}"
        )
    job = Job(
        name=name,
        experiment=EXPERIMENT,
        setup=machine_setup(UV_EXTRAS),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked{UV_EXTRAS} python -m experiments.{EXPERIMENT}",
        repo=str(root),
        revision=revision,
        config=spec.model_dump_json(),
        until=until,
        max_hours=hours,
        max_usd=usd,
        max_attempts=1,
        inputs=inputs,
    )
    return [
        run.model_copy(update={"id": unit_key(allocated.unit)}) if allocated else run
        for run in job.runs(0)
    ]


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
    science requires every declared export before evaluation can begin. A restart
    uses a fresh Run, so `out` never holds an earlier attempt.
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


def checkpoint_input(artifact: ArtifactReference) -> str:
    """The input name of a checkpoint; delivered bytes are addressed by digest."""
    return f"{artifact['sha256']}.pt"


def evaluation_inputs(
    run: TrainingRun, run_json: Path, prefix: str
) -> tuple[Input, ...]:
    """A learner's run.json and every scheduled checkpoint, as bucket objects."""
    run_sha256 = file_sha256(run_json)
    points = [point.artifact for point in checkpoints(run)]
    return (
        Input(
            name="run.json",
            key=f"{prefix}/{run_sha256}.json",
            sha256=run_sha256,
            size_bytes=run_json.stat().st_size,
        ),
        *(
            Input(
                name=checkpoint_input(artifact),
                key=f"{prefix}/{artifact['sha256']}.pt",
                sha256=artifact["sha256"],
                size_bytes=artifact["bytes"],
            )
            for artifact in points
        ),
    )


class EndpointReceipt(BaseModel):
    """Hashes of one finished attempt, written beside its monitor.json."""

    model_config = ConfigDict(extra="forbid")
    recovery_sha256: str
    run_id: str
    run_sha256: str
    artifact_sha256: str
    status: str
    # Attempt-relative path to SHA256 for the monitor, its traces and allocations.
    files: dict[str, str]


def _attempt_files(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }


def evaluate_unit(
    unit: EvaluationUnit, inputs: Path, out: Path, deadline: float
) -> None:
    """Score declared endpoints in order, retaining each before starting the next.

    The retained run.json names checkpoint paths on a machine that no longer
    exists. Exact bytes come from `inputs`; `checkpoint_manifest` then checks
    digest, size and world/ABI/engine identities against the running source.
    """
    source = inputs / "run.json"
    run = TrainingRun.model_validate_json(source.read_bytes())
    if (
        file_sha256(source) != unit.run_sha256
        or run.seed != unit.seed
        or run.regime_digest != recipe_digests()[unit.arm]
    ):
        raise ValueError("run is outside the declared GPU cohort")
    points = {point.coordinates.updates: point for point in checkpoints(run)}
    for endpoint in unit.endpoints:
        if time.time() >= deadline:
            raise RuntimeError("evaluation allocation expired")
        point = points[endpoint.update]
        directory = out / endpoint.directory(unit.arm, unit.seed)
        artifact: ArtifactReference = {
            **point.artifact,
            "path": str(inputs / checkpoint_input(point.artifact)),
        }
        directory.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate_checkpoint(
            run,
            artifact,
            point.coordinates,
            directory,
            protocol=evaluation_protocol(heldout=endpoint.heldout),
            concurrent_activity=unit.concurrent_activity,
            execution=EvalExecution(
                device="cpu",
                max_active_games=4,
                max_batch_rows=32,
                resident_model_bytes=1024**3,
                deadline_unix=deadline,
            ),
        )
        atomic_json(
            directory / "receipt.json",
            EndpointReceipt(
                recovery_sha256=canonical_sha256(RECOVERY),
                run_id=run.id,
                run_sha256=unit.run_sha256,
                artifact_sha256=point.artifact["sha256"],
                status=result.status,
                files=_attempt_files(directory),
            ).model_dump(mode="json"),
        )
        if result.status != "completed":
            raise RuntimeError("incomplete evaluation retained in study denominator")


# The evaluation ledger: delivered attempts under one root, keyed by directory.


def verified_result(directory: Path) -> MonitorResult | None:
    """A scored attempt's result, or None for one that produced no score.

    A scored attempt is never repeated, so one that fails its hashes is an error
    for a person to resolve, not a reason to play the endpoint again.
    """
    monitor = directory / "monitor.json"
    if not monitor.exists():
        return None
    result = MonitorResult.model_validate_json(monitor.read_bytes())
    if result.score is None:
        return None
    path = directory / "receipt.json"
    if not path.exists():
        raise ValueError(f"scored attempt lacks a receipt: {directory}")
    receipt = EndpointReceipt.model_validate_json(path.read_bytes())
    if (
        receipt.status != "completed"
        or receipt.artifact_sha256 != result.artifact["sha256"]
        or receipt.files != _attempt_files(directory)
    ):
        raise ValueError(f"scored attempt fails hash verification: {directory}")
    return result


def endpoint_attempts(
    root: Path, arm: Arm, seed: int, endpoint: Endpoint
) -> tuple[Path, ...]:
    return tuple(sorted((root / f"{arm}-{seed}" / endpoint.name).glob("attempt-*")))


def pending_endpoints(root: Path, arm: Arm, seed: int) -> tuple[Endpoint, ...]:
    """What this learner still needs: unscored endpoints, each at its next attempt."""
    pending: list[Endpoint] = []
    for endpoint in ENDPOINTS:
        attempts = endpoint_attempts(root, arm, seed, endpoint)
        scored = [path for path in attempts if verified_result(path) is not None]
        if len(scored) > 1:
            raise ValueError(f"endpoint scored more than once: {scored}")
        if not scored:
            pending.append(endpoint.model_copy(update={"attempt": len(attempts) + 1}))
    return tuple(pending)


def recovery_comparison(root: Path) -> dict[str, object]:
    """The frozen analysis, once every endpoint of all eight learners is scored."""
    curves: list[ArmEvaluation] = []
    final: list[ArmEvaluation] = []
    for arm, seed in ASSIGNMENTS:
        if pending_endpoints(root, arm, seed):
            raise ValueError(
                "the frozen analysis needs every endpoint; partial curves are descriptive"
            )
        for endpoint in ENDPOINTS:
            (result,) = filter(
                None,
                map(verified_result, endpoint_attempts(root, arm, seed, endpoint)),
            )
            (final if endpoint.heldout else curves).append(
                ArmEvaluation(arm=arm, result=result)
            )
    return {
        "plateau": asdict(plateau(curves)),
        "heldout": asdict(compare_heldout(final)),
    }


def compare_heldout(results: list[ArmEvaluation]) -> PairedEstimate:
    return compare_cohort(
        results, SEEDS, UPDATES, evaluation_protocol(heldout=True), recipe_digests()
    )


# Unit execution and child lifetime ownership.

_CHILD = (sys.executable, "-m", f"experiments.{EXPERIMENT}")


def host_record() -> dict[str, object]:
    """What a learner's or evaluator's host was, beside Etude's provider receipt."""
    cpuinfo = Path("/proc/cpuinfo")
    models = (
        {
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith("model name")
        }
        if cpuinfo.exists()
        else set()
    )
    return {
        "node": platform.node(),
        "cpu_models": sorted(models),
        "cpu_count": os.cpu_count(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "load": os.getloadavg(),
    }


def run_unit(
    unit: LearnerUnit | EvaluationUnit,
    out: Path,
    inputs: Path | None,
    stop: Path,
    deadline: float,
) -> None:
    """Execute one recovery unit in a bounded child and report Etude's clock."""
    atomic_json(out / "recovery.json", RECOVERY)
    logs = out / "logs"
    logs.mkdir()
    if isinstance(unit, LearnerUnit):
        command = [
            *_CHILD,
            "--train-arm",
            unit.arm,
            "--seed",
            str(unit.seed),
            "--out",
            str(out / f"{unit.arm}-{unit.seed}"),
            "--stop",
            str(stop),
        ]
        clock = {"updates": UPDATES}
    else:
        if inputs is None:
            raise ValueError("evaluation needs delivered inputs")
        document = out / "unit.json"
        atomic_json(document, unit.model_dump(mode="json"))
        command = [
            *_CHILD,
            "--evaluate-unit",
            str(document),
            "--inputs",
            str(inputs),
            "--out",
            str(out),
            "--deadline",
            str(deadline - TEARDOWN_SECONDS),
        ]
        clock = {"endpoints": len(unit.endpoints)}
    print(f"{unit.kind}: {unit.arm} seed {unit.seed}", flush=True)
    _run_child(command, logs, stop, deadline - TEARDOWN_SECONDS)
    atomic_json(out / "etude-result.json", {"clocks": clock})


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
    """Etude owns the capped allocation and the learner or evaluator child."""
    bind_etude_source()
    spec = StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    work_hours = spec.recovery.max_hours if spec.recovery else ADMISSION_WORK_HOURS
    required = work_hours * 3600 - 180
    if deadline - time.time() < required:
        raise ValueError("remaining allocation cannot cover declared work")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "host.json", host_record())
    if spec.recovery is not None and isinstance(spec.recovery.unit, EvaluationUnit):
        inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
        run_unit(spec.recovery.unit, out, inputs, stop, deadline)
        return
    gate = check_numerics(out, min(time.time() + 300, deadline))
    if gate.status != "completed":
        raise RuntimeError("CUDA numerical admission failed")
    if spec.recovery is None:
        admit_throughput(out, stop, deadline, spec.admission_limits)
        return
    if torch.cuda.get_device_name(0) != GPU_CLASS:
        raise ValueError("learner hardware differs from the amendment's GPU class")
    run_unit(spec.recovery.unit, out, None, stop, deadline)


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
    mode.add_argument("--evaluate-unit", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop", type=Path)
    args = parser.parse_args()
    if args.admit_arm:
        if args.out is None or args.stop is None:
            parser.error("admission needs output and stop path")
        train_arm(args.admit_arm, ADMISSION_SEED, args.out, args.stop, admission=True)
    elif args.train_arm:
        if args.seed is None or args.out is None or args.stop is None:
            parser.error("training needs seed, output and stop path")
        train_arm(args.train_arm, args.seed, args.out, args.stop)
    elif args.evaluate_unit:
        if args.inputs is None or args.out is None or args.deadline is None:
            parser.error("evaluation needs inputs, output and deadline")
        unit = EvaluationUnit.model_validate_json(args.evaluate_unit.read_bytes())
        evaluate_unit(unit, args.inputs, args.out, args.deadline)
    else:
        worker()


if __name__ == "__main__":
    main()
