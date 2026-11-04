"""Sequential GPU admission and prospectively budgeted representation learning.

The same complete-game executor owns both excluded throughput admission and
eight independent scientific learners. A frozen allocation binds measured
throughput and a rental quote before scientific execution; one child owns the
GPU at a time. Monitoring exports and evaluation retain every scheduled point.
"""

import argparse
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
SEEDS = (74211, 74212, 74213, 74214)
STREAMS = 256
UPDATES = 256
CADENCE = 32
POINTS = tuple(range(0, UPDATES + 1, CADENCE))
ADMISSION_UPDATES = 4
ADMISSION_SEED = 74200
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


class ThroughputAdmission(BaseModel):
    """Ordinary two-arm work, including child startup and checkpoint exports.

    Project the entire eight-learner workload, not a per-process latency. A 50%
    reserve covers early-run extrapolation; this is capacity evidence, never a
    learning result or a guarantee about later game length.
    """

    model_config = ConfigDict(extra="forbid")
    concurrency: Literal[1]
    elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    updates_per_arm: int = Field(ge=1)

    def cohort_hours(self) -> float:
        training = self.elapsed_seconds * len(SEEDS) * UPDATES / self.updates_per_arm
        # Preserve 90 minutes for evaluation and 15 minutes for source setup.
        return (1.5 * training + 5400 + 900) / 3600

    def projected_usd(self, hourly_usd: Decimal) -> Decimal:
        if not hourly_usd.is_finite() or hourly_usd <= 0:
            raise ValueError("hourly price must be finite and positive")
        return Decimal(str(self.cohort_hours())) * hourly_usd


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
        return self.admission.cohort_hours() - 0.25

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


def runs(revision: str, allocation: str = "") -> list[Run]:
    """Default to admission; science requires an explicit frozen allocation file."""
    root = Path(__file__).resolve().parents[1]
    admitted = (
        ScientificAllocation.model_validate_json(Path(allocation).read_bytes())
        if allocation
        else None
    )
    spec = StudyJob(source_bundle=bundle_at(root, revision), allocation=admitted)
    job = Job(
        name=EXPERIMENT,
        experiment=EXPERIMENT,
        setup=machine_setup(" --extra dev --extra artifacts --extra evaluation"),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked --extra dev --extra artifacts --extra evaluation python -m experiments.state_representation_gpu",
        repo=str(root),
        revision=revision,
        config=spec.model_dump_json(),
        until=f"updates={UPDATES if admitted else ADMISSION_UPDATES}",
        max_hours=admitted.work_hours if admitted else 0.5,
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
    regime = recipe(arm)
    if admission:
        regime.id = f"representation-gpu-admission-{arm}"
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = ADMISSION_UPDATES
        stage.execution.wall_seconds = regime.wall_seconds = 600
        regime = TrainingRegime.model_validate(regime.model_dump())
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
        or run.regime_digest != canonical_sha256(recipe(arm).model_dump(mode="json"))
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
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    protocol = evaluation_protocol(heldout=True)
    recipe_digests = {
        arm: canonical_sha256(recipe(arm).model_dump(mode="json")) for arm in ARMS
    }
    scores: dict[tuple[Arm, int], float] = {}
    for item in results:
        key = item.arm, item.result.training_seed
        if (
            key not in expected
            or key in scores
            or item.result.coordinates.updates != UPDATES
        ):
            raise ValueError("duplicate or undeclared held-out endpoint")
        scores[key] = item.score_for(protocol, recipe_digests[item.arm])
    if set(scores) != expected:
        raise ValueError("every held-out arm and seed is required")
    return paired_interval(
        tuple(scores["typed_gru", seed] - scores["typed_pool", seed] for seed in SEEDS)
    )


def scientific_cohort(out: Path, stop: Path, deadline: float) -> None:
    """Sequential balanced arm order; stop and retain the denominator on failure."""
    assignments = [
        (arm, seed)
        for index, seed in enumerate(SEEDS)
        for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
    ]
    base = [sys.executable, "-m", "experiments.state_representation_gpu"]
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
            command = base + [
                f"--{mode}-arm",
                arm,
                "--seed",
                str(seed),
                "--out",
                str(out / f"{arm}-{seed}"),
            ]
            bound = deadline - (5400 if mode == "train" else 120)
            command += (
                ["--stop", str(stop)] if mode == "train" else ["--deadline", str(bound)]
            )
            print(f"{mode}: {arm} seed {seed}", flush=True)
            _run_child(command, logs, stop, bound)
    curves: list[ArmEvaluation] = []
    final: list[ArmEvaluation] = []
    for arm, seed in assignments:
        directory = out / f"{arm}-{seed}"
        for update in POINTS:
            curves.append(
                ArmEvaluation(
                    arm=arm,
                    result=MonitorResult.model_validate_json(
                        (directory / f"curve-{update:04d}/monitor.json").read_bytes()
                    ),
                )
            )
        final.append(
            ArmEvaluation(
                arm=arm,
                result=MonitorResult.model_validate_json(
                    (directory / "heldout/monitor.json").read_bytes()
                ),
            )
        )
    atomic_json(
        out / "comparison.json",
        {
            "plateau": asdict(plateau(curves)),
            "heldout": asdict(compare_heldout(final)),
        },
    )
    atomic_json(out / "etude-result.json", {"clocks": {"updates": UPDATES}})


def admit_throughput(out: Path, stop: Path, deadline: float) -> None:
    """Measure a complete two-arm sequential batch without scoring."""
    phase = out / "admission-1"
    phase.mkdir()
    started = time.monotonic()
    for index, arm in enumerate(ARMS):
        batch = phase / f"batch-{index}"
        batch.mkdir()
        print(
            f"admission: {arm}, {STREAMS} streams, {ADMISSION_UPDATES} updates",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "experiments.state_representation_gpu",
            "--admit-arm",
            arm,
            "--out",
            str(phase / arm),
            "--stop",
            str(stop),
        ]
        _run_child(command, batch, stop, min(deadline - 120, time.time() + 720))
        print(f"admission: {arm} completed", flush=True)
    result = ThroughputAdmission(
        concurrency=1,
        elapsed_seconds=time.monotonic() - started,
        updates_per_arm=ADMISSION_UPDATES,
    )
    atomic_json(out / "throughput.json", [result.model_dump()])
    atomic_json(out / "etude-result.json", {"clocks": {"updates": ADMISSION_UPDATES}})


def _run_child(command: list[str], out: Path, stop: Path, deadline: float) -> None:
    """Bound one worker and its descendants by the rental on every exit."""
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
    try:
        while child.poll() is None:
            if stop.exists() or time.time() >= deadline:
                raise RuntimeError("worker cancelled or allocation expired")
            time.sleep(1)
        if child.returncode != 0:
            raise RuntimeError(f"worker child failed: exit {child.returncode}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        atomic_json(out / "processes.json", {"exit_codes": [child.returncode]})


def worker() -> None:
    """Etude owns the capped allocation and every learner/evaluator child."""
    bind_etude_source()
    spec = StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    required = spec.allocation.work_hours * 3600 - 180 if spec.allocation else 1620
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
        admit_throughput(out, stop, deadline)


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
    recipe_digests = {
        arm: canonical_sha256(recipe(arm).model_dump(mode="json")) for arm in ARMS
    }
    scores: dict[tuple[Arm, int, int], float] = {}
    for item in points:
        result = item.result
        key = item.arm, result.training_seed, result.coordinates.updates
        if key in scores or key not in expected:
            raise ValueError("duplicate or undeclared curve point")
        scores[key] = item.score_for(protocol, recipe_digests[item.arm])
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
