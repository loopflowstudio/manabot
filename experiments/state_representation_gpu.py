"""GPU study contract and bounded admission for its replacement allocation.

The failed eight-context runner is retained in source 27a126422. Current runs
measure ordinary learning at one and two concurrent learners on an excluded
seed, without scoring. Scientific recipes and plateau analysis remain here;
a replacement scientific allocation requires measured affordability first.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from etude.launch.records import Job, Run
from etude.report import Report
from pydantic import BaseModel, ConfigDict, Field
import torch

from experiments.batched_baseline import batched_baseline
from experiments.state_representation_v7 import (
    ARMS,
    Arm,
    PairedEstimate,
    paired_interval,
)
from manabot.env.observation import ObservationConfig
from manabot.eval.monitor import MonitorProtocol, MonitorResult
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.experiment.numerics import check_numerics
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.recipes.builders import move_learning
from manabot.training.execution import execute_regime
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

EXPERIMENT = "state_representation_gpu"
SEEDS = (74201, 74202, 74203, 74204)
UPDATES = 2048
CADENCE = 256
POINTS = tuple(range(0, UPDATES + 1, CADENCE))
ADMISSION_UPDATES = 8
ADMISSION_SEED = 74200
CURVE_DEALS = tuple(range(85001, 85033))
HELDOUT_DEALS = tuple(range(86001, 86065))
SCORING = {
    "name": "state-representation-gpu-horizon-5da5f4",
    "seeds": SEEDS,
    "arms": ARMS,
    "points": POINTS,
    "curve_deals": CURVE_DEALS,
    "heldout_deals": HELDOUT_DEALS,
    "plateau": "last three successive 256-update gains: each arm's paired-seed 95% t interval entirely within +/-0.05; fixed final window",
    "primary": "GRU minus pooled at common final update 2048 on untouched held-out deals",
    "failure": "missing seeds/checkpoints/games prevent a qualified result; no retries or selection",
}


def recipe(arm: Arm) -> TrainingRegime:
    """Fresh CUDA learning with absolute, nonzero LR/tau schedules in both arms."""
    if arm not in ARMS:
        raise ValueError("unknown study arm")
    regime = batched_baseline()
    regime.id = f"state-representation-gpu-{arm}"
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
    stage.streams = 32
    stage.updates = UPDATES
    stage.learning = move_learning()
    stage.execution.device = "cuda"
    stage.execution.memory_bytes = 6 * 1024**3
    stage.execution.wall_seconds = 66000
    regime.wall_seconds = 66000
    return TrainingRegime.model_validate(regime.model_dump())


class StudyJob(BaseModel):
    """Source identity for one excluded throughput admission."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle


def runs(revision: str) -> list[Run]:
    """One capped admission rental; this launches no scientific replicates."""
    root = Path(__file__).resolve().parents[1]
    spec = StudyJob(source_bundle=bundle_at(root, revision))
    job = Job(
        name=EXPERIMENT,
        experiment=EXPERIMENT,
        setup=machine_setup(" --extra dev --extra artifacts --extra evaluation"),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked --extra dev --extra artifacts --extra evaluation python -m experiments.state_representation_gpu",
        repo=str(root),
        revision=revision,
        config=spec.model_dump_json(),
        until=f"updates={ADMISSION_UPDATES}",
        max_hours=1,
        max_usd=Decimal("0.50"),
        max_attempts=1,
    )
    # This is an allocation identity, never a scientific training seed.
    return job.runs(0)


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


def admit_arm(arm: Arm, out: Path, stop: Path) -> None:
    """Ordinary learning on one excluded seed, never an architecture score."""
    if not torch.cuda.is_available():
        raise RuntimeError("this admission requires CUDA")
    # At most two learners: leave 20% outside their caching allocators for
    # contexts and workspaces. Capacity failure remains a recorded failure.
    torch.cuda.set_per_process_memory_fraction(0.40)
    regime = recipe(arm)
    regime.id = f"representation-gpu-admission-{arm}"
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = ADMISSION_UPDATES
    stage.execution.wall_seconds = 600
    regime.wall_seconds = 600
    regime = TrainingRegime.model_validate(regime.model_dump())
    out.mkdir(parents=True, exist_ok=False)
    with VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(
            regime,
            ADMISSION_SEED,
            out / "training",
            store,
            experiment=f"{EXPERIMENT}_admission",
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
    if run.status != "completed" or run.updates_through() != ADMISSION_UPDATES:
        raise RuntimeError(f"incomplete admission: {run.status}: {run.error}")


class ThroughputAdmission(BaseModel):
    """Ordinary two-arm work, including child startup and checkpoint exports.

    Project the entire eight-learner workload, not a per-process latency. A 50%
    reserve covers early-run extrapolation; this is capacity evidence, never a
    learning result or a guarantee about later game length.
    """

    model_config = ConfigDict(extra="forbid")
    concurrency: Literal[1, 2]
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


def admit_throughput(out: Path, stop: Path, deadline: float) -> None:
    """Measure single and paired learners without scoring or scientific seeds."""
    results: list[ThroughputAdmission] = []
    for concurrency in (1, 2):
        phase = out / f"admission-{concurrency}"
        phase.mkdir()
        started = time.monotonic()
        for index in range(0, len(ARMS), concurrency):
            batch = phase / f"batch-{index}"
            batch.mkdir()
            commands = [
                [
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
                for arm in ARMS[index : index + concurrency]
            ]
            _batch(commands, batch, stop, min(deadline - 120, time.time() + 720))
        results.append(
            ThroughputAdmission(
                concurrency=concurrency,
                elapsed_seconds=time.monotonic() - started,
                updates_per_arm=ADMISSION_UPDATES,
            )
        )
        atomic_json(
            out / "throughput.json", [result.model_dump() for result in results]
        )
    atomic_json(out / "etude-result.json", {"clocks": {"updates": ADMISSION_UPDATES}})


def _batch(commands: list[list[str]], out: Path, stop: Path, deadline: float) -> None:
    """Bound child lifetimes by the shared rental, and reap every child on exit."""
    children: list[subprocess.Popen[bytes]] = []
    # Each child creates its own local tracking run under its retained output.
    # Passing the parent's launcher identity would attach every child to one run.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("ETUDE_")
    }
    try:
        for index, command in enumerate(commands):
            with (out / f"child-{index}.log").open("ab") as log:
                children.append(
                    subprocess.Popen(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                )
        while any(child.poll() is None for child in children):
            if (
                stop.exists()
                or time.time() >= deadline
                or any(child.returncode not in (None, 0) for child in children)
            ):
                raise RuntimeError(
                    "cohort child failed, cancellation requested or allocation expired"
                )
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("cohort child failed")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic_json(
            out / "processes.json",
            {"exit_codes": [child.returncode for child in children]},
        )


def worker() -> None:
    """Etude owns the capped admission and all of its learner subprocesses."""
    bind_etude_source()
    StudyJob.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_bytes())
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    if deadline - time.time() < 2400:
        raise ValueError("remaining allocation cannot cover throughput admission")
    out.mkdir(parents=True, exist_ok=True)
    gate = check_numerics(out, min(time.time() + 300, deadline))
    if gate.status != "completed":
        raise RuntimeError("CUDA numerical admission failed")
    admit_throughput(out, stop, deadline)


class CurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: Arm
    result: MonitorResult


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


def plateau(points: list[CurvePoint]) -> PlateauResult:
    """Positive bounded-change evidence across seeds, not a nonsignificant slope.

    The fixed final window avoids choosing a favorable flat patch after seeing
    the curve. This conditional small-sample diagnostic is not equivalence of
    architectures or proof that no later improvement is possible.
    """
    scores: dict[tuple[Arm, int, int], float] = {}
    for item in points:
        result = item.result
        result.validate_cohort()
        key = item.arm, result.training_seed, result.coordinates.updates
        if (
            key in scores
            or result.training_seed not in SEEDS
            or result.coordinates.updates not in POINTS
            or result.status != "completed"
            or result.score is None
            or result.protocol != evaluation_protocol()
            or result.regime_digest
            != canonical_sha256(recipe(item.arm).model_dump(mode="json"))
        ):
            raise ValueError("incomplete, duplicate or undeclared curve point")
        scores[key] = result.score.mean
    expected = {
        (arm, seed, update) for arm in ARMS for seed in SEEDS for update in POINTS
    }
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
    parser.add_argument("--admit-arm", choices=ARMS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop", type=Path)
    args = parser.parse_args()
    if args.admit_arm:
        if args.out is None or args.stop is None:
            parser.error("admission needs output and stop path")
        admit_arm(args.admit_arm, args.out, args.stop)
    else:
        worker()


if __name__ == "__main__":
    main()
