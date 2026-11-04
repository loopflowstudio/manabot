"""Prospective GPU horizon study, separate from the frozen v7 CPU cohort.

One Etude allocation owns eight independent TrainingRuns. execute_regime owns
learning and checkpoints; the ordinary monitor owns
complete games, viewer admission and replay. Curves use one deal set, and final
held-out scoring uses another. No score changes a replicate's declared horizon.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from etude.launch.records import Job, Run
from etude.report import Report
from pydantic import BaseModel, ConfigDict
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
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

EXPERIMENT = "state_representation_gpu"
SEEDS = (74201, 74202, 74203, 74204)
UPDATES = 2048
CADENCE = 256
POINTS = tuple(range(0, UPDATES + 1, CADENCE))
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
    """One paid allocation for the complete independent-seed cohort."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle


def runs(revision: str) -> list[Run]:
    """One supervised rental; child TrainingRuns retain independent identities."""
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
        until=f"updates={UPDATES}",
        max_hours=20,
        max_usd=Decimal("17.20"),
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


def checkpoints(run: TrainingRun) -> list[Checkpoint]:
    """Only declared coordinates; a wall stop is retained, never called an endpoint."""
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
        if saved.updates not in POINTS:
            continue
        if saved.artifact is None or saved.error is not None:
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
    if final is not None and final.coordinates.updates == UPDATES:
        points.append(final)
    return sorted(points, key=lambda point: point.coordinates.updates)


def train_arm(arm: Arm, seed: int, out: Path, stop: Path) -> None:
    """An independent ordinary learner; no shared weights, optimizer or RNG."""
    if seed not in SEEDS:
        raise ValueError("undeclared training seed")
    if not torch.cuda.is_available():
        raise RuntimeError("this study requires CUDA")
    # Eight contexts share one GPU. Leave 20% outside their caching allocators
    # for driver/cuDNN overhead; capacity failure remains a recorded failure.
    torch.cuda.set_per_process_memory_fraction(0.10)
    out.mkdir(parents=True, exist_ok=False)
    with VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(
            recipe(arm),
            seed,
            out / "training",
            store,
            experiment=EXPERIMENT,
            checkpoint_updates=POINTS[1:-1],
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
    if run.status != "completed" or run.updates_through() != UPDATES:
        raise RuntimeError(f"incomplete training: {run.status}: {run.error}")


def evaluate_arm(arm: Arm, seed: int, out: Path, deadline: float) -> None:
    run = TrainingRun.model_validate_json((out / "training/run.json").read_bytes())
    if run.seed != seed or run.regime_digest != canonical_sha256(
        recipe(arm).model_dump(mode="json")
    ):
        raise ValueError("run is outside the declared GPU cohort")
    deadline = min(deadline - 120, time.time() + 1200)
    results: list[MonitorResult] = []
    for point in checkpoints(run):
        update = point.coordinates.updates
        for heldout in (False, True) if update == UPDATES else (False,):
            result = evaluate_checkpoint(
                run,
                point.artifact,
                point.coordinates,
                out / ("heldout" if heldout else f"curve-{update:04d}"),
                protocol=evaluation_protocol(heldout=heldout),
                concurrent_activity="all training finished; at most two cohort evaluators active",
                execution=EvalExecution(
                    device="cpu",
                    max_active_games=4,
                    max_batch_rows=32,
                    resident_model_bytes=1024**3,
                    deadline_unix=deadline,
                ),
            )
            results.append(result)
            atomic_json(
                out / "curve.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "results": [item.model_dump(mode="json") for item in results],
                },
            )
            if result.status != "completed":
                raise RuntimeError(
                    "evaluation failed; retained in the study denominator"
                )


def _batch(commands: list[list[str]], out: Path, stop: Path, deadline: float) -> None:
    """Bound child lifetimes by the shared rental, and reap every child on exit."""
    children: list[subprocess.Popen[bytes]] = []
    # Each child creates its own local tracking run under its retained output.
    # Passing the parent's launcher identity would attach all eight to one run.
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
    """The Etude allocation owns all children and stops on any failed replicate."""
    bind_etude_source()
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    deadline = datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
    if deadline - time.time() < 71400:
        raise ValueError("remaining allocation cannot cover training and evaluation")
    out.mkdir(parents=True, exist_ok=True)
    gate = check_numerics(out, min(time.time() + 300, deadline))
    if gate.status != "completed":
        raise RuntimeError("CUDA numerical admission failed")
    assignments = [
        (arm, seed)
        for index, seed in enumerate(SEEDS)
        for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
    ]
    base = [sys.executable, "-m", "experiments.state_representation_gpu"]
    training = [
        base
        + [
            "--train-arm",
            arm,
            "--seed",
            str(seed),
            "--out",
            str(out / f"{arm}-{seed}"),
            "--stop",
            str(stop),
        ]
        for arm, seed in assignments
    ]
    _batch(training, out, stop, min(deadline - 5400, time.time() + 66400))
    for index in range(0, len(assignments), 2):
        evaluation = [
            base
            + [
                "--evaluate-arm",
                arm,
                "--seed",
                str(seed),
                "--out",
                str(out / f"{arm}-{seed}"),
                "--deadline",
                str(deadline),
            ]
            for arm, seed in assignments[index : index + 2]
        ]
        phase = out / f"evaluation-{index // 2}"
        phase.mkdir()
        _batch(evaluation, phase, stop, deadline - 120)
    curves: list[CurvePoint] = []
    final: dict[tuple[Arm, int], float] = {}
    for arm, seed in assignments:
        document = json.loads((out / f"{arm}-{seed}" / "curve.json").read_text())
        for raw in document["results"]:
            result = MonitorResult.model_validate(raw)
            if result.protocol == evaluation_protocol(heldout=True):
                result.validate_cohort()
                if result.status != "completed" or result.score is None:
                    raise ValueError("incomplete held-out scoring")
                final[arm, seed] = result.score.mean
            else:
                curves.append(CurvePoint(arm=arm, result=result))
    estimate = paired_interval(
        tuple(final["typed_gru", seed] - final["typed_pool", seed] for seed in SEEDS)
    )
    atomic_json(
        out / "comparison.json",
        {"plateau": asdict(plateau(curves)), "heldout": asdict(estimate)},
    )
    atomic_json(out / "etude-result.json", {"clocks": {"updates": UPDATES}})


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-arm", choices=ARMS)
    mode.add_argument("--evaluate-arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.train_arm:
        if args.seed is None or args.out is None or args.stop is None:
            parser.error("training needs seed, output and stop path")
        train_arm(args.train_arm, args.seed, args.out, args.stop)
    elif args.evaluate_arm:
        if args.seed is None or args.out is None or args.deadline is None:
            parser.error("evaluation needs seed, output and deadline")
        evaluate_arm(args.evaluate_arm, args.seed, args.out, args.deadline)
    else:
        worker()


if __name__ == "__main__":
    main()
