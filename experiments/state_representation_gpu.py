"""Prospective GPU horizon study, separate from the frozen v7 CPU cohort.

TrainingJob owns training, checkpoints and watchdogs; the ordinary monitor owns
complete games, viewer admission and replay. Curves use one deal set, and final
held-out scoring uses another. No score changes a replicate's declared horizon.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import time

from etude.launch.records import Job, Run
from etude.report import Report
from pydantic import BaseModel, ConfigDict

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
from manabot.experiment.job import TOOLCHAIN_ENV, TrainingJob, machine_setup, run_worker
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.recipes.builders import move_learning
from manabot.sim.execution import EvalExecution
from manabot.training.panels import ML, PERF
from manabot.training.records import TrainingCoordinates, TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay

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
    stage.execution.memory_bytes = 32 * 1024**3
    stage.execution.wall_seconds = 6000
    regime.wall_seconds = 6000
    return TrainingRegime.model_validate(regime.model_dump())


def runs(revision: str) -> list[Run]:
    """Declare eight attempts; allocation includes setup, evaluation and delivery."""
    jobs: dict[Arm, Job] = {}
    for arm in ARMS:
        job = TrainingJob(
            regime=recipe(arm),
            experiment=EXPERIMENT,
            milestones=POINTS[1:-1],
            finalization_seconds=1200,
            validate_numerics=True,
        ).job(
            repo=Path(__file__).resolve().parents[1],
            revision=revision,
            max_hours=2.0,
            max_usd=Decimal("2.15"),
        )
        jobs[arm] = job.model_copy(
            update={
                "setup": machine_setup(
                    " --extra dev --extra artifacts --extra evaluation"
                ),
                "command": f"{TOOLCHAIN_ENV}\nuv run --locked --extra dev --extra artifacts --extra evaluation python -m experiments.state_representation_gpu",
            }
        )
    return [
        run
        for index, seed in enumerate(SEEDS)
        for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
        for run in jobs[arm].runs(seed)
    ]


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


def worker() -> None:
    """Train and evaluate through existing owners inside the rental deadline."""
    run = run_worker()
    arm = run.regime.agent.state_encoder
    if run.seed not in SEEDS or run.regime_digest != canonical_sha256(
        recipe(arm).model_dump(mode="json")
    ):
        raise ValueError("run is outside the declared GPU cohort")
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    deadline = min(
        datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp() - 120,
        time.time() + 1200,
    )
    results = []
    for point in checkpoints(run):
        update = point.coordinates.updates
        for heldout in (False, True) if update == UPDATES else (False,):
            result = evaluate_checkpoint(
                run,
                point.artifact,
                point.coordinates,
                out / ("heldout" if heldout else f"curve-{update:04d}"),
                protocol=evaluation_protocol(heldout=heldout),
                concurrent_activity="this replicate finished training; other declared GPU replicates may be active",
                execution=EvalExecution(
                    device="cpu",
                    max_active_games=4,
                    max_batch_rows=32,
                    resident_model_bytes=1024**3,
                    deadline_unix=deadline,
                ),
            )
            results.append(result.model_dump(mode="json"))
            atomic_json(
                out / "curve.json", {"arm": arm, "seed": run.seed, "results": results}
            )
            if result.status != "completed":
                raise RuntimeError(
                    "evaluation failed; retained in the study denominator"
                )


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


if __name__ == "__main__":
    worker()
