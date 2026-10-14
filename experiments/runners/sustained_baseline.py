"""ETU-118 weekly-first masked-mean experiment through the shared runner.

Calibration measures the exact CPU recipe with recoverable collection. Freeze
admits the fixed multi-seed horizon from throughput and retained-storage bounds;
it never selects using scores. Run consumes that immutable plan. Daily regression
selection belongs to subsequent analysis, after sustained improvement exists.
"""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import socket

import numpy as np
from pydantic import Field, model_validator

from experiments.runners.sustained_baseline_result import (
    FinalReceipt,
    finalize,
    refresh,
)
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.experiment_execution import (
    ExperimentRun,
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
from manabot.training.experiment_runner import _runtime, run_experiment
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import (
    RecoveryPolicy,
    Strict,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import MonitorProtocol
from manabot.training.recovery import load_update
from manabot.verify.store import VerifyStore

SEEDS = (11851, 11852, 11853)
CALIBRATION_SEEDS = (11841, 11842, 11843)
CALIBRATION_UPDATES = 32
CALIBRATION_ALLOWANCE = 7200
CALIBRATION_LEARNER_SECONDS = 1800
STAGE_MILESTONES = (620, 1240, 2500, 5000, 7500, 10000)
SCORED_MILESTONES = (0, 400, 620, 800, 1240, 2500, 5000, 7500, 10000)
RANDOM_MONITORING_DEALS = tuple(range(1_911_185_000, 1_911_185_025))
MONITORING_DEALS = tuple(range(1_911_183_000, 1_911_183_025))
ENDPOINT_DEALS = tuple(range(1_911_184_000, 1_911_184_100))
MAX_ACTIVE_SECONDS = 7 * 86400
FINAL_RESERVE = 86400
REPORT_RESERVE = 3600
MONITOR_RESERVE = 86400
DISK_RESERVE = 4 * 1024**3


class SustainedPlan(Strict):
    regime: TrainingRegime
    schedule: ExperimentSchedule
    hardware: HardwareInventory
    runtime: dict[str, str]
    regime_digest: str
    analysis_sha256: str
    protocol_sha256: str
    declaration_sha256: str
    calibration_sha256: str
    preparation_seconds: float = Field(ge=0)
    calibrated_seconds_per_update: float = Field(gt=0)
    projected_bytes: int = Field(gt=0)
    milestones: tuple[int, ...]
    report_reserve_seconds: float = REPORT_RESERVE
    final_reserve_seconds: float = FINAL_RESERVE
    maximum_active_seconds: float = MAX_ACTIVE_SECONDS

    @model_validator(mode="after")
    def allocation(self) -> "SustainedPlan":
        if (
            self.maximum_active_seconds != MAX_ACTIVE_SECONDS
            or self.report_reserve_seconds != REPORT_RESERVE
            or self.final_reserve_seconds != FINAL_RESERVE
            or self.schedule.monitoring.seconds != MONITOR_RESERVE
            or not self.schedule.active_runtime
            or self.schedule.seeds != SEEDS
            or self.schedule.scientific_deal_seeds != ENDPOINT_DEALS
            or self.schedule.monitoring.protocol.deal_seeds != MONITORING_DEALS
            or self.milestones != STAGE_MILESTONES
            or self.schedule.checkpoint_updates != (400, 800)
            or self.schedule.diagnostic_updates != SCORED_MILESTONES
            or self.schedule.diagnostic_protocol
            != MonitorProtocol(
                deal_seeds=RANDOM_MONITORING_DEALS,
                opponent="random",
                game_seconds=60,
            )
            or abs(
                self.preparation_seconds
                + self.schedule.process_seconds
                + self.final_reserve_seconds
                - MAX_ACTIVE_SECONDS
            )
            > 1e-6
        ):
            raise ValueError(
                "sustained allocation differs from the fixed weekly protocol"
            )
        expected = declaration(self.regime, self.schedule).resolve().cases[0].regime
        if canonical_sha256(expected.model_dump(mode="json")) != self.regime_digest:
            raise ValueError("frozen regime identity differs")
        return self

    def admit(self) -> None:
        if (
            self.runtime != _runtime()
            or self.declaration_sha256 != file_sha256(Path(__file__))
            or self.analysis_sha256
            != file_sha256(Path(__file__).with_name("sustained_baseline_result.py"))
            or self.protocol_sha256
            != file_sha256(Path(__file__).resolve().parents[1] / "current-baseline.md")
        ):
            raise ValueError("frozen source, analysis, protocol or runtime changed")


def hardware() -> HardwareInventory:
    return HardwareInventory(
        resources=(
            Hardware(name="etu118-mini", host=socket.gethostname(), cpu_threads=2),
        )
    )


def recipe(milestones: tuple[int, ...], allowance: float) -> TrainingRegime:
    """Keep the candidate's architecture, estimator and absolute iteration schedules."""
    base = TrainingRegime.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "regimes/current-baseline-mini-source.json"
        ).read_text()
    )
    base.id = "current-baseline-masked-mean"
    base.agent.value_aggregation = "masked_mean"
    base.recovery = RecoveryPolicy(checkpoint_updates=128)
    base.wall_seconds = allowance
    template = deepcopy(base.stages[0])
    assert isinstance(template, TrainSelfPlay)
    base.stages = []
    previous = 0
    for milestone in milestones:
        if milestone <= previous:
            raise ValueError("milestones must increase")
        stage = deepcopy(template)
        stage.id = f"update-{milestone}"
        stage.initial = base.stages[-1].id if base.stages else None
        stage.updates = milestone - previous
        stage.execution.wall_seconds = allowance
        base.stages.append(stage)
        previous = milestone
    return TrainingRegime.model_validate(base.model_dump())


def declaration(regime: TrainingRegime, schedule: ExperimentSchedule) -> Experiment:
    return Experiment(
        name="current-baseline",
        baseline=Baseline.capture("masked-mean-current-self", regime),
        schedule=schedule,
    )


def calibration(out: Path, preparation_seconds: float) -> ExperimentRun:
    """One three-seed mini calibration inside the remaining original week."""
    if not math.isfinite(preparation_seconds) or preparation_seconds < 0:
        raise ValueError("preparation cost must be finite and nonnegative")
    if (
        preparation_seconds + CALIBRATION_ALLOWANCE
        >= MAX_ACTIVE_SECONDS - FINAL_RESERVE - MONITOR_RESERVE
    ):
        raise ValueError("remaining week cannot admit calibration")
    schedule = ExperimentSchedule(
        seeds=CALIBRATION_SEEDS,
        hardware="etu118-mini",
        wall_seconds=CALIBRATION_ALLOWANCE,
        process_seconds=CALIBRATION_ALLOWANCE,
        active_runtime=True,
        monitoring=MonitoringBudget(
            seconds=1200,
            attempt_seconds=200,
            active_runtime=True,
            include_initial=True,
            protocol=MonitorProtocol(deal_seeds=(1_911_182_000,), game_seconds=30),
        ),
        checkpoint_seconds=3600,
    )
    return run_experiment(
        declaration(
            recipe((CALIBRATION_UPDATES,), CALIBRATION_LEARNER_SECONDS), schedule
        ),
        hardware(),
        out,
    )


def freeze(
    calibration_root: Path, output: Path, preparation_seconds: float
) -> SustainedPlan:
    """Use timing/bytes only. No evaluation score enters allocation or horizon selection."""
    if output.exists():
        raise FileExistsError("a frozen plan cannot be overwritten")
    record_path = calibration_root / "experiment.json"
    exported = ExperimentRun.model_validate_json(record_path.read_text())
    if not (calibration_root / "experiment.sqlite").is_file():
        raise ValueError("calibration authority is unavailable")
    with VerifyStore(calibration_root / "experiment.sqlite") as store:
        record = store.experiment_run(exported.id)
        if record != exported:
            raise ValueError("calibration export differs from VerifyStore")
        runs = []
        for attempt in record.attempts:
            if attempt.run_id is None:
                raise ValueError("calibration attempt has no TrainingRun")
            run = store.training_run(attempt.run_id)
            if run != TrainingRun.model_validate_json(
                (Path(attempt.path) / "run.json").read_text()
            ):
                raise ValueError(
                    "calibration TrainingRun export differs from VerifyStore"
                )
            runs.append(run)
    if (
        record.status != "completed"
        or len(record.attempts) != 3
        or record.runtime != _runtime()
        or record.hardware != hardware().resources[0]
    ):
        raise ValueError("complete current-source calibration required")
    calibration_schedule = ExperimentSchedule.model_validate(record.intent["schedule"])
    expected = (
        declaration(
            recipe((CALIBRATION_UPDATES,), CALIBRATION_LEARNER_SECONDS),
            calibration_schedule,
        )
        .resolve()
        .cases[0]
        .regime
    )
    if tuple(r.seed for r in runs) != CALIBRATION_SEEDS or any(
        r.status != "completed"
        or r.regime != expected
        or r.updates_through() != CALIBRATION_UPDATES
        or sum(s.optimizer_exposures for s in r.stages) == 0
        for r in runs
    ):
        raise ValueError("calibration recipe, seeds or actual work differs")
    if preparation_seconds < max(record.elapsed_seconds, record.process_seconds):
        raise ValueError("preparation must include calibration and prior experiments")
    per_update = []
    snapshot_sizes = []
    json_sizes = []
    for run, attempt in zip(runs, record.attempts, strict=True):
        costs = np.array(
            [
                d["coordinates"]["training_seconds"]
                for s in run.stages
                for d in s.diagnostics
            ],
            dtype=float,
        )
        # Whole-run average includes startup/snapshots; upper-tail windows protect
        # against short calibration optimism. Later phase changes remain a risk.
        per_update.extend(
            (
                run.seconds / CALIBRATION_UPDATES,
                float(np.quantile(np.diff(costs), 0.95)),
            )
        )
        assert run.recovery_artifact is not None
        state_path = Path(run.recovery_artifact["path"])
        final_size = state_path.stat().st_size
        if (
            final_size != run.recovery_artifact["bytes"]
            or file_sha256(state_path) != run.recovery_artifact["sha256"]
        ):
            raise ValueError("retained calibration recovery bytes changed")
        with VerifyStore(calibration_root / "experiment.sqlite") as store:
            snapshot = load_update(run, store)
        if snapshot.format_version != 3:
            raise ValueError("compact-snapshot calibration required")
        # Stage-entry Adam is empty. Treat the entire admitted final snapshot as
        # fixed cost; diagnostics live once in the canonical run, not snapshots.
        # One MiB per snapshot additionally covers the bounded current-game journal.
        snapshot_sizes.append(final_size + 1024**2)
        json_sizes.append((Path(attempt.path) / "run.json").stat().st_size)
    conservative = 1.5 * max(per_update)
    learning = (
        MAX_ACTIVE_SECONDS - preparation_seconds - FINAL_RESERVE - MONITOR_RESERVE
    )
    per_seed = learning / len(SEEDS)
    free = shutil.disk_usage(output.parent).free - DISK_RESERVE
    # The protocol fixes the horizon; calibration admits it or fails, never
    # selects a shorter run. Retain snapshots and growing run/store/evaluator copies.
    updates = STAGE_MILESTONES[-1]
    ratio = updates / CALIBRATION_UPDATES
    checkpoints = math.ceil(updates / 128) + 30
    projected = int(
        len(SEEDS) * (checkpoints * max(snapshot_sizes) + 4 * max(json_sizes) * ratio)
        + 2 * 1024**3
    )
    if not (conservative * updates <= per_seed and projected <= free):
        raise ValueError(
            "no serious horizon fits calibrated time/storage; retain calibration and revise allocation/storage explicitly"
        )
    schedule = ExperimentSchedule(
        seeds=SEEDS,
        hardware="etu118-mini",
        active_runtime=True,
        wall_seconds=learning + MONITOR_RESERVE,
        process_seconds=learning + MONITOR_RESERVE,
        monitoring=MonitoringBudget(
            seconds=MONITOR_RESERVE,
            attempt_seconds=1800,
            active_runtime=True,
            include_initial=True,
            protocol=MonitorProtocol(deal_seeds=MONITORING_DEALS, game_seconds=60),
        ),
        diagnostic_protocol=MonitorProtocol(
            deal_seeds=RANDOM_MONITORING_DEALS,
            opponent="random",
            game_seconds=60,
        ),
        diagnostic_updates=SCORED_MILESTONES,
        checkpoint_updates=(400, 800),
        scientific_deal_seeds=ENDPOINT_DEALS,
        checkpoint_seconds=3600,
    )
    regime = recipe(STAGE_MILESTONES, per_seed)
    resolved = declaration(regime, schedule).resolve().cases[0].regime
    plan = SustainedPlan(
        regime_digest=canonical_sha256(resolved.model_dump(mode="json")),
        analysis_sha256=file_sha256(
            Path(__file__).with_name("sustained_baseline_result.py")
        ),
        protocol_sha256=file_sha256(
            Path(__file__).resolve().parents[1] / "current-baseline.md"
        ),
        regime=regime,
        schedule=schedule,
        hardware=hardware(),
        runtime=_runtime(),
        declaration_sha256=file_sha256(Path(__file__)),
        calibration_sha256=file_sha256(record_path),
        preparation_seconds=preparation_seconds,
        calibrated_seconds_per_update=conservative,
        projected_bytes=projected,
        milestones=STAGE_MILESTONES,
    )
    atomic_json(output, plan.model_dump(mode="json"))
    return plan


def run(
    plan_path: Path, out: Path, *, resume: bool = False, recover: tuple[int, ...] = ()
) -> ExperimentRun:
    plan = SustainedPlan.model_validate_json(plan_path.read_text())
    plan.admit()
    return run_experiment(
        declaration(plan.regime, plan.schedule),
        plan.hardware,
        out,
        resume=resume,
        recover=recover,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("out", type=Path)
    calibrate.add_argument("--preparation-seconds", required=True, type=float)
    select = commands.add_parser("freeze")
    select.add_argument("calibration", type=Path)
    select.add_argument("output", type=Path)
    select.add_argument("--preparation-seconds", required=True, type=float)
    execute = commands.add_parser("run")
    execute.add_argument("plan", type=Path)
    execute.add_argument("out", type=Path)
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--recover", type=int, action="append", default=[])
    report = commands.add_parser("report")
    report.add_argument("out", type=Path)
    final = commands.add_parser("finalize")
    final.add_argument("plan", type=Path)
    final.add_argument("out", type=Path)
    final.add_argument("--resume", action="store_true")
    pause = commands.add_parser("pause")
    pause.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        result = calibration(args.out, args.preparation_seconds)
        print(result.status)
        if result.status != "completed":
            raise SystemExit(1)
    elif args.command == "freeze":
        print(
            freeze(
                args.calibration, args.output, args.preparation_seconds
            ).model_dump_json(indent=2)
        )
    elif args.command == "report":
        print(refresh(args.out))
    elif args.command == "finalize":
        plan = SustainedPlan.model_validate_json(args.plan.read_text())
        print(finalize(plan, args.plan, args.out, resume=args.resume).status)
    elif args.command == "pause":
        record = ExperimentRun.model_validate_json(
            (args.out / "experiment.json").read_text()
        )
        final_path = args.out / "final-result.json"
        final_running = (
            final_path.exists()
            and FinalReceipt.model_validate_json(final_path.read_text()).status
            == "running"
        )
        if record.status != "running" and not final_running:
            raise ValueError("experiment is not running")
        (args.out / "pause.request").touch(exist_ok=True)
        print(
            "Pause requested; wait for experiment.json paused=true or final-result.json status=paused before shutdown."
        )
    else:
        result = run(
            args.plan, args.out, resume=args.resume, recover=tuple(args.recover)
        )
        print(result.status)
        if result.status != "completed":
            raise SystemExit(1)
        plan = SustainedPlan.model_validate_json(args.plan.read_text())
        print(
            finalize(
                plan,
                args.plan,
                args.out,
                resume=(args.out / "final-result.json").exists(),
            ).status
        )


if __name__ == "__main__":
    main()
