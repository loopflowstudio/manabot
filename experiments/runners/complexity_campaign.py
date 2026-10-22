"""Frozen MTG-122 scientific cohort, executed only inside a bounded remote Etude job.

A plan is admitted from complete remote calibration, before scoring. The same
three-seed/four-cell cohort is never reduced to fit a deadline. Failure retains
all attempts and stops; it does not select a new seed or retry a learned run.

Plans are versioned by `schema_version`: 1 is the historical Mini campaign, 2
the combined CUDA campaign, and 3 (`SplitComplexityPlan`) the current split:
the GPU job trains and retains checkpoints only, then an operator runs
`uv run python -m experiments.runners.complexity_campaign evaluate <campaign>` on the
Mini against the fetched outputs. Both halves score the same 36,000 games.
`--anchor heuristic` later scores the same cohort against the rule-of-thumb
anchor, a supplementary comparison added after the frozen results.
"""

import argparse
from collections.abc import Callable
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Annotated, Literal, Protocol

from etude.train.records import Step
from etude.train.run import track
from pydantic import Field, TypeAdapter, model_validator
import torch

from experiments.runners.baseline_monitoring import DiagnosticMonitor
from experiments.runners.environment_complexity import Pool, regime, roster
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import ObservationSpace
from manabot.model import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiments import Baseline, Case, Environment, Experiment
from manabot.training.models import (
    ArtifactReference,
    IterationSchedule,
    Strict,
    TrainingCoordinates,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    MonitorResult,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore
import managym

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (12251, 12252, 12253)
POOLS: tuple[Pool, ...] = ("l1", "l2", "l3")
Anchor = Literal["scripted_greedy", "heuristic", "random"]
# The frozen cohort's anchors; `heuristic` is opt-in and supplementary.
FROZEN_ANCHORS: tuple[Anchor, ...] = ("scripted_greedy", "random")
# MTG-135 protocol017's selected 64-stream recipe: 128 iterations of 64 streams
# x 64 transitions (524,288 learner transitions), with learning rate and tau
# following the first 2,048 iterations of the reference curve.
TESTED_SCHEDULE = IterationSchedule(steps=128, reference_steps=2048)


class _Cohort:
    """Deal families, matrices and resolved recipes shared by every plan version.

    Pydantic subclasses supply `updates`, `training_recipe` and `model_dump`;
    the scored milestones are initialization, midpoint and endpoint.
    """

    updates: int
    training_seconds_per_run: float

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return (0, self.updates // 2, self.updates)

    @property
    def digest(self) -> str:
        assert isinstance(self, Strict)
        return canonical_sha256(self.model_dump(mode="json"))

    def training_recipe(self, pool: Pool) -> TrainingRegime:
        return regime(pool, updates=self.updates, seconds=self.training_seconds_per_run)

    def experiment(self) -> Experiment:
        baseline = self.training_recipe("l1")
        return Experiment(
            name="mtg122",
            baseline=Baseline.capture("complexity", baseline),
            cases=tuple(
                Case(pool, (Environment(match=roster(pool).match()),)) for pool in POOLS
            ),
        )

    def recipes(self) -> dict[str, TrainingRegime]:
        return {
            case.name.removeprefix("mtg122-"): case.regime
            for case in self.experiment().resolve().cases
        }

    def deals(self, milestone: int) -> tuple[int, ...]:
        if milestone not in self.evaluation_updates:
            raise ValueError("unplanned milestone")
        ordinal = self.evaluation_updates.index(milestone)
        # Same within-world cohort across seeds; no claim of paired physical
        # deals between different worlds. Calibration/training namespaces differ.
        return tuple(
            range(1_225_100_000 + ordinal * 1000, 1_225_100_050 + ordinal * 1000)
        )

    def protocol(self, pool: Pool, anchor: Anchor, update: int) -> MonitorProtocol:
        return MonitorProtocol(
            authored_matchup=roster(pool),
            include_mirrors=True,
            deal_seeds=self.deals(update),
            opponent=anchor,
            purpose="frozen-study-evaluation",
            study_protocol_sha256=self.digest,
            bootstrap_seed=12290,
            bootstrap_replicates=10000,
        )


class ComplexityPlan(_Cohort, Strict):
    """Exact counts and calibration evidence; no implicit machine allocation."""

    schema_version: Literal[1] = 1
    updates: int = Field(ge=1000, le=10000)
    seed_order: tuple[int, int, int] = SEEDS
    training_seconds_per_run: float = Field(gt=0, le=7200)
    total_seconds: float = Field(gt=0, le=47 * 3600)
    evaluation_reserve_seconds: float = Field(gt=0)
    calibration_hashes: dict[str, str]
    deals_per_checkpoint: Literal[50] = 50
    margin: Literal[1.5] = 1.5

    @model_validator(mode="after")
    def fixed_cohort(self) -> "ComplexityPlan":
        if self.updates % 100 or self.seed_order != SEEDS:
            raise ValueError(
                "cohort requires fixed three seeds and 100-update granularity"
            )
        if len(self.calibration_hashes) < 9:
            raise ValueError(
                "all three training and six anchor calibration receipts required"
            )
        if (
            9 * self.training_seconds_per_run + self.evaluation_reserve_seconds + 600
            > self.total_seconds
        ):
            raise ValueError(
                "cohort exceeds inclusive allocation with setup/publication reserve"
            )
        return self


class GPUComplexityPlan(ComplexityPlan):
    """Versioned CUDA contract, separate from historical Mini recipe identities.

    Provider price includes rental hardware/storage; artifact/settlement reserve
    is additional. Actual calibration charges remain visible even on failure.
    Launcher limits must be no larger than these admitted seconds and dollars.
    """

    schema_version: Literal[2] = 2
    streams: Literal[64] = 64
    device: Literal["cuda"] = "cuda"
    max_usd_per_hour: float = Field(gt=0)
    calibration_usd: float = Field(ge=0, le=2)
    settlement_reserve_usd: Literal[3] = 3
    inclusive_cap_usd: Literal[35] = 35

    @model_validator(mode="after")
    def spending_cap(self) -> "GPUComplexityPlan":
        rental = self.total_seconds / 3600 * self.max_usd_per_hour
        if (
            rental > 30
            or rental + self.calibration_usd + self.settlement_reserve_usd > 35
        ):
            raise ValueError("GPU cohort exceeds the inclusive $35 allocation")
        return self

    def training_recipe(self, pool: Pool) -> TrainingRegime:
        return regime(
            pool,
            updates=self.updates,
            seconds=self.training_seconds_per_run,
            streams=self.streams,
            device=self.device,
        )


class SplitComplexityPlan(_Cohort, Strict):
    """CUDA training of the tested 64x2 recipe; the Mini evaluates its exports.

    Horizon, schedule and per-run watchdog are MTG-135 protocol017's tested
    64-stream values rather than a calibrated fit to a budget. Inline diagnostic
    evaluation is off; the scored cohorts (initialization, midpoint, endpoint,
    frozen controls) are unchanged and run afterwards on the Mini. The quarter
    export is retained but not scored. `prior_usd` is GPU spending already
    settled or reserved against the inclusive cap.
    """

    schema_version: Literal[3] = 3
    updates: Literal[128] = 128
    seed_order: tuple[int, int, int] = SEEDS
    training_seconds_per_run: Literal[1740] = 1740
    total_seconds: float = Field(gt=0)
    evaluation_reserve_seconds: Literal[0] = 0
    evaluation_placement: Literal["mini"] = "mini"
    streams: Literal[64] = 64
    device: Literal["cuda"] = "cuda"
    max_usd_per_hour: float = Field(gt=0, le=0.69)
    prior_usd: float = Field(ge=0)
    settlement_reserve_usd: Literal[3] = 3
    inclusive_cap_usd: Literal[35] = 35
    deals_per_checkpoint: Literal[50] = 50

    @model_validator(mode="after")
    def admit(self) -> "SplitComplexityPlan":
        if self.seed_order != SEEDS:
            raise ValueError("cohort requires its fixed three seeds")
        if 9 * self.training_seconds_per_run + 600 > self.total_seconds:
            raise ValueError("nine bounded fits plus setup exceed the allocation")
        rental = self.total_seconds / 3600 * self.max_usd_per_hour
        if rental + self.prior_usd + self.settlement_reserve_usd > 35:
            raise ValueError("GPU training exceeds the inclusive $35 allocation")
        return self

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        return (self.updates // 4, self.updates // 2)

    def training_recipe(self, pool: Pool) -> TrainingRegime:
        return regime(
            pool,
            updates=self.updates,
            seconds=self.training_seconds_per_run,
            streams=self.streams,
            device=self.device,
            schedule=TESTED_SCHEDULE,
        )


AnyComplexityPlan = ComplexityPlan | GPUComplexityPlan | SplitComplexityPlan


def read_complexity_plan(payload: str) -> AnyComplexityPlan:
    """Read an exact versioned plan without adapting historical recipe meaning."""
    adapter: TypeAdapter[AnyComplexityPlan] = TypeAdapter(
        Annotated[AnyComplexityPlan, Field(discriminator="schema_version")]
    )
    return adapter.validate_json(payload)


def calibrated_plan(
    calibration: Path,
    *,
    total_seconds: float,
    gpu_max_usd_per_hour: float | None = None,
    calibration_usd: float = 0,
) -> ComplexityPlan:
    """Admit unchanged cohort using 1.5× observed worst block/cell costs.

    Sparse calibration extrapolates, not guarantees. The remote guardian and
    retained failures still bound execution if long-run rates deteriorate.
    """
    hashes: dict[str, str] = {}
    update_costs: list[float] = []
    evaluation_costs: list[float] = []
    for pool in POOLS:
        path = calibration / pool / "training/run.json"
        run = TrainingRun.model_validate_json(path.read_text())
        if (
            run.status != "completed"
            or run.seed != 12250
            or run.updates_through() != 100
        ):
            raise ValueError(f"incomplete calibration training: {pool}")
        expected = regime(
            pool,
            updates=100,
            seconds=1200 if gpu_max_usd_per_hour is not None else 600,
            streams=64 if gpu_max_usd_per_hour is not None else 4,
            device="cuda" if gpu_max_usd_per_hour is not None else "cpu",
        )
        if run.regime != expected:
            raise ValueError(f"calibration recipe differs: {pool}")
        hashes[str(path.relative_to(calibration))] = file_sha256(path)
        clocks = [0.0] + [
            float(d["coordinates"]["active_training_seconds"])
            for d in run.stages[0].diagnostics
        ]
        update_costs.extend(
            (clocks[n] - clocks[n - 20]) / 20 for n in range(20, 101, 20)
        )
        for anchor in ("scripted_greedy", "random"):
            path = calibration / pool / anchor / "monitor.json"
            result = MonitorResult.model_validate_json(path.read_text())
            result.validate_cohort()
            if result.status != "completed" or len(result.rows) != 8:
                raise ValueError(f"incomplete calibration evaluation: {pool}/{anchor}")
            hashes[str(path.relative_to(calibration))] = file_sha256(path)
            # Charge the full coordinator/replay cost; also protect slow role cells.
            panel_costs = [result.evaluation_seconds / len(result.rows)]
            by_cell = ((0, 3), (1, 2), (4, 5), (6, 7))
            panel_costs.extend(
                sum(row.game_seconds for row in result.rows if row.leg in legs) / 2
                for legs in by_cell
            )
            evaluation_costs.append(max(panel_costs))
    slowest = max(update_costs)
    if slowest <= 0:
        raise ValueError("calibration has no positive measured update cost")
    updates = min(10000, int((7200 - 120) / (1.5 * slowest)) // 100 * 100)
    plan_type = ComplexityPlan if gpu_max_usd_per_hour is None else GPUComplexityPlan
    fields: dict[str, object] = {}
    if gpu_max_usd_per_hour is not None:
        fields.update(
            max_usd_per_hour=gpu_max_usd_per_hour, calibration_usd=calibration_usd
        )
    return plan_type(
        **fields,
        updates=updates,
        training_seconds_per_run=7200,
        total_seconds=total_seconds,
        evaluation_reserve_seconds=6000 * 1.5 * sum(evaluation_costs),
        calibration_hashes=hashes,
    )


def parameter_digest(agent: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in agent.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CampaignPlan(Protocol):
    """A frozen recipe cohort consumed by the shared arena execution path."""

    updates: int
    seed_order: tuple[int, int, int]
    training_seconds_per_run: float
    evaluation_reserve_seconds: float

    @property
    def evaluation_updates(self) -> tuple[int, ...]: ...
    @property
    def digest(self) -> str: ...
    def model_dump(self, *, mode: str) -> dict[str, object]: ...
    def experiment(self) -> Experiment: ...
    def recipes(self) -> dict[str, TrainingRegime]: ...
    def protocol(self, pool: Pool, anchor: Anchor, update: int) -> MonitorProtocol: ...


def _relocated(artifact: ArtifactReference, training: Path | None) -> ArtifactReference:
    """Point an artifact at its fetched copy; evaluation re-verifies its bytes."""
    if training is None:
        return artifact
    return {**artifact, "path": str(training / Path(artifact["path"]).name)}


def fit_checkpoints(
    plan: CampaignPlan,
    run: TrainingRun,
    expected_digest: str,
    training: Path | None = None,
) -> tuple[Checkpoint, ...]:
    """Initialization, midpoint and endpoint exports of one completed fit.

    `training` relocates artifacts recorded on another host (the GPU worker)
    to their fetched copies; every scored checkpoint is still hash-checked.
    """
    if run.status != "completed" or run.updates_through() != plan.updates:
        raise ValueError("evaluation requires the complete fixed target")
    initial = Checkpoint(
        artifact=_relocated(run.stages[0].artifacts["initial_raw"], training),
        coordinates=TrainingCoordinates(
            stage_id="initialization", updates=0, training_seconds=0
        ),
    )
    saved_initial, _ = load_checkpoint_agent(initial.artifact["path"])
    if parameter_digest(saved_initial) != expected_digest:
        raise ValueError("actual producer initialization differs from paired preflight")
    middle = next(
        c for c in run.monitoring_checkpoints if c.updates == plan.updates // 2
    )
    if middle.artifact is None:
        raise RuntimeError("midpoint export unavailable")
    endpoint = stage_checkpoint(run, run.stages[0].id)
    if endpoint is None:
        raise RuntimeError("endpoint export unavailable")
    return (
        initial,
        Checkpoint(
            artifact=_relocated(middle.artifact, training),
            coordinates=TrainingCoordinates.model_validate(
                middle.model_dump(), extra="ignore"
            ),
        ),
        Checkpoint(
            artifact=_relocated(endpoint.artifact, training),
            coordinates=endpoint.coordinates,
        ),
    )


def evaluate_fit(
    plan: CampaignPlan,
    run: TrainingRun,
    directory: Path,
    pool: Pool,
    expected_digest: str,
    should_stop: Callable[[], bool],
    *,
    training: Path | None = None,
    anchors: tuple[Anchor, ...] = FROZEN_ANCHORS,
) -> int:
    """Score a completed fit's five cohorts per anchor; return cells newly run.

    A cell whose retained result completed is kept, not rerun, so an interrupted
    manual evaluation resumes. Any other retained cell attempt stops evaluation:
    failures are never overwritten.
    """
    initial, *later = fit_checkpoints(plan, run, expected_digest, training)
    evaluated = 0
    for point in (initial, *later):
        milestone = point.coordinates.updates
        if milestone not in plan.evaluation_updates:
            continue
        for anchor in anchors:
            protocol = plan.protocol(pool, anchor, milestone)
            variants = (
                (("candidate", point),)
                if milestone == 0
                else (("candidate", point), ("control", initial))
            )
            for variant, checkpoint in variants:
                cell = directory / f"{milestone}-{anchor}-{variant}"
                if cell.exists():
                    retained = cell / "monitor.json"
                    if retained.exists():
                        prior = MonitorResult.model_validate_json(retained.read_text())
                        if prior.status == "completed":
                            prior.validate_cohort()
                            continue
                    raise RuntimeError(f"{cell} retains an incomplete attempt")
                if should_stop():
                    raise RuntimeError(
                        "allocation exhausted with incomplete evaluation cohort"
                    )
                result = evaluate_checkpoint(
                    run,
                    checkpoint.artifact,
                    checkpoint.coordinates,
                    cell,
                    protocol=protocol,
                )
                if result.status != "completed":
                    raise RuntimeError(
                        f"{run.seed}/{milestone}/{anchor}/{variant} incomplete"
                    )
                evaluated += 1
    return evaluated


def allocation_stop(deadline: float) -> Callable[[], bool]:
    """Inside an Etude allocation: stop near its deadline or on cancellation."""
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    return lambda: time.time() >= deadline - 60 or stop.exists()


def evaluate_retained(
    campaign: Path,
    *,
    pools: tuple[Pool, ...] = POOLS,
    seeds: tuple[int, ...] = SEEDS,
    anchors: tuple[Anchor, ...] = FROZEN_ANCHORS,
) -> None:
    """Score fetched split-plan training outputs on the Mini, in place.

    Run manually by an operator, after the GPU job's outputs are fetched into
    `campaign` with their original layout. Results land where the offline report
    reads them. Interrupting is safe; rerunning skips completed cells.
    """
    if not socket.gethostname().startswith("mini-heart"):
        raise ValueError("scientific evaluation runs on the Mini; local forbidden")
    plan = read_complexity_plan((campaign / "plan.json").read_text())
    if not isinstance(plan, SplitComplexityPlan):
        raise ValueError("only split plans defer evaluation to the Mini")
    expected = json.loads((campaign / "initialization-parameters.json").read_text())
    recipes = plan.recipes()
    for seed in seeds:
        for pool in pools:
            directory = campaign / f"{pool}-{seed}"
            run = TrainingRun.model_validate_json(
                (directory / "training/run.json").read_text()
            )
            if run.seed != seed or run.regime != recipes[pool]:
                raise ValueError(f"{pool}-{seed} differs from the frozen plan")
            count = evaluate_fit(
                plan,
                run,
                directory,
                pool,
                expected[str(seed)],
                lambda: False,
                training=directory / "training",
                anchors=anchors,
            )
            print(f"{pool}-{seed}: {count} cells evaluated", flush=True)


def execute(
    plan: CampaignPlan,
    *,
    pools: tuple[Pool, ...] = POOLS,
    seeds: tuple[int, ...] | None = None,
    frequent_diagnostics: bool = False,
) -> None:
    """Execute all declared runs/cohorts; launcher's guardian owns hard stopping."""
    selected = plan.seed_order if seeds is None else seeds
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(plan.seed_order)
    ):
        raise ValueError("selected seeds must be an explicit nonempty cohort subset")
    if frequent_diagnostics and pools != ("l3",):
        raise ValueError("baseline diagnostics require the product matchup")
    env = os.environ
    if not env.get("ETUDE_RUN"):
        raise ValueError("scientific campaign requires an Etude remote allocation")
    split = isinstance(plan, SplitComplexityPlan)
    if isinstance(plan, GPUComplexityPlan | SplitComplexityPlan):
        if not torch.cuda.is_available():
            raise ValueError("GPU campaign requires CUDA; no CPU fallback")
        if managym.WORLD_VERSION != "w6":
            raise ValueError("GPU complexity study requires w6 native runtime")
    elif not socket.gethostname().startswith("mini-heart"):
        raise ValueError("historical campaign requires Mini, not local execution")
    deadline = datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp()
    if deadline - time.time() < (
        len(pools) * len(selected) * plan.training_seconds_per_run
        + plan.evaluation_reserve_seconds * len(selected) / len(plan.seed_order)
        + (9900 * len(selected) if frequent_diagnostics else 0)
        + 300
    ):
        raise ValueError("remaining SDK allocation cannot admit the frozen campaign")
    output = Path(env["ETUDE_OUTPUT_DIR"])
    manifest = output / "campaign.json"
    if manifest.exists():
        raise ValueError("campaign evidence already exists")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan.model_dump(mode="json"))
    atomic_json(
        output / "experiment.json",
        {c.name: c.receipt() for c in plan.experiment().resolve().cases},
    )
    root_run, root_tracking = env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"]
    root_output = env["ETUDE_OUTPUT_DIR"]
    start = time.time()
    evaluation_tracker = (
        None
        if split
        else track(
            experiment="mtg122-evaluation",
            clocks=[
                Step(
                    "completed_runs",
                    "Fits with all declared training and evaluation completed",
                )
            ],
            run=f"{root_run}-evaluations",
            directory=Path(root_tracking) / "evaluations",
            config=plan.model_dump(mode="json"),
            strict=True,
        )
    )
    completed: list[str] = []
    state: dict[str, object] = {
        "status": "running",
        "plan_sha256": plan.digest,
        "started_unix": start,
        "completed": completed,
        "selected_seeds": selected,
        "frequent_diagnostics": frequent_diagnostics,
    }
    atomic_json(manifest, state)
    try:
        recipes = plan.recipes()
        expected: dict[int, str] = {}
        for seed in selected:
            for pool in pools:
                # Orthogonal initialization runs a QR decomposition whose x86
                # MKL result depends on the intra-op thread count, so match the
                # producer's setting before predicting its initial tensors.
                stage = recipes[pool].stages[0]
                assert isinstance(stage, TrainSelfPlay)
                torch.set_num_threads(stage.execution.threads)
                torch.manual_seed(seed)
                initial_model = Agent(
                    ObservationSpace(recipes[pool].observation), recipes[pool].agent
                )
                digest = parameter_digest(initial_model)
                if seed in expected and expected[seed] != digest:
                    raise ValueError("pool initialization changes learnable tensors")
                expected[seed] = digest
        atomic_json(output / "initialization-parameters.json", expected)
        for seed_index, seed in enumerate(selected):
            offset = seed_index % len(pools)
            order = pools[offset:] + pools[:offset]
            for pool in order:
                if deadline - time.time() < plan.training_seconds_per_run + 60:
                    raise RuntimeError("allocation exhausted before next declared run")
                key = f"{pool}-{seed}"
                directory = output / key
                directory.mkdir()
                env["ETUDE_OUTPUT_DIR"] = str(directory)
                env["ETUDE_RUN"] = f"{root_run}-{key}"
                env["ETUDE_TRACK_DIR"] = str(Path(root_tracking) / key)
                monitor = None
                if frequent_diagnostics:
                    monitor = DiagnosticMonitor(
                        directory / "training/run.json", directory / "diagnostics"
                    )
                    monitor.start()
                try:
                    with VerifyStore(directory / "training.sqlite") as store:
                        run = execute_regime(
                            recipes[pool],
                            seed,
                            directory / "training",
                            store,
                            checkpoint_updates=500
                            if frequent_diagnostics
                            else plan.checkpoint_updates
                            if isinstance(plan, SplitComplexityPlan)
                            else (plan.updates // 2,),
                            stop_path=Path(env["ETUDE_STOP_FILE"]),
                        )
                finally:
                    if monitor is not None:
                        monitor.finish()
                if run.status != "completed":
                    raise RuntimeError(f"{key} failed: {run.error}")
                if split:
                    # Exports are checked now so a missing one fails on the GPU.
                    fit_checkpoints(plan, run, expected[seed])
                else:
                    evaluate_fit(
                        plan,
                        run,
                        directory,
                        pool,
                        expected[seed],
                        allocation_stop(deadline),
                    )
                completed.append(key)
                if evaluation_tracker is not None:
                    evaluation_tracker.log(completed_runs=len(completed))
                state["elapsed_seconds"] = time.time() - start
                atomic_json(manifest, state)
        state["status"] = "completed"
        atomic_json(
            output / "etude-result.json",
            {"clocks": {"completed_runs": len(pools) * len(selected)}},
        )
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if evaluation_tracker is not None:
            evaluation_tracker.finish(
                "finished" if state["status"] == "completed" else "failed"
            )
        state["elapsed_seconds"] = time.time() - start
        atomic_json(manifest, state)
        env["ETUDE_OUTPUT_DIR"] = root_output
        env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"] = root_run, root_tracking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    evaluate = commands.add_parser("evaluate", help="score fetched outputs on Mini")
    evaluate.add_argument("campaign", type=Path)
    evaluate.add_argument("--pool", choices=POOLS, action="append")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, action="append")
    evaluate.add_argument(
        "--anchor", choices=("scripted_greedy", "heuristic", "random"), action="append"
    )
    arguments = parser.parse_args()
    if arguments.command == "evaluate":
        evaluate_retained(
            arguments.campaign,
            pools=tuple(arguments.pool or POOLS),
            seeds=tuple(arguments.seed or SEEDS),
            anchors=tuple(arguments.anchor or FROZEN_ANCHORS),
        )
    else:
        # Etude worker entry: the job config carries the frozen plan.
        payload = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())["plan"]
        execute(read_complexity_plan(json.dumps(payload)))
