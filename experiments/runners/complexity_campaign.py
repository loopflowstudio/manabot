"""Frozen MTG-122 scientific cohort, executed only inside a bounded remote Etude job.

A plan is admitted from complete remote calibration, before scoring. The same
three-seed/four-cell cohort is never reduced to fit a deadline. Failure retains
all attempts and stops; it does not select a new seed or retry a learned run.
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Literal, Protocol

from etude.track.records import Step
from etude.track.run import Run as TrackerRun, init
from pydantic import Field, model_validator
import torch

from experiments.runners.baseline_monitoring import DiagnosticMonitor
from experiments.runners.environment_complexity import Pool, regime, roster
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import ObservationSpace
from manabot.model import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.etude_evaluation import publish_saved_evaluation
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiments import Baseline, Case, Environment, Experiment
from manabot.training.models import (
    Strict,
    TrainingCoordinates,
    TrainingRegime,
    TrainingRun,
)
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    MonitorResult,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore

SEEDS = (12251, 12252, 12253)
POOLS: tuple[Pool, ...] = ("l1", "l2", "l3")


class ComplexityPlan(Strict):
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

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return (0, self.updates // 2, self.updates)

    @property
    def digest(self) -> str:
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
        if milestone not in (0, self.updates // 2, self.updates):
            raise ValueError("unplanned milestone")
        ordinal = (0, self.updates // 2, self.updates).index(milestone)
        # Same within-world cohort across seeds; no claim of paired physical
        # deals between different worlds. Calibration/training namespaces differ.
        return tuple(
            range(1_225_100_000 + ordinal * 1000, 1_225_100_050 + ordinal * 1000)
        )

    def protocol(
        self, pool: Pool, anchor: Literal["scripted_greedy", "random"], update: int
    ) -> MonitorProtocol:
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
    def protocol(
        self, pool: Pool, anchor: Literal["scripted_greedy", "random"], update: int
    ) -> MonitorProtocol: ...


def evaluate_fit(
    plan: CampaignPlan,
    run: TrainingRun,
    directory: Path,
    pool: Pool,
    expected_digest: str,
    evaluation_tracker: TrackerRun,
    deadline: float,
) -> None:
    """Score a completed fit, including retained seed1 after a cadence handoff."""
    env = os.environ
    if run.status != "completed" or run.updates_through() != plan.updates:
        raise ValueError("evaluation requires the complete fixed target")
    initial = Checkpoint(
        artifact=run.stages[0].artifacts["initial_raw"],
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
    points = (
        initial,
        Checkpoint(
            artifact=middle.artifact,
            coordinates=TrainingCoordinates.model_validate(
                middle.model_dump(), extra="ignore"
            ),
        ),
        endpoint,
    )
    for point in points:
        milestone = point.coordinates.updates
        if milestone not in plan.evaluation_updates:
            continue
        for anchor in ("scripted_greedy", "random"):
            protocol = plan.protocol(pool, anchor, milestone)
            variants = (
                (("candidate", point),)
                if milestone == 0
                else (("candidate", point), ("control", initial))
            )
            for variant, checkpoint in variants:
                if (
                    time.time() >= deadline - 60
                    or Path(env["ETUDE_STOP_FILE"]).exists()
                ):
                    raise RuntimeError(
                        "allocation exhausted with incomplete evaluation cohort"
                    )
                result = evaluate_checkpoint(
                    run,
                    checkpoint.artifact,
                    checkpoint.coordinates,
                    directory / f"{milestone}-{anchor}-{variant}",
                    protocol=protocol,
                )
                publish_saved_evaluation(
                    evaluation_tracker,
                    directory / f"{milestone}-{anchor}-{variant}" / "monitor.json",
                )
                if result.status != "completed":
                    raise RuntimeError(
                        f"{run.seed}/{milestone}/{anchor}/{variant} incomplete"
                    )


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
    if isinstance(plan, GPUComplexityPlan):
        if not torch.cuda.is_available():
            raise ValueError("GPU campaign requires CUDA; no CPU fallback")
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
    evaluation_tracker = init(
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
                            else (plan.updates // 2,),
                            stop_path=Path(env["ETUDE_STOP_FILE"]),
                        )
                finally:
                    if monitor is not None:
                        monitor.finish()
                if frequent_diagnostics:
                    for evidence in sorted(
                        (directory / "diagnostics").glob(
                            "attempt-*/evaluation/monitor.json"
                        )
                    ):
                        publish_saved_evaluation(evaluation_tracker, evidence)
                if run.status != "completed":
                    raise RuntimeError(f"{key} failed: {run.error}")
                evaluate_fit(
                    plan,
                    run,
                    directory,
                    pool,
                    expected[seed],
                    evaluation_tracker,
                    deadline,
                )
                completed.append(key)
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
        evaluation_tracker.finish(
            "finished" if state["status"] == "completed" else "failed"
        )
        state["elapsed_seconds"] = time.time() - start
        atomic_json(manifest, state)
        env["ETUDE_OUTPUT_DIR"] = root_output
        env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"] = root_run, root_tracking


if __name__ == "__main__":
    payload = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())["plan"]
    plan_type = (
        GPUComplexityPlan if payload.get("schema_version") == 2 else ComplexityPlan
    )
    execute(plan_type.model_validate(payload))
