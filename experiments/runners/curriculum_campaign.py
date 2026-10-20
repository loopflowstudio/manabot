"""MTG-122 curriculum follow-up: does a quarter of simple games speed L3 learning?

Each seed trains two arms on the tested 64x2 recipe for 128 absolute
iterations. Both arms run iterations 1-32 as a source segment and 33-128 as an
L3 continuation that carries raw weights, EMA, Adam and the schedule clock;
only game streams restart. The arms differ only in the source decks:

- `curriculum`: L1 simple creatures and lands, then a declared setup transfer.
- `scratch`: L3 throughout, through the same continuation boundary.

The GPU job (`execute`) trains and retains exports only. An operator then runs
`python -m experiments.runners.curriculum_campaign evaluate <campaign>` inside a
rented CPU allocation. Initialization, 64 and 128 reuse the frozen complexity
plan's L3 protocols exactly, so arm-C (continuous scratch) cells there are
directly comparable; 32 has its own disjoint deal family. Design and decisions:
`experiments/curriculum-design.md`.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Literal

from pydantic import Field, model_validator
import torch

from experiments.runners.complexity_campaign import (
    TESTED_SCHEDULE,
    Anchor,
    SplitComplexityPlan,
    allocation_stop,
    parameter_digest,
    read_complexity_plan,
)
from experiments.runners.environment_complexity import Pool, regime, roster
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import ObservationSpace
from manabot.model import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.checkpoint_queue import checkpoints
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import (
    ArtifactReference,
    IterationSchedule,
    LearningStateImport,
    Strict,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    MonitorResult,
    evaluate_checkpoint,
)
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]
COMPLEXITY_PLAN = ROOT / "experiments/environment-complexity-split-plan.json"
# 12251-12253 share proven initializations with the complexity fits, whose
# continuous L3 runs are arm C; 12271-12273 are fresh.
SEEDS = (12251, 12252, 12253, 12271, 12272, 12273)
Arm = Literal["curriculum", "scratch"]
ARMS: tuple[Arm, ...] = ("curriculum", "scratch")
SOURCE_POOL: dict[Arm, Pool] = {"curriculum": "l1", "scratch": "l3"}
ANCHORS: tuple[Anchor, ...] = ("heuristic", "scripted_greedy", "random")
# Disjoint from the complexity (1_225_1xx_xxx) and baseline (1_226_xxx_xxx)
# namespaces; the boundary milestone has no complexity counterpart.
BOUNDARY_DEALS = tuple(range(1_227_132_000, 1_227_132_050))
# Every fit is a corrected-world fit, including CPU test fixtures; `regime`
# pins this only for CUDA to keep historical CPU recipes unchanged.
WORLD = "w6"
# Continuations require this declaration. Ataraxos move updates read only the
# absolute iteration, so learning is the complexity recipe's either way.
SCHEDULE_CLOCK: Literal["iteration_fraction"] = "iteration_fraction"


@dataclass(frozen=True)
class Recipe:
    """Horizon and placement shared by both arms; tests shrink it for CPU."""

    updates: int
    switch: int
    streams: Literal[4, 64]
    device: Literal["cpu", "cuda"]
    schedule: IterationSchedule
    seconds_per_fit: float

    def seconds(self, updates: int) -> float:
        return self.seconds_per_fit * updates / self.updates


def source_recipe(arm: Arm, recipe: Recipe) -> TrainingRegime:
    """Iterations 1..switch on the arm's source decks, on the full schedule."""
    source = regime(
        SOURCE_POOL[arm],
        updates=recipe.switch,
        seconds=recipe.seconds(recipe.switch),
        streams=recipe.streams,
        device=recipe.device,
        schedule=recipe.schedule,
    )
    source.id = f"mtg122-curriculum-{arm}-source"
    source.world = WORLD
    source.schedule_clock = SCHEDULE_CLOCK
    return TrainingRegime.model_validate(source.model_dump())


def target_recipe(
    arm: Arm, recipe: Recipe, source: TrainingRun, source_run: ArtifactReference
) -> TrainingRegime:
    """Iterations switch+1..updates on L3, continuing the source's learning state.

    The curriculum arm declares its source match as a setup transfer; scratch
    continues on an unchanged match. Both cross the same boundary otherwise.
    """
    target = regime(
        "l3",
        updates=recipe.updates,
        seconds=recipe.seconds(recipe.updates - recipe.switch),
        streams=recipe.streams,
        device=recipe.device,
        schedule=recipe.schedule,
    )
    target.id = f"mtg122-curriculum-{arm}-target"
    target.world = WORLD
    target.schedule_clock = SCHEDULE_CLOCK
    stage = target.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    record = source.stages[0]
    stage.learning_state = LearningStateImport(
        source_run=source_run,
        source_stage=record.id,
        raw=record.artifacts["raw"],
        ema=record.artifacts["ema"],
        optimizer=record.artifacts["optimizer"],
        source_match=source.regime.match if arm == "curriculum" else None,
    )
    return TrainingRegime.model_validate(target.model_dump())


def _reference(path: Path) -> ArtifactReference:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _relocated(artifact: ArtifactReference, directory: Path) -> ArtifactReference:
    """Point a GPU-host artifact at its fetched copy; scoring re-verifies bytes."""
    return {**artifact, "path": str(directory / Path(artifact["path"]).name)}


def initialization_digest(recipe: Recipe, seed: int) -> str:
    """Predict the producer's initial tensors; both source pools must agree.

    Orthogonal initialization runs a QR decomposition whose x86 MKL result
    depends on the intra-op thread count, so match the producer's setting.
    """
    digests = set()
    for arm in ARMS:
        definition = source_recipe(arm, recipe)
        stage = definition.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        torch.set_num_threads(stage.execution.threads)
        torch.manual_seed(seed)
        model = Agent(ObservationSpace(definition.observation), definition.agent)
        digests.add(parameter_digest(model))
    if len(digests) != 1:
        raise ValueError("source pools change the initial learnable tensors")
    return digests.pop()


@dataclass(frozen=True)
class Fit:
    """One arm of one seed: its source segment and L3 continuation."""

    source: TrainingRun
    target: TrainingRun


def train_fit(
    arm: Arm,
    recipe: Recipe,
    seed: int,
    directory: Path,
    expected_digest: str,
    *,
    stop_path: Path | None = None,
) -> Fit:
    """Train both segments in this process; the transfer never crosses hosts."""
    with VerifyStore(directory / "training.sqlite") as store:
        source = execute_regime(
            source_recipe(arm, recipe),
            seed,
            directory / "source",
            store,
            stop_path=stop_path,
        )
        if source.status != "completed":
            raise RuntimeError(f"{arm}-{seed} source failed: {source.error}")
        initial, _ = load_checkpoint_agent(
            source.stages[0].artifacts["initial_raw"]["path"]
        )
        if parameter_digest(initial) != expected_digest:
            raise ValueError("actual producer initialization differs from preflight")
        target = execute_regime(
            target_recipe(
                arm, recipe, source, _reference(directory / "source/run.json")
            ),
            seed,
            directory / "target",
            store,
            checkpoint_updates=tuple(
                n for n in (recipe.updates // 2,) if recipe.switch < n < recipe.updates
            ),
            stop_path=stop_path,
        )
    if target.status != "completed":
        raise RuntimeError(f"{arm}-{seed} target failed: {target.error}")
    return Fit(source, target)


def fit_checkpoints(fit: Fit, directory: Path | None = None) -> dict[int, Checkpoint]:
    """L3-bound exports keyed by absolute iteration: switch, midpoint, endpoint.

    The continuation's initial export is the source's final policy rebound to
    L3, so the curriculum arm's boundary checkpoint is scored on decks it never
    trained on. `directory` relocates artifacts fetched from the GPU host.
    """
    found = {
        c.coordinates.updates: c for c in checkpoints(fit.target, include_initial=True)
    }
    if directory is not None:
        found = {
            n: Checkpoint(
                artifact=_relocated(c.artifact, directory / "target"),
                coordinates=c.coordinates,
            )
            for n, c in found.items()
        }
    return found


def initial_checkpoint(scratch: Fit, directory: Path | None = None) -> Checkpoint:
    """The shared initialization, taken from the L3-bound scratch source."""
    (initial,) = [
        c
        for c in checkpoints(scratch.source, include_initial=True)
        if c.coordinates.updates == 0
    ]
    if directory is None:
        return initial
    return Checkpoint(
        artifact=_relocated(initial.artifact, directory / "source"),
        coordinates=initial.coordinates,
    )


class CurriculumPlan(Strict):
    """Frozen two-arm, six-seed cohort; GPU training and rented-CPU scoring.

    Charged to MTG-122's inclusive $35 cap (Jack Heart, 2026-10-19).
    `prior_usd` is what that cap has already spent or reserved;
    `evaluation_usd` is the rented-CPU allowance frozen from calibration.
    Each fit's watchdog is protocol017's 1,740 s, split between its segments
    in proportion to their iterations.
    """

    schema_version: Literal[1] = 1
    seed_order: tuple[int, ...] = SEEDS
    updates: Literal[128] = 128
    switch: Literal[32] = 32
    streams: Literal[64] = 64
    device: Literal["cuda"] = "cuda"
    training_seconds_per_fit: Literal[1740] = 1740
    total_seconds: float = Field(gt=0)
    max_usd_per_hour: float = Field(gt=0, le=0.69)
    prior_usd: float = Field(ge=0)
    evaluation_usd: float = Field(ge=0)
    settlement_reserve_usd: Literal[3] = 3
    inclusive_cap_usd: Literal[35] = 35
    evaluation_placement: Literal["rented-cpu"] = "rented-cpu"
    complexity_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def admit(self) -> "CurriculumPlan":
        if self.seed_order != SEEDS:
            raise ValueError("cohort requires its fixed six seeds")
        fits = len(self.seed_order) * len(ARMS)
        if fits * self.training_seconds_per_fit + 600 > self.total_seconds:
            raise ValueError("twelve bounded fits plus setup exceed the allocation")
        rental = self.total_seconds / 3600 * self.max_usd_per_hour
        if (
            rental + self.prior_usd + self.evaluation_usd + self.settlement_reserve_usd
            > self.inclusive_cap_usd
        ):
            raise ValueError("curriculum exceeds MTG-122's inclusive $35 allocation")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    @property
    def recipe(self) -> Recipe:
        return Recipe(
            updates=self.updates,
            switch=self.switch,
            streams=self.streams,
            device=self.device,
            schedule=TESTED_SCHEDULE,
            seconds_per_fit=self.training_seconds_per_fit,
        )

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return (0, self.switch, self.updates // 2, self.updates)

    def complexity(self) -> SplitComplexityPlan:
        plan = read_complexity_plan(COMPLEXITY_PLAN.read_text())
        if not isinstance(plan, SplitComplexityPlan) or plan.digest != (
            self.complexity_plan_sha256
        ):
            raise ValueError("frozen complexity plan differs from the declared digest")
        return plan

    def protocol(self, anchor: Anchor, milestone: int) -> MonitorProtocol:
        if milestone == self.switch:
            return MonitorProtocol(
                authored_matchup=roster("l3"),
                include_mirrors=True,
                deal_seeds=BOUNDARY_DEALS,
                opponent=anchor,
                purpose="frozen-study-evaluation",
                study_protocol_sha256=self.digest,
                bootstrap_seed=12290,
                bootstrap_replicates=10000,
            )
        # Byte-identical to the complexity cohort's L3 cells at 0, 64 and 128.
        return self.complexity().protocol("l3", anchor, milestone)


# ---- Scoring ---------------------------------------------------------------

ProtocolFor = Callable[[Anchor, int], MonitorProtocol]
Variant = Literal["initial", "curriculum", "scratch", "control"]
VARIANTS: tuple[Variant, ...] = ("initial", "curriculum", "scratch", "control")


def evaluate_seed(
    fits: dict[Arm, Fit],
    milestones: tuple[int, ...],
    protocol: ProtocolFor,
    directory: Path,
    should_stop: Callable[[], bool],
    *,
    anchors: tuple[Anchor, ...] = ANCHORS,
    variants: tuple[Variant, ...] = VARIANTS,
    fetched: dict[Arm, Path] | None = None,
) -> int:
    """Score one seed's arms and frozen controls; return cells newly run.

    Milestone 0 is the shared initialization, scored once. Later milestones
    score each arm and the unchanged initialization as a same-deal control. A
    completed cell is kept; any other retained attempt stops scoring.
    `variants` selects cells so separate processes can share one seed; each
    cell has its own directory. `fetched` maps each arm to its fetched fit
    directory when artifacts were recorded on another host.
    """
    initial = initial_checkpoint(
        fits["scratch"], None if fetched is None else fetched["scratch"]
    )
    exports = {
        arm: fit_checkpoints(fit, None if fetched is None else fetched[arm])
        for arm, fit in fits.items()
    }
    evaluated = 0
    for milestone in milestones:
        cells: list[tuple[Variant, TrainingRun, Checkpoint]] = (
            [("initial", fits["scratch"].source, initial)]
            if milestone == 0
            else [
                *((arm, fits[arm].target, exports[arm][milestone]) for arm in fits),
                ("control", fits["scratch"].target, initial),
            ]
        )
        for anchor in anchors:
            for variant, run, checkpoint in cells:
                if variant not in variants:
                    continue
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
                    raise RuntimeError("allocation exhausted with incomplete cohort")
                result = evaluate_checkpoint(
                    run,
                    checkpoint.artifact,
                    checkpoint.coordinates,
                    cell,
                    protocol=protocol(anchor, milestone),
                )
                if result.status != "completed":
                    raise RuntimeError(f"{cell.name} incomplete")
                evaluated += 1
    return evaluated


def _fetched_fit(plan: CurriculumPlan, arm: Arm, seed: int, directory: Path) -> Fit:
    """Read one fetched fit and check it is exactly the frozen plan's."""
    source = TrainingRun.model_validate_json(
        (directory / "source/run.json").read_text()
    )
    target = TrainingRun.model_validate_json(
        (directory / "target/run.json").read_text()
    )
    stage = target.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
    reference = stage.learning_state.source_run
    if file_sha256(directory / "source/run.json") != reference["sha256"]:
        raise ValueError(f"{arm}-{seed} source run differs from its continuation")
    if (
        source.seed != seed
        or target.seed != seed
        or source.regime != source_recipe(arm, plan.recipe)
        or target.regime != target_recipe(arm, plan.recipe, source, reference)
        or source.status != "completed"
        or target.status != "completed"
    ):
        raise ValueError(f"{arm}-{seed} differs from the frozen plan")
    return Fit(source, target)


def evaluate_retained(
    campaign: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    anchors: tuple[Anchor, ...] = ANCHORS,
    milestones: tuple[int, ...] | None = None,
    variants: tuple[Variant, ...] = VARIANTS,
) -> None:
    """Score fetched GPU outputs in place, inside a rented CPU allocation.

    Interrupting is safe: a rerun skips completed cells and stops at a
    retained incomplete one.
    """
    env = os.environ
    if not env.get("ETUDE_RUN") or not env.get("ETUDE_DEADLINE"):
        raise ValueError(
            "curriculum scoring runs in a rented allocation; local forbidden"
        )
    plan = CurriculumPlan.model_validate_json((campaign / "plan.json").read_text())
    if milestones is not None and not set(milestones) <= set(plan.evaluation_updates):
        raise ValueError("unplanned milestone")
    expected = json.loads((campaign / "initialization-parameters.json").read_text())
    should_stop = allocation_stop(
        datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp()
    )
    for seed in seeds:
        fits = {
            arm: _fetched_fit(plan, arm, seed, campaign / f"{arm}-{seed}")
            for arm in ARMS
        }
        for arm in ARMS:
            path = _relocated(
                fits[arm].source.stages[0].artifacts["initial_raw"],
                campaign / f"{arm}-{seed}/source",
            )["path"]
            initial, _ = load_checkpoint_agent(path)
            if parameter_digest(initial) != expected[str(seed)]:
                raise ValueError(f"{arm}-{seed} initialization differs from preflight")
        count = evaluate_seed(
            fits,
            plan.evaluation_updates if milestones is None else milestones,
            plan.protocol,
            campaign / f"evaluation/{seed}",
            should_stop,
            anchors=anchors,
            variants=variants,
            fetched={arm: campaign / f"{arm}-{seed}" for arm in ARMS},
        )
        print(f"{seed}: {count} cells evaluated", flush=True)


# ---- GPU training job ------------------------------------------------------


def execute(plan: CurriculumPlan) -> None:
    """Train all twelve fits; the launcher's guardian owns hard stopping."""
    env = os.environ
    if not env.get("ETUDE_RUN"):
        raise ValueError("curriculum training requires an Etude remote allocation")
    if not torch.cuda.is_available():
        raise ValueError("curriculum training requires CUDA; no CPU fallback")
    plan.complexity()
    deadline = datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp()
    fits = len(plan.seed_order) * len(ARMS)
    if deadline - time.time() < fits * plan.training_seconds_per_fit + 300:
        raise ValueError("remaining SDK allocation cannot admit the frozen campaign")
    output = Path(env["ETUDE_OUTPUT_DIR"])
    manifest = output / "campaign.json"
    if manifest.exists():
        raise ValueError("campaign evidence already exists")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan.model_dump(mode="json"))
    start = time.time()
    completed: list[str] = []
    state: dict[str, object] = {
        "status": "running",
        "plan_sha256": plan.digest,
        "started_unix": start,
        "completed": completed,
    }
    atomic_json(manifest, state)
    try:
        expected = {
            seed: initialization_digest(plan.recipe, seed) for seed in plan.seed_order
        }
        atomic_json(output / "initialization-parameters.json", expected)
        for index, seed in enumerate(plan.seed_order):
            # Alternate which arm trains first so host drift is not arm-aligned.
            for arm in ARMS[index % 2 :] + ARMS[: index % 2]:
                if deadline - time.time() < plan.training_seconds_per_fit + 60:
                    raise RuntimeError("allocation exhausted before next declared fit")
                key = f"{arm}-{seed}"
                directory = output / key
                directory.mkdir()
                fit = train_fit(
                    arm,
                    plan.recipe,
                    seed,
                    directory,
                    expected[seed],
                    stop_path=Path(env["ETUDE_STOP_FILE"]),
                )
                # Exports are checked now so a missing one fails on the GPU.
                missing = set(plan.evaluation_updates[1:]) - set(fit_checkpoints(fit))
                if missing:
                    raise RuntimeError(f"{key} lacks exports at {sorted(missing)}")
                completed.append(key)
                state["elapsed_seconds"] = time.time() - start
                atomic_json(manifest, state)
        state["status"] = "completed"
        atomic_json(output / "etude-result.json", {"clocks": {"completed_runs": fits}})
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state["elapsed_seconds"] = time.time() - start
        atomic_json(manifest, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    evaluate = commands.add_parser(
        "evaluate", help="score fetched outputs on rented CPU"
    )
    evaluate.add_argument("campaign", type=Path)
    evaluate.add_argument("--seed", type=int, choices=SEEDS, action="append")
    evaluate.add_argument("--anchor", choices=ANCHORS, action="append")
    evaluate.add_argument("--milestone", type=int, action="append")
    evaluate.add_argument("--variant", choices=VARIANTS, action="append")
    arguments = parser.parse_args()
    if arguments.command == "evaluate":
        evaluate_retained(
            arguments.campaign,
            seeds=tuple(arguments.seed or SEEDS),
            anchors=tuple(arguments.anchor or ANCHORS),
            milestones=None
            if arguments.milestone is None
            else tuple(arguments.milestone),
            variants=tuple(arguments.variant or VARIANTS),
        )
    else:
        # Etude worker entry: the job config carries the frozen plan.
        payload = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())["plan"]
        execute(CurriculumPlan.model_validate(payload))
