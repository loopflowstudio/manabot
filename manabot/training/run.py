"""The run handle: one bounded training attempt that regime code acts through.

`Run` owns everything about an attempt besides the training math: its
`TrainingRun` record in `VerifyStore` and the `run.json` export, the named seed
streams, wall/stop/pause supervision, monitoring exports, failure capture and
the settlement of a failed stage. Stage functions (`manabot.training.stages`)
open one `StageScope` each with `Run.begin_stage` and close it with
`Run.finish_stage`, so a stage's record, cost and artifacts look the same
whichever recipe called it.

Two kinds of recipe run through it:

- a regime function: any module-level function taking the `Run` (or a
  `functools.partial` binding its keyword arguments) that creates `Model`s
  and calls stage functions. `execute` runs one under a writer lease;
  `smoke` runs it once in its tiny form.
- a saved `TrainingRegime`, run by `execute_regime`
  (`manabot.training.execution`) through `stages.run_saved_regime`.

`attempt` is the lease and recovery admission both share: it holds the writer
lease and, when resuming, the parent's, loads the parent's last committed
update snapshot and settles a crashed parent's unobserved cost.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import functools
import importlib
import inspect
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Any, Literal, get_type_hints
import uuid
import zlib

import numpy as np
import psutil
from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError, to_jsonable_python
import torch

from manabot.env.failures import CaptureScope, capture_scope
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.files import atomic_json
from manabot.infra.host import host_facts
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints, source_bundle_sha256
from manabot.infra.runtime import RunContext
from manabot.infra.source import source_commit, source_uncommitted_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.policy_distribution import NumericalError
from manabot.training.decks import UniformDecks
from manabot.training.learning.learner import Learner
from manabot.training.mps import mps_torch_error
from manabot.training.precision import apply_matmul_precision
from manabot.training.records import (
    MonitoringCheckpoint,
    NumericalFailure,
    RegimeFunctionRecord,
    SourceRevision,
    StageRecord,
    TrainingRun,
)
from manabot.training.regime import (
    ArtifactReference,
    Execution,
    RecoveryPolicy,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.game_errors import GameErrorLedger, GameErrorPolicy
from manabot.training.state.incidents import save_incident
from manabot.training.state.recovery import (
    UpdateSnapshot,
    admit_recovery_identities,
    attempt_lock,
    load_update,
    settle_orphan,
)
from manabot.verify.store import VerifyStore
import managym

from .clock import boot_identity, watchdog_seconds

if TYPE_CHECKING:
    from manabot.training.rollout.collection import RolloutBatch


# Longest a run.json export may lag the store while a stage is making progress.
PROGRESS_EXPORT_SECONDS = 30.0
# Largest share of a stage's wall time that progress exports may consume.
PROGRESS_EXPORT_SHARE = 0.02

# Named seed streams, as offsets from the run seed.
SEED_OFFSETS = (
    ("initialization", 0),
    ("collection", 10000),
    ("minibatches", 20000),
    ("evaluation", 30000),
    ("belief_collection", 40000),
    ("belief_initialization", 50000),
)


def artifact(path: Path) -> ArtifactReference:
    """Reference published bytes by digest and size."""
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def process_tree_rss(process: psutil.Process) -> int:
    """Resident bytes of a process and its live descendants.

    A child can exit between enumeration and its RSS read. psutil's
    `is_running()` is still True for a zombie, and on macOS reading a zombie's
    memory raises `ZombieProcess`; an exited child holds no memory to count,
    so `NoSuchProcess` (its base class) skips it. Other failures, such as
    `AccessDenied`, still surface: the memory cap must not stop measuring.
    """
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def export_training_run(
    run_id: str, store: "VerifyStore", out: str | Path
) -> TrainingRun:
    run = store.training_run(run_id)
    atomic_json(Path(out) / "run.json", run.model_dump(mode="json"))
    return run


# ---------------------------------------------------------------------------
# Stage supervision


@dataclass
class StageScope:
    """Clocks, limits and failure state of the one stage running now.

    `check()` is the supervision point stage code calls at every boundary and
    native step: it raises on the wall deadline, a launcher stop or the memory
    cap. The self-play fields hold what numerical-incident capture needs if
    the stage fails.
    """

    run: "Run"
    record: StageRecord
    execution: Execution
    self_play: bool
    restoring_completed: bool
    stage_start: float
    stage_watchdog_start: float
    prior_stage_watchdog: float
    cpu_start: float
    deadline: float
    last_memory_sample: float = float("-inf")
    learner: Learner | None = None
    collector: SeatRoutedCollector | None = None
    batch: "RolloutBatch | None" = None
    ema: ActorCritic | None = None
    rng: np.random.Generator | None = None
    iteration: int = 0

    def synchronize(self) -> None:
        if self.execution.device == "cuda":
            torch.cuda.synchronize()
        elif self.execution.device == "mps":
            torch.mps.synchronize()

    def watchdog(self) -> float:
        """This stage's watchdog seconds across attempts, as of now."""
        return (
            self.prior_stage_watchdog
            + self.run.budget_now()
            - self.stage_watchdog_start
        )

    def remaining(self) -> float:
        """Seconds left before the first of the watchdog clocks expires."""
        run = self.run
        charged = (
            run.record.prior_watchdog_seconds + run.budget_now() - run.watchdog_start
        )
        left = min(self.deadline - time.perf_counter(), run.wall_seconds - charged)
        if run.recoverable and not self.restoring_completed:
            left = min(left, self.execution.wall_seconds - self.watchdog())
        return left

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("training resource wall deadline exceeded")
        # Deadline checks run at every native step. On macOS, recursive
        # child discovery scans the whole host process table; sampling
        # RSS once per second avoids making supervision the workload.
        now = time.perf_counter()
        if now - self.last_memory_sample < 1.0:
            return
        self.last_memory_sample = now
        stop_path = self.run.stop_path
        if stop_path is not None and stop_path.exists():
            # Cancellation retains failed scientific receipts; unlike
            # pause it makes no complete-state recovery promise.
            raise RuntimeError("training cancelled by launcher")
        memory = process_tree_rss(psutil.Process())
        if not self.restoring_completed:
            self.record.sampled_peak_rss_bytes = max(
                self.record.sampled_peak_rss_bytes, memory
            )
        if memory > self.execution.memory_bytes:
            raise MemoryError("training process tree memory limit exceeded")


# ---------------------------------------------------------------------------
# Models


@dataclass
class Model:
    """A named network of the run, its configuration and its current weights.

    Stage functions read and change a model in place; it is never copied
    implicitly. `session` is the live learner state self-play continues when
    the same model trains again; `exports` holds published checkpoints of the
    current weights by kind (a stage that changes the weights replaces it),
    so an unchanged model is published once.
    """

    name: str
    agent: ModelConfig
    observation: ObservationConfig
    network: ActorCritic
    ema: ActorCritic | None = None
    session: Any = None
    exports: dict[str, ArtifactReference] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The run


class Run:
    """One training attempt, handed to a regime function or saved regime.

    Regime code uses `id`, `seed`, `world`, `model`, `export`, `metrics` and
    `stage`; the remaining attributes are the plumbing stage functions share.
    """

    def __init__(
        self,
        *,
        record: TrainingRun,
        store: "VerifyStore",
        out: Path,
        seeds: dict[str, int],
        snapshot: UpdateSnapshot | None,
        prior_records: dict[str, StageRecord],
        start: float,
        calendar_start: float,
        wall_seconds: float,
        recovery: RecoveryPolicy | None,
        recoverable: bool,
        checkpoint_seconds: float | None,
        checkpoint_updates: int | None,
        checkpoint_milestones: tuple[int, ...],
        pause_path: Path | None,
        stop_path: Path | None,
        stop_reserve_seconds: float | None,
        initial_admission: Path | None,
        experiment: str | None,
        training_exposure: bool,
        game_ledger: GameErrorLedger | None,
        smoke_execution: Execution | None = None,
    ) -> None:
        self.record = record
        self.store = store
        self.out = out
        self.seeds = seeds
        self.snapshot = snapshot
        self.prior_records = prior_records
        self.start = start
        self.calendar_start = calendar_start
        self.budget_now: Callable[[], float] = (
            time.monotonic if recovery is not None else watchdog_seconds
        )
        self.watchdog_start = self.budget_now()
        self.wall_seconds = wall_seconds
        self.recovery = recovery
        self.recoverable = recoverable
        self.checkpoint_seconds = checkpoint_seconds
        self.checkpoint_updates = checkpoint_updates
        self.checkpoint_milestones = checkpoint_milestones
        self.pause_path = pause_path
        self.stop_path = stop_path
        self.stop_reserve_seconds = stop_reserve_seconds
        self.initial_admission = initial_admission
        self.experiment = experiment
        self.training_exposure = training_exposure
        self.game_ledger = game_ledger
        # Set by `smoke`: the execution every stage runs under.
        self.smoke_execution = smoke_execution
        self.resources = ExitStack()
        self.resources.callback(apply_matmul_precision, "float32")
        # The live collector; a new collector closes the previous one.
        self.producers = self.resources.enter_context(ExitStack())
        self.current: StageScope | None = None
        self.failures: CaptureScope | None = None
        self.belief_games = 0
        self.wall_bounded = False
        self.models: dict[str, Model] = {}
        # The self-play session whose collector is open; only it can continue.
        self.live_session: object = None
        self._stage_names: dict[str, int] = {}
        self._last_monitor_seconds = 0.0
        self._last_export = float("-inf")
        self._export_wait = PROGRESS_EXPORT_SECONDS

    # -- identity -----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def seed(self) -> int:
        return self.record.seed

    @property
    def world(self) -> str:
        return managym.WORLD_VERSION

    def stage_name(self, base: str) -> str:
        """`base` numbered by call order: `self-play`, `self-play-2`, ..."""
        count = self._stage_names.get(base, 0) + 1
        self._stage_names[base] = count
        return base if count == 1 else f"{base}-{count}"

    def completed(self, stage_id: str) -> StageRecord | None:
        """The committed record of a stage an earlier attempt completed.

        A resumed function re-enters from the top; its stages up to the
        recovery snapshot return these records instead of running again.
        """
        for record in self.record.stages:
            if record.id == stage_id:
                if record.status != "completed":
                    raise ValueError(f"stage {stage_id!r} was already run")
                return record
        return None

    # -- models -------------------------------------------------------------

    def model(
        self,
        name: str,
        agent: ModelConfig,
        observation: ObservationConfig,
        *,
        seed: int | None = None,
    ) -> Model:
        """A new randomly initialized model; seeded from the run seed and name.

        Initialization runs under a forked RNG, so creating a model never
        shifts the run's own streams.
        """
        if not name or name != name.lower() or not name.replace("-", "").isalnum():
            raise ValueError("model names are lowercase letters, digits and dashes")
        space = ObservationSpace(observation)
        initialization = (
            seed if seed is not None else self.seed + zlib.crc32(name.encode())
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization)
            network = ActorCritic(space, agent)
        return self.adopt(name, network, space.encoder.config)

    def adopt(
        self, name: str, network: ActorCritic, observation: ObservationConfig
    ) -> Model:
        """Register an existing network as the run's model `name`."""
        if name in self.models:
            raise ValueError(f"model {name!r} already exists in this run")
        model = Model(
            name=name, agent=network.config, observation=observation, network=network
        )
        self.models[name] = model
        return model

    def export(
        self,
        model: Model,
        weights: Literal["raw", "ema"] = "raw",
        *,
        decks: MatchConfig | UniformDecks | None = None,
    ) -> ArtifactReference:
        """Publish `model`'s current weights through ordinary admission.

        The checkpoint binds the world when `decks` is omitted or a deck
        source (which needs a model reading the complete world input), and
        otherwise the setups of `decks`. An export is a stage of its own
        (`export-<model>`); weights already published are returned as they
        are.
        """
        if weights in model.exports:
            return model.exports[weights]
        network = model.network if weights == "raw" else model.ema
        if network is None:
            raise ValueError(f"model {model.name!r} has no {weights} weights")
        setups = Match(decks).to_rust() if isinstance(decks, MatchConfig) else None
        if setups is None and model.agent.world != managym.WORLD_SETUP:
            raise ValueError(
                f"model {model.name!r} does not read the complete world input; "
                "export it for the decks it plays"
            )
        name = self.stage_name(f"export-{model.name}")
        recorded = self.completed(name)
        if recorded is not None:
            reference: ArtifactReference = recorded.artifacts[weights]  # type: ignore[assignment]
            model.exports[weights] = reference
            return reference
        scope = self.begin_stage(name, Execution(), operation="export")
        assert scope is not None
        tick = time.perf_counter()
        target = self.out / f"{name}-{weights}.pt"
        save_checkpoint(
            network,
            ObservationSpace(model.observation),
            target.with_suffix(".tmp"),
            player_configs=setups,
            extra={
                "run_id": self.id,
                "stage_id": name,
                "model": model.name,
                "weights": weights,
                "regime_digest": self.record.regime_digest,
                "value_semantic": "signed_outcome",
            },
        )
        os.replace(target.with_suffix(".tmp"), target)
        candidate = artifact(target)
        try:
            with torch.random.fork_rng(devices=[]):
                load_checkpoint(str(target))
        except Exception:
            scope.record.rejected_artifacts[weights] = candidate
            raise
        scope.record.artifacts[weights] = candidate
        scope.record.export_seconds = time.perf_counter() - tick
        self.finish_stage(scope)
        model.exports[weights] = candidate
        return candidate

    def metrics(self, values: Mapping[str, float], *, update: int) -> None:
        """Record named metrics of the regime's own at an update coordinate."""
        if self.current is None or self.current.record.status != "running":
            raise ValueError("metrics belong to a running stage; use run.stage(name)")
        self.current.record.diagnostics.append(
            {"update": update, **{key: float(value) for key, value in values.items()}}
        )
        self.persist(progress=True)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageScope]:
        """Record one stage of the regime's own: its cost, status and artifacts.

        Code inside calls `scope.check()` at its boundaries and adds artifacts
        with `scope.record.artifacts[name] = artifact(path)`.
        """
        stage_id = self.stage_name(name)
        if self.completed(stage_id) is not None:
            raise ValueError(
                f"stage {stage_id!r} completed in an earlier attempt; "
                "custom stages cannot be replayed on resume"
            )
        scope = self.begin_stage(stage_id, Execution(), operation="custom")
        assert scope is not None
        yield scope
        self.finish_stage(scope)

    # -- persistence --------------------------------------------------------

    def persist(self, *, progress: bool = False) -> None:
        """Commit the run to the store and refresh its run.json export.

        The store is the authority and is written every time. Rewriting the
        export costs time proportional to the whole record, so saves made only
        to record progress inside a stage refresh it at most once per
        PROGRESS_EXPORT_SECONDS, and less often once one export takes more
        than PROGRESS_EXPORT_SHARE of that interval. Stage boundaries,
        completion and failure always export.
        """
        run = self.record
        run.seconds = time.perf_counter() - self.start
        run.watchdog_seconds = self.budget_now() - self.watchdog_start
        run.last_recorded_wall_seconds = time.time()
        run.last_recorded_active_seconds = time.monotonic()
        run.calendar_seconds = max(
            0.0, run.last_recorded_wall_seconds - self.calendar_start
        )
        run.downtime_seconds = max(0.0, run.calendar_seconds - run.seconds)
        if self.game_ledger is not None and run.stages:
            run.stages[-1].rejected_artifacts["game-attempts"] = artifact(
                self.game_ledger.path
            )
        self.store.save_training_run(run)
        began = time.monotonic()
        if progress and began - self._last_export < self._export_wait:
            return
        export_training_run(run.id, self.store, self.out)
        self._last_export = time.monotonic()
        self._export_wait = max(
            PROGRESS_EXPORT_SECONDS,
            (self._last_export - began) / PROGRESS_EXPORT_SHARE,
        )

    def coordinates(self) -> dict[str, int | float]:
        run = self.record
        scope = self.current
        origins = [
            s.learning_state_origin
            for s in run.stages
            if s.learning_state_origin is not None
        ]
        cpu_start = scope.cpu_start if scope is not None else time.process_time()
        return {
            "updates": run.updates_through(),
            **(
                {"segment_updates": scope.record.updates}
                if scope is not None and scope.self_play
                else {}
            ),
            **(
                {"optimizer_updates": count}
                if (count := run.optimizer_updates_through()) is not None
                else {}
            ),
            "training_seconds": run.prior_seconds
            + time.perf_counter()
            - self.start
            + sum(origin.cumulative_seconds for origin in origins),
            "active_training_seconds": sum(
                s.collection_seconds + s.learning_seconds for s in run.stages
            )
            + sum(origin.active_training_seconds for origin in origins),
            **{
                name: sum(getattr(s, name) for s in run.stages)
                + sum(getattr(origin, name) for origin in origins)
                for name in (
                    "environment_decisions",
                    "learner_transitions",
                    "optimizer_exposures",
                    "games",
                )
            },
            "idle_stream_steps": sum(s.idle_stream_steps for s in run.stages),
            "rss_bytes": psutil.Process().memory_info().rss,
            "host_load_1m": os.getloadavg()[0],
            "process_cpu_seconds": time.process_time()
            - cpu_start
            + sum(s.cpu_seconds for s in run.stages[:-1]),
        }

    def monitor_checkpoint(
        self,
        scope: StageScope,
        stage: TrainSelfPlay,
        model: ActorCritic,
        space: ObservationSpace,
        setups: list[managym.PlayerConfig] | None,
        experiment: RunContext | None = None,
    ) -> None:
        """Export a monitoring checkpoint when the run's cadence is due."""
        run = self.record
        record = scope.record
        active_endpoint = stage.active_seconds is not None
        elapsed = (
            sum(s.collection_seconds + s.learning_seconds for s in run.stages)
            if active_endpoint
            else time.perf_counter() - self.start - run.monitoring_export_seconds
        )
        point = self.coordinates()
        if record.updates == stage.updates:
            return
        scheduled = point["updates"] in self.checkpoint_milestones and not any(
            c.updates == point["updates"] for c in run.monitoring_checkpoints
        )
        if self.checkpoint_updates is not None:
            if point["updates"] == 0 or point["updates"] % self.checkpoint_updates:
                return
        elif not scheduled and (
            self.checkpoint_seconds is None
            or elapsed - self._last_monitor_seconds < self.checkpoint_seconds
        ):
            return
        self._last_monitor_seconds = elapsed
        receipt = MonitoringCheckpoint(
            stage_id=record.id,
            ordinal=len(run.monitoring_checkpoints),
            **{
                key: point[key]
                for key in ("segment_updates", "optimizer_updates")
                if key in point
            },
            **{
                k: point[k]
                for k in (
                    "updates",
                    "training_seconds",
                    "active_training_seconds",
                    "environment_decisions",
                    "learner_transitions",
                    "optimizer_exposures",
                    "games",
                )
            },
        )
        run.monitoring_checkpoints.append(receipt)
        tick = time.perf_counter()
        target = self.out / f"monitor-{receipt.ordinal:08d}.pt"
        try:
            save_checkpoint(
                model,
                space,
                target.with_suffix(".tmp"),
                player_configs=setups,
                extra={
                    "run_id": run.id,
                    "stage_id": record.id,
                    "regime_digest": run.regime_digest,
                    "weights": "raw",
                    "value_semantic": "signed_outcome",
                },
            )
            os.replace(target.with_suffix(".tmp"), target)
            # Admission constructs a fresh model. Its initialization must not
            # consume the learner's action-sampling RNG stream.
            export_device = next(model.parameters()).device.type
            with torch.random.fork_rng(
                devices=[0] if export_device in {"cuda", "mps"} else [],
                device_type="mps" if export_device == "mps" else "cuda",
            ):
                load_checkpoint(str(target))
            receipt.artifact = artifact(target)
            if experiment is not None:
                experiment.checkpoint(
                    target, transitions=int(point["learner_transitions"])
                )
        except Exception as error:
            receipt.error = f"{type(error).__name__}: {error}"
            if target.exists():
                record.rejected_artifacts[target.name] = artifact(target)
        finally:
            run.monitoring_export_seconds += time.perf_counter() - tick

    # -- stages -------------------------------------------------------------

    def begin_stage(
        self,
        stage_id: str,
        execution: Execution,
        *,
        self_play: bool = False,
        keep_restored: bool = False,
        operation: str | None = None,
        arguments: Mapping[str, object] | None = None,
    ) -> StageScope | None:
        """Open the stage's record and supervision; `None` if it needs no work.

        The first stage to open after a recovery snapshot continues the
        snapshot's record. A snapshot taken when that stage had completed
        returns `None` unless `keep_restored`, which a stage asks for when the
        next stage continues its live state.
        """
        snapshot = self.snapshot
        if snapshot is not None and snapshot.record.id != stage_id:
            raise ValueError("recovery stage order differs from the parent attempt")
        restoring_completed = (
            snapshot is not None and snapshot.record.status == "completed"
        )
        record = (
            snapshot.record.model_copy(deep=True)
            if snapshot is not None
            else StageRecord(id=stage_id)
        )
        if restoring_completed:
            # Canonical completion includes publication cost measured after
            # the snapshot bytes were frozen. Never rewrite that record.
            record = self.prior_records[stage_id].model_copy(deep=True)
            if not keep_restored:
                self.record.stages.append(record)
                self.snapshot = None
                return None
        if operation is not None and self.record.regime_function is not None:
            record.operation = operation
            if arguments is not None:
                record.arguments = to_jsonable_python(dict(arguments))
        if self_play:
            if snapshot is None:
                record.optimizer_updates = 0
            if not restoring_completed:
                record.admission_wait_seconds = 0.0
                record.collection_timings = {}
                record.learning_timings = {}
        if not restoring_completed:
            record.status = "running"
            record.error = None
        if snapshot is not None and not restoring_completed:
            # Counts describe the continued scientific trajectory; timing
            # describes this attempt only. Prior attempt costs live on run.
            for timing in (
                "seconds",
                "collection_seconds",
                "learning_seconds",
                "export_seconds",
                "cpu_seconds",
            ):
                setattr(record, timing, 0.0)
            record.cumulative_seconds = None
        self.record.stages.append(record)
        stage_start = time.perf_counter()
        prior_stage_watchdog = (
            self.prior_records[stage_id].watchdog_seconds
            if stage_id in self.prior_records
            else 0.0
        )
        scope = StageScope(
            run=self,
            record=record,
            execution=execution,
            self_play=self_play,
            restoring_completed=restoring_completed,
            stage_start=stage_start,
            stage_watchdog_start=self.budget_now(),
            prior_stage_watchdog=prior_stage_watchdog,
            cpu_start=time.process_time(),
            deadline=min(
                self.start + self.wall_seconds - self.record.prior_watchdog_seconds,
                float("inf")
                if restoring_completed
                else stage_start + execution.wall_seconds - prior_stage_watchdog,
            ),
        )
        self.current = scope
        torch.set_num_threads(execution.threads)
        apply_matmul_precision(execution.precision)
        if execution.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but unavailable; CPU fallback is forbidden"
            )
        if execution.device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError(
                    "MPS requested but unavailable; CPU fallback is forbidden"
                )
            if (error := mps_torch_error()) is not None:
                raise ValueError(error)
        scope.synchronize()
        if not restoring_completed:
            record.actual_threads = torch.get_num_threads()
        return scope

    def finish_stage(
        self,
        scope: StageScope,
        *,
        completion: Callable[[], None] | None = None,
    ) -> None:
        """Complete the stage: freeze its costs, then commit.

        `completion` runs after the costs are frozen (self-play's final
        recovery snapshot); if it fails the stage is still running and its
        failure is the stage's.
        """
        record = scope.record
        scope.check()

        def freeze() -> None:
            record.seconds = time.perf_counter() - scope.stage_start
            record.cpu_seconds = time.process_time() - scope.cpu_start
            # Freeze the admission cost; later persistence belongs to later outputs.
            record.cumulative_seconds = (
                self.record.prior_seconds + time.perf_counter() - self.start
            )
            record.watchdog_seconds = scope.watchdog()

        freeze()
        record.status = "paused" if self.wall_bounded else "completed"
        if completion is not None:
            try:
                completion()
            except BaseException:
                record.status = "running"
                raise
            freeze()
        self.persist()

    # -- lifecycle ----------------------------------------------------------

    def execute(
        self, identities: Callable[[], dict[str, object]], body: Callable[[], None]
    ) -> TrainingRun:
        """Run `body` as this attempt: settle it on success, failure or pause."""
        run = self.record
        try:
            run.identities = identities()
            run.host = host_facts()
            if self.snapshot is not None:
                admit_recovery_identities(
                    self.snapshot.identities,
                    run.identities,
                    run.source_repair_reason,
                )
            run.setup_seconds = time.perf_counter() - self.start
            self.persist()
            self.failures = CaptureScope(
                self.out / "failures",
                lambda: {
                    "run": run.id,
                    "seed": run.seed,
                    "regime_digest": run.regime_digest,
                    "stage": run.stages[-1].id if run.stages else None,
                    "identities": run.identities,
                    "coordinates": self.coordinates(),
                    "recovery_artifact": run.recovery_artifact,
                    "allocation_deadline": run.allocation_deadline,
                    "parent_run_id": run.parent_run_id,
                },
                training_exposure=self.training_exposure,
            )
            self.resources.enter_context(capture_scope(self.failures))
            body()
            completed_models = [
                item.artifacts["raw"] for item in run.stages if "raw" in item.artifacts
            ]
            run.selected_artifact = completed_models[-1] if completed_models else None
            if self.wall_bounded:
                run.status = "paused"
                run.error = (
                    "wall budget ended before the update target; learner state exported"
                )
            else:
                run.status = "completed"
            self.persist()
        except BaseException as error:
            self._settle_failure(error)
            raise
        finally:
            self.resources.close()
            if self.training_exposure and run.stages:
                for path in sorted((self.out / "exposure").glob("*.jsonl")):
                    run.stages[-1].artifacts[f"exposure/{path.name}"] = artifact(path)
                self.persist()
        return run

    def _settle_failure(self, error: BaseException) -> None:
        run = self.record
        run.status = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, TimeoutError))
            else "failed"
        )
        run.error = f"{type(error).__name__}: {error}"
        scope = self.current
        if run.stages and run.stages[-1].status != "completed" and scope is not None:
            record = run.stages[-1]
            record.watchdog_seconds = scope.watchdog()
            record.status = run.status
            record.error = run.error
            record.seconds = time.perf_counter() - scope.stage_start
            for failure in (self.out / "failures").glob("*.jsonl"):
                record.rejected_artifacts[f"failures/{failure.name}"] = artifact(
                    failure
                )
            retained_paths = {item["path"] for item in record.artifacts.values()}
            for journal in self.out.glob(f"{record.id}-*.receipts.jsonl"):
                if str(journal) not in retained_paths:
                    record.rejected_artifacts[journal.name] = artifact(journal)
            record.cpu_seconds = time.process_time() - scope.cpu_start
            unaccounted = max(
                0.0,
                record.seconds
                - record.collection_seconds
                - record.learning_seconds
                - record.export_seconds
                - record.diagnostic_seconds
                - (record.admission_wait_seconds or 0.0),
            )
            # Unmeasured setup, persistence and waits are never active training.
            # Self-play operation timers settle in finally, including failed work.
            record.diagnostic_seconds += unaccounted
            if isinstance(error, NumericalError) and scope.learner is not None:
                record.numerical_failure = NumericalFailure(
                    invariant=error.invariant,
                    iteration=scope.iteration + 1,
                    health={
                        **error.health,
                        "rejected_updates": 1,
                        "nonfinite_failure": int(
                            error.invariant.startswith("nonfinite")
                        ),
                    },
                )
                capture_start = time.perf_counter()
                try:
                    paths = save_incident(
                        self.out / f"{record.id}-numerical-incident",
                        error,
                        scope.learner,
                        scope.collector,
                        scope.batch,
                        run,
                        scope.iteration + 1,
                        scope.ema,
                        scope.rng,
                    )
                    for path in paths:
                        record.rejected_artifacts[f"numerical/{path.name}"] = artifact(
                            path
                        )
                except Exception as capture_error:
                    # Failure capture cannot replace the primary training error.
                    run.error += (
                        f"; incident capture failed: {type(capture_error).__name__}"
                    )
                    record.error = run.error
                record.diagnostic_seconds += time.perf_counter() - capture_start
                record.seconds = time.perf_counter() - scope.stage_start
        try:
            self.persist()
        except Exception as telemetry_error:
            error.add_note(
                f"terminal telemetry failed: {type(telemetry_error).__name__}"
            )
            try:
                atomic_json(
                    self.out / "terminal-error.json",
                    {"error": run.error, "run": run.id},
                )
            except OSError:
                pass


def open_run(
    store: "VerifyStore",
    out: Path,
    *,
    seed: int,
    regime: TrainingRegime | None,
    regime_function: RegimeFunctionRecord | None,
    regime_digest: str,
    wall_seconds: float,
    recovery: RecoveryPolicy | None,
    recoverable: bool,
    game_errors: GameErrorPolicy | None,
    parent: TrainingRun | None,
    snapshot: UpdateSnapshot | None,
    source_repair_reason: str | None = None,
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    checkpoint_milestones: tuple[int, ...] = (),
    pause_path: Path | None = None,
    stop_path: Path | None = None,
    stop_reserve_seconds: float | None = None,
    initial_admission: Path | None = None,
    experiment: str | None = None,
    training_exposure: bool = False,
    smoke_execution: Execution | None = None,
) -> Run:
    """Create and commit the attempt's record; the caller holds the leases."""
    # A retry can fail during setup before reaching its stage. Walk the retained
    # lineage so that failure cannot erase a stage allowance or completion receipt.
    prior_records: dict[str, StageRecord] = {}
    ancestor = parent
    while ancestor is not None:
        for prior_record in ancestor.stages:
            prior_records.setdefault(prior_record.id, prior_record)
        ancestor = (
            store.training_run(ancestor.parent_run_id)
            if ancestor.parent_run_id
            else None
        )
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    calendar_start = time.time()
    seeds = {name: seed + offset for name, offset in SEED_OFFSETS}
    record = TrainingRun(
        id=uuid.uuid4().hex,
        regime=regime,
        regime_function=regime_function,
        regime_digest=regime_digest,
        seed=seed,
        seed_streams=seeds,
        identities=deepcopy(parent.identities) if parent is not None else {},
        status="running",
        monitoring_checkpoint_seconds=checkpoint_seconds,
        monitoring_checkpoint_updates=checkpoint_updates,
        monitoring_checkpoint_milestones=checkpoint_milestones,
        recovery_lock_path=str(out.with_name(out.name + ".writer.lock")),
        recovery_host=socket.gethostname(),
        last_recorded_wall_seconds=time.time(),
        last_recorded_active_seconds=time.monotonic(),
        recovery_boot_identity=boot_identity() if recovery else None,
        parent_run_id=parent.id if parent is not None else None,
        source_repair_reason=source_repair_reason,
        game_error_ledgers=list(parent.game_error_ledgers) if parent else [],
        prior_seconds=parent.prior_seconds + parent.seconds
        if parent is not None
        else 0,
        prior_watchdog_seconds=(parent.prior_watchdog_seconds + parent.watchdog_seconds)
        if parent is not None
        else 0,
        recovery_artifact=parent.recovery_artifact if parent is not None else None,
        metrics_run=parent.metrics_run if parent is not None else None,
        stages=deepcopy(snapshot.completed_stages) if snapshot is not None else [],
    )
    if parent is not None and (checkpoint_updates is not None or checkpoint_milestones):
        assert snapshot is not None
        restored_updates = sum(
            s.updates for s in (*snapshot.completed_stages, snapshot.record)
        )
        # Retain admitted exports from the committed prefix. Their artifact paths,
        # original costs and coordinates remain unchanged across process attempts.
        record.monitoring_checkpoints = [
            c.model_copy(deep=True)
            for c in parent.monitoring_checkpoints
            if c.updates <= restored_updates
        ]
        for receipt in record.monitoring_checkpoints:
            if receipt.artifact is not None and (
                file_sha256(receipt.artifact["path"]) != receipt.artifact["sha256"]
                or Path(receipt.artifact["path"]).stat().st_size
                != receipt.artifact["bytes"]
            ):
                raise ValueError("recovery monitoring artifact differs")
    game_ledger = None
    if game_errors is not None:
        game_ledger = GameErrorLedger(
            out / "game-attempts.jsonl", game_errors, parents=record.game_error_ledgers
        )
        record.game_error_ledgers.append(str(game_ledger.path))
        record.game_errors = game_ledger.state
    if parent is None:
        store.save_training_run(record)
    else:
        store.claim_training_recovery(parent.id, record)
    return Run(
        record=record,
        store=store,
        out=out,
        seeds=seeds,
        snapshot=snapshot,
        prior_records=prior_records,
        start=start,
        calendar_start=calendar_start,
        wall_seconds=wall_seconds,
        recovery=recovery,
        recoverable=recoverable,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_updates=checkpoint_updates,
        checkpoint_milestones=checkpoint_milestones,
        pause_path=pause_path,
        stop_path=stop_path,
        stop_reserve_seconds=stop_reserve_seconds,
        initial_admission=initial_admission,
        experiment=experiment,
        training_exposure=training_exposure,
        game_ledger=game_ledger,
        smoke_execution=smoke_execution,
    )


@dataclass(frozen=True)
class Attempt:
    """An admitted attempt: its output directory and, when resuming, its parent."""

    out: Path
    parent: TrainingRun | None
    snapshot: UpdateSnapshot | None
    source_repair_reason: str | None


@contextmanager
def attempt(
    store: "VerifyStore",
    out: str | Path,
    *,
    seed: int,
    regime_digest: str,
    identities: Callable[[], dict[str, object]],
    resume_from: str | None,
    source_repair_reason: str | None,
    cadence: tuple[int | None, float | None, tuple[int, ...]],
    game_error_recovery: bool,
) -> Iterator[Attempt]:
    """Hold the writer lease (and a resumed parent's) and admit the parent.

    A parent must be a stopped attempt of the same recipe digest, seed and
    monitoring cadence whose last snapshot matches this runtime.
    """
    destination = Path(out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lease = destination.with_name(destination.name + ".writer.lock")
    with ExitStack() as leases:
        leases.enter_context(attempt_lock(lease))
        parent = None
        snapshot = None
        if resume_from is not None:
            parent = store.training_run(resume_from)
            if parent.recovery_lock_path is None:
                raise ValueError("attempt has no recovery writer lease")
            leases.enter_context(
                attempt_lock(Path(parent.recovery_lock_path), existing=True)
            )
            if parent.seed != seed or parent.regime_digest != regime_digest:
                raise ValueError("recovery recipe or seed mismatch")
            if cadence != (
                parent.monitoring_checkpoint_updates,
                parent.monitoring_checkpoint_seconds,
                parent.monitoring_checkpoint_milestones,
            ):
                raise ValueError("recovery checkpoint cadence differs from parent")
            snapshot = load_update(parent)
            admit_recovery_identities(
                parent.identities, identities(), source_repair_reason
            )
            # Game-error recovery cannot renew its original allowance, and nothing
            # here can reproduce the rental deadline of a run that had one.
            if game_error_recovery and parent.allocation_deadline is not None:
                raise ValueError(
                    "recovery cannot preserve the original allocation deadline"
                )
            settled = settle_orphan(parent)
            if settled != parent:
                # Preserve the old run.json and snapshots. The canonical record
                # explains settlement, including the explicitly estimated cost.
                store.save_training_run(settled)
            parent = settled
            if parent.status not in {"failed", "interrupted", "paused"}:
                raise ValueError(
                    "recovery requires a stopped failed, interrupted or paused attempt"
                )
            if (
                source_repair_reason is None
                and snapshot.identities != parent.identities
            ):
                # A setup failure can retain the pre-repair snapshot. Carry its
                # existing admission forward after checking the current runtime.
                source_repair_reason = parent.source_repair_reason
        yield Attempt(destination, parent, snapshot, source_repair_reason)


# ---------------------------------------------------------------------------
# Regime functions

# A regime: any callable taking the run, usually a function or a
# `functools.partial` binding its keyword arguments.
Regime = Callable[[Run], None]


def _signature(regime: Regime) -> tuple[Callable[..., None], dict[str, object]]:
    """The function a regime calls and every keyword argument it receives."""
    if isinstance(regime, functools.partial):
        if regime.args:
            raise TypeError("bind regime arguments by name")
        function, bound = regime.func, dict(regime.keywords)
    else:
        function, bound = regime, {}
    if not inspect.isfunction(function) or "<locals>" in function.__qualname__:
        raise TypeError("a regime is a module-level function, or a partial of one")
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("a regime function takes the run as its first argument")
    hints = get_type_hints(function)
    arguments: dict[str, object] = {}
    for parameter in parameters[1:]:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("regime function arguments are named")
        if parameter.name in bound:
            value = bound.pop(parameter.name)
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            raise TypeError(f"regime argument {parameter.name!r} is not bound")
        if parameter.name not in hints:
            raise TypeError(f"annotate regime argument {parameter.name!r}")
        # Validation returns typed instances unchanged and types JSON values;
        # it rejects a wrong type here, before a run opens.
        arguments[parameter.name] = TypeAdapter(hints[parameter.name]).validate_python(
            value
        )
    if bound:
        raise TypeError(f"unknown regime arguments: {', '.join(sorted(bound))}")
    return function, arguments


def regime_record(
    regime: Regime, *, wall_seconds: float, recovery: RecoveryPolicy | None
) -> RegimeFunctionRecord:
    """What a run of `regime` records as its identity.

    Arguments must serialize to JSON (numbers, strings, the typed records and
    collections of them); a lambda or an open file is rejected.
    """
    function, arguments = _signature(regime)
    try:
        encoded = to_jsonable_python(arguments)
    except PydanticSerializationError as error:
        raise TypeError(f"regime arguments must serialize to JSON: {error}") from error
    return RegimeFunctionRecord(
        function=f"{function.__module__}:{function.__qualname__}",
        arguments=encoded,
        arguments_sha256=canonical_sha256(encoded),
        revision=source_revision(),
        wall_seconds=wall_seconds,
        recovery=recovery,
    )


def load_regime(path: str, arguments: Mapping[str, object]) -> Regime:
    """The function at `module:qualname` bound to JSON `arguments`.

    `execute` validates each argument against the function's annotation, so a
    recorded or remote regime rebinds to the same typed values.
    """
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError("a regime function path is module:function")
    target: Any = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    return functools.partial(target, **arguments)


def source_revision() -> SourceRevision:
    """The commit and uncommitted tracked changes this process runs from."""
    try:
        commit: str | None = source_commit()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return SourceRevision(commit=commit, uncommitted_sha256=source_uncommitted_sha256())


def host_identities() -> dict[str, object]:
    """Machine, Torch and training-source identities every attempt records."""
    return {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "training_source_sha256": source_bundle_sha256(
            sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        ),
        "torch": torch.__version__,
        "policy_numerics": "masked-log-softmax-v1",
        "source_commit": source_commit(),
    }


def _function_identities(seed: int) -> dict[str, object]:
    # Opening the world fixes content and input identities. A function run's
    # setups are its stages' arguments, never a representative pair.
    identities = runtime_fingerprints(seed)
    del identities["matchup"], identities["matchup_sha256"]
    return identities | host_identities()


@dataclass(frozen=True)
class ModelCheckpoints:
    """A model's latest published checkpoints of its final weights."""

    raw: ArtifactReference | None
    ema: ArtifactReference | None


@dataclass(frozen=True)
class RunResult:
    """A finished regime-function run: its record and each model's checkpoints."""

    record: TrainingRun
    checkpoints: dict[str, ModelCheckpoints]

    @property
    def status(self) -> str:
        return self.record.status


def execute(
    regime: Regime,
    *,
    out: str | Path,
    seed: int,
    store: "VerifyStore | None" = None,
    resume_from: str | None = None,
    wall_seconds: float = 900.0,
    recovery: RecoveryPolicy | None = None,
    checkpoint_updates: int | None = None,
    pause_path: Path | None = None,
    stop_path: Path | None = None,
    experiment: str | None = None,
    smoke_execution: Execution | None = None,
) -> RunResult:
    """Run `regime(run)` as one attempt, or resume a stopped one.

    Without `store` the run opens `training.sqlite` beside `out`. Resuming
    requires the same function, arguments, revision, supervision and seed;
    the function re-enters from the top, completed stages return their
    recorded results, and the interrupted stage continues from its last
    recovery snapshot at an update boundary.
    """
    if pause_path is not None and recovery is None:
        raise ValueError("safe pause requires complete-state recovery")
    function, arguments = _signature(regime)
    record = regime_record(regime, wall_seconds=wall_seconds, recovery=recovery)
    digest = canonical_sha256(record.model_dump(mode="json"))
    with ExitStack() as stack:
        if store is None:
            store = stack.enter_context(
                VerifyStore(Path(out).resolve().parent / "training.sqlite")
            )
        with attempt(
            store,
            out,
            seed=seed,
            regime_digest=digest,
            identities=lambda: _function_identities(seed),
            resume_from=resume_from,
            source_repair_reason=None,
            cadence=(checkpoint_updates, None, ()),
            game_error_recovery=False,
        ) as admitted:
            run = open_run(
                store,
                admitted.out,
                seed=seed,
                regime=None,
                regime_function=record,
                regime_digest=digest,
                wall_seconds=wall_seconds,
                recovery=recovery,
                recoverable=recovery is not None,
                game_errors=None,
                parent=admitted.parent,
                snapshot=admitted.snapshot,
                checkpoint_updates=checkpoint_updates,
                pause_path=pause_path,
                stop_path=stop_path,
                experiment=experiment or record.function.partition(":")[2],
                smoke_execution=smoke_execution,
            )
            result = run.execute(
                lambda: _function_identities(seed),
                lambda: function(run, **arguments),
            )
            return RunResult(
                result,
                {
                    name: ModelCheckpoints(
                        raw=model.exports.get("raw"), ema=model.exports.get("ema")
                    )
                    for name, model in run.models.items()
                },
            )


def smoke(
    regime: Regime,
    *,
    out: str | Path,
    seed: int = 0,
    store: "VerifyStore | None" = None,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> RunResult:
    """Run `regime` in its tiny form: real, cheap, written to `out`.

    Every stage does one update of one transition per stream (at most one
    game per stream) on `device`, and exports and reloads its checkpoints
    through admission, so `out` holds real, loadable checkpoints for an
    evaluation's smoke games. A bad deck, capacity, world mismatch or error in
    the function fails here in seconds. A smoke run says nothing about speed
    or learning.
    """
    return execute(
        regime,
        out=out,
        seed=seed,
        store=store,
        smoke_execution=Execution(device=device),
    )
