"""Measured run records, separate from executable recipe definitions.

RunRecord shares retained measurements across execution and report readers.
TrainingRun binds those measurements to its recipe: a currently admitted
TrainingRegime, or the RegimeFunctionRecord of a regime function.
"""

from typing import Any, Literal

from pydantic import Field, model_validator

from manabot.infra.host import HostFacts
from manabot.training.regime import (
    ArtifactReference,
    LearningStateOrigin,
    ProducerCost,
    RecoveryPolicy,
    Strict,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.state.game_errors import GameErrorState


class NumericalFailure(Strict):
    """A rejected update is not a completed training diagnostic coordinate."""

    invariant: str
    iteration: int
    health: dict[str, int | float] = {}


class PhaseTiming(Strict):
    """Accumulated host wall seconds and calls; never device utilization.

    Phase boundaries sit at points that already wait for the device, so a
    phase includes the device work it queued. Nested phases sum to at most
    their parent; the remainder is untimed work inside the parent. Native
    stepping and observation encoding come from the engine's own clock, as the
    mean across parallel producer shards.
    """

    seconds: float = Field(ge=0, allow_inf_nan=False)
    calls: int = Field(ge=0)


class SourceRevision(Strict):
    """The commit a run executed and the digest of uncommitted tracked changes."""

    commit: str | None
    # SHA-256 of `git diff HEAD` over tracked files; None when clean.
    uncommitted_sha256: str | None = None


class RegimeFunctionRecord(Strict):
    """A regime function's identity: what ran, with which arguments.

    `function` is the callable's `module:qualname`. `arguments` holds every
    keyword argument after the run (bound by a `functools.partial`, or a
    default) as JSON. Two runs with equal `function`, `arguments_sha256` and
    `revision` ran the same recipe.
    """

    function: str = Field(pattern=r"^[\w.]+:[\w.]+$")
    arguments: dict[str, Any]
    arguments_sha256: str
    revision: SourceRevision
    # Run-level supervision, part of the identity a resumed attempt must match.
    wall_seconds: float = Field(gt=0)
    recovery: RecoveryPolicy | None = None


StageOperation = Literal[
    "train_self_play",
    "import_policy",
    "collect_belief",
    "train_belief",
    "export",
    "custom",
]


class StageRecord(Strict):
    # What a regime function's stage did and the arguments it resolved; a
    # saved regime's stages declare both in the regime.
    operation: StageOperation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    arguments: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    learning_state_origin: LearningStateOrigin | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    numerical_failure: NumericalFailure | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Cumulative across attempts for this stage; run costs remain attempt-local.
    watchdog_seconds: float = 0
    id: str
    producer_cost: ProducerCost | None = None
    actual_device: str | None = None
    actual_threads: int | None = None
    status: Literal["running", "completed", "failed", "interrupted", "paused"] = (
        "running"
    )
    seconds: float = 0
    cumulative_seconds: float | None = Field(default=None, ge=0)
    collection_seconds: float = 0
    learning_seconds: float = 0
    export_seconds: float = 0
    diagnostic_seconds: float = 0
    # Attempt-local learner wall time waiting for initial evaluation admission.
    # None means historical/unsupported measurement, rather than zero wait.
    admission_wait_seconds: float | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    collection_timings: dict[str, PhaseTiming] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Learning subphases keyed like collection's ("Minibatches/Forward and
    # loss"); None for records written before learning was timed.
    learning_timings: dict[str, PhaseTiming] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Completed update iterations in this segment: self-play collect/update
    # iterations. Imported learning-state iterations are
    # counted by learning_state_origin, not here.
    updates: int = Field(default=0, ge=0)
    games: int = 0
    environment_decisions: int = 0
    # Slots idled while slower streams closed the batch; 0 in older records.
    idle_stream_steps: int = Field(default=0, ge=0, exclude_if=lambda value: not value)
    learner_transitions: int = 0
    optimizer_exposures: int = 0
    # Completed optimizer.step calls; unknown in older or unsupported learners.
    optimizer_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    sampled_peak_rss_bytes: int = 0
    cpu_seconds: float = 0
    inputs: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    rejected_artifacts: dict[str, dict] = {}
    # Bounded rows of stages Etude does not track: selection or belief
    # summaries (historical records may also hold compound summaries).
    # Self-play update rows are Etude metrics only (TrainingRun.metrics_run);
    # historical exports retain theirs here.
    diagnostics: list[dict] = []
    error: str | None = None


class TrainingCoordinates(Strict):
    # Segment-local completed iterations; updates includes inherited absolute offsets.
    segment_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    optimizer_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    stage_id: str
    updates: int = Field(ge=0)
    training_seconds: float = Field(ge=0)
    active_training_seconds: float | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    environment_decisions: int = Field(default=0, ge=0)
    learner_transitions: int = Field(default=0, ge=0)
    optimizer_exposures: int = Field(default=0, ge=0)
    games: int = Field(default=0, ge=0)


class MonitoringCheckpoint(TrainingCoordinates):
    ordinal: int
    artifact: ArtifactReference | None = None
    error: str | None = None


class RunRecord[RegimeT](Strict):
    """Shared measured run state; each reader chooses its recipe admission type."""

    game_error_ledgers: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    game_errors: GameErrorState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_repair_reason: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    schema_version: Literal[1] = 1
    id: str
    regime_digest: str
    # Exactly one recipe: a saved regime, or the call of a regime function.
    regime: RegimeT | None = Field(default=None, exclude_if=lambda v: v is None)
    regime_function: RegimeFunctionRecord | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    seed: int
    seed_streams: dict[str, int]
    identities: dict
    # This attempt's machine; absent in runs recorded before it was read.
    host: HostFacts | None = Field(default=None, exclude_if=lambda value: value is None)
    status: Literal[
        "pending", "running", "completed", "failed", "interrupted", "paused"
    ] = "pending"
    stages: list[StageRecord] = []
    seconds: float = 0
    setup_seconds: float = 0
    parent_run_id: str | None = None
    recovery_artifact: ArtifactReference | None = None
    recovery_lock_path: str | None = None
    recovery_host: str | None = None
    last_recorded_wall_seconds: float | None = None
    last_recorded_active_seconds: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recovery_boot_identity: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    calendar_seconds: float = Field(default=0, exclude_if=lambda value: value == 0)
    downtime_seconds: float = Field(default=0, exclude_if=lambda value: value == 0)
    unobserved_seconds: float = 0
    recovery_seconds: float = 0
    prior_seconds: float = 0
    watchdog_seconds: float = 0
    prior_watchdog_seconds: float = 0
    selected_artifact: dict | None = None
    error: str | None = None
    monitoring_checkpoints: list[MonitoringCheckpoint] = []
    monitoring_export_seconds: float = 0
    monitoring_checkpoint_milestones: tuple[int, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    monitoring_checkpoint_seconds: float | None = None
    monitoring_checkpoint_updates: int | None = Field(
        default=None, gt=0, strict=True, exclude_if=lambda value: value is None
    )
    allocation_deadline: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Folder of the Etude run holding self-play update metrics, named for the
    # run. Recovery children continue their parent's run in place, so one
    # Etude run covers every attempt; readers find it by name under a source.
    metrics_run: str | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="before")
    @classmethod
    def _drop_empty_fixed_validation(cls, data: Any) -> Any:
        """Read hash-bound exports from when runs carried a distillation cohort.

        Producer and parent run.json files are admitted by digest and cannot be
        rewritten. Every non-distillation export holds `"fixed_validation": null`.
        """
        if isinstance(data, dict) and data.get("fixed_validation", 0) is None:
            return {k: v for k, v in data.items() if k != "fixed_validation"}
        return data

    @model_validator(mode="before")
    @classmethod
    def _count_retained_updates(cls, data: Any) -> Any:
        """Exports written before StageRecord.updates kept one row per update.

        A continuation admits such a parent by digest, so its absolute update
        comes from those inline rows rather than a rewritten record.
        """
        if isinstance(data, dict) and isinstance(data.get("stages"), list):
            data = dict(data)
            data["stages"] = [
                {"updates": len(stage["diagnostics"]), **stage}
                if isinstance(stage, dict)
                and "updates" not in stage
                and isinstance(stage.get("diagnostics"), list)
                else stage
                for stage in data["stages"]
            ]
        return data

    @model_validator(mode="after")
    def _one_recipe(self) -> "RunRecord[RegimeT]":
        if (self.regime is None) == (self.regime_function is None):
            raise ValueError("a run records exactly one regime or regime call")
        return self

    @property
    def recovery_policy(self) -> RecoveryPolicy | None:
        """The run's active recovery policy, whichever recipe declared it."""
        if self.regime_function is not None:
            return self.regime_function.recovery
        return getattr(self.regime, "recovery", None)

    def stage_operations(self) -> dict[str, str]:
        """Each recorded stage's operation, from its declaration or record."""
        if self.regime is None:
            return {s.id: s.operation or "custom" for s in self.stages}
        return {
            s.id: s.operation
            for s in getattr(self.regime, "stages", ())
            if isinstance(getattr(s, "operation", None), str)
        }

    def _updates_through(self, learning: set[str], stage_id: str | None) -> int:
        total = 0
        for stage in self.stages:
            if stage.id in learning:
                total += stage.updates
                if stage.learning_state_origin is not None:
                    total += stage.learning_state_origin.iteration
            if stage.id == stage_id:
                break
        return total

    def optimizer_updates_through(self, stage_id: str | None = None) -> int | None:
        """Cumulative measured calls; any missing stage/ancestor remains unavailable."""
        total = 0
        for stage in self.stages:
            if stage.optimizer_updates is None:
                return None
            total += stage.optimizer_updates
            origin = stage.learning_state_origin
            if origin is not None:
                if origin.optimizer_updates is None:
                    return None
                total += origin.optimizer_updates
            if stage.id == stage_id:
                break
        return total


class TrainingRun(RunRecord[TrainingRegime]):
    """A run whose recipe passes current execution admission."""

    def updates_through(self, stage_id: str | None = None) -> int:
        """Completed learner iterations; collection records are not updates."""
        if self.regime is None:
            learning = {s.id for s in self.stages if s.operation == "train_self_play"}
        else:
            learning = {
                s.id for s in self.regime.stages if isinstance(s, TrainSelfPlay)
            }
        return self._updates_through(learning, stage_id)
