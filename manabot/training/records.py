"""Measured run records, separate from executable recipe definitions.

RunRecord shares retained measurements across execution and report readers.
TrainingRun binds those measurements to a currently admitted TrainingRegime.
"""

from typing import Literal

from pydantic import Field

from manabot.training.regime import (
    ArtifactReference,
    LearningStateOrigin,
    ProducerCost,
    Strict,
    TrainCompound,
    TrainingRegime,
    TrainSelfPlay,
    TrainSupervised,
)
from manabot.training.state.game_errors import GameErrorState


class NumericalFailure(Strict):
    """A rejected update is not a completed training diagnostic coordinate."""

    invariant: str
    iteration: int
    health: dict[str, int | float] = {}


class PhaseTiming(Strict):
    """Accumulated host wall seconds and calls; never device utilization."""

    seconds: float = Field(ge=0, allow_inf_nan=False)
    calls: int = Field(ge=0)


class StageRecord(Strict):
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
    games: int = 0
    environment_decisions: int = 0
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
    diagnostics: list[dict] = []
    error: str | None = None

    def observed_optimizer_updates(self) -> int | None:
        """Read saved counters once at import/recovery, never infer from exposures."""
        if self.optimizer_updates is not None:
            return self.optimizer_updates
        total = 0
        for diagnostic in self.diagnostics:
            health = diagnostic.get("numerical")
            count = health.get("optimizer_steps") if isinstance(health, dict) else None
            if type(count) is not int or count < 0:
                return None
            total += count
        return total if self.diagnostics else None


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


class FixedValidationCohort(Strict):
    artifact: ArtifactReference
    games: list[int]
    policy_target_kind: str
    source_inputs: dict[str, ArtifactReference]


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
    regime: RegimeT
    seed: int
    seed_streams: dict[str, int]
    identities: dict
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
    fixed_validation: FixedValidationCohort | None = None

    def _updates_through(self, learning: set[str], stage_id: str | None) -> int:
        total = 0
        for stage in self.stages:
            if stage.id in learning:
                total += len(stage.diagnostics)
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
        """Completed learner iterations/epochs; collection records are not updates."""
        learning = {
            s.id
            for s in self.regime.stages
            if isinstance(s, (TrainSelfPlay, TrainSupervised, TrainCompound))
        }
        return self._updates_through(learning, stage_id)
