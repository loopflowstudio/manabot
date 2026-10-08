"""Validated training recipes and durable execution records."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.sim.local_update import LocalSearchConfig


class ArtifactReference(TypedDict):
    """Immutable local bytes admitted by content digest."""

    path: str
    sha256: str
    bytes: int


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Schedule(Strict):
    initial: float = Field(ge=0)
    decay: float = Field(default=0, ge=0)
    power: float = Field(default=1, ge=0)

    def at(self, progress: float) -> float:
        return (
            self.initial
            * (1 + self.decay * min(1.0, max(0.0, progress))) ** -self.power
        )


class Learning(Strict):
    gradient: Literal["ppo"] = "ppo"
    gamma: float = Field(default=1, ge=0, le=1)
    policy_lambda: float = Field(default=0.95, ge=0, le=1)
    value_lambda: float = Field(default=1, ge=0, le=1)
    filter_kind: Literal["top_count", "quantile"] = "top_count"
    filter_scope: Literal["actor_critic", "actor"] = "actor_critic"
    retained_fraction: float = Field(default=0.5, gt=0, le=1)
    min_advantage: float = Field(default=0, ge=0)
    reference: Literal["uniform", "action_type_uniform"] = "uniform"
    learning_rate: Schedule = Schedule(initial=2.5e-4, decay=9)
    tau: Schedule = Schedule(initial=0.01, decay=9)
    collection_kl: float = Field(default=0.1, ge=0)
    clip: float = Field(default=0.1, gt=0)
    value_weight: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    epochs: int = Field(default=4, ge=1)
    minibatches: int = Field(default=4, ge=1)
    ema: float | None = Field(default=None, ge=0, lt=1)


class AtaraxosMoveLearning(Strict):
    """Supplement S3.4 move update; MTG reference and batch sizes are adaptations.

    Iteration schedules, one epoch and no advantage normalization belong to
    this method, rather than inheriting the PPO control's elapsed-time recipe.
    """

    gradient: Literal["ataraxos_move"]
    policy_lambda: float = Field(default=0.5, ge=0, le=1)
    value_lambda: float = Field(default=0.8, ge=0, le=1)
    filter_kind: Literal["quantile", "top_count"] = "quantile"
    filter_scope: Literal["actor_critic", "actor"] = "actor_critic"
    advantage_quantile: float = Field(default=0.75, ge=0, le=1)
    min_advantage: float = Field(default=0.01, ge=0)
    reference: Literal["uniform", "action_type_uniform"] = "action_type_uniform"
    clip: float = Field(default=0.2, gt=0, lt=1)
    collection_kl: float = Field(default=0.1, ge=0)
    learning_rate_scale: float = Field(default=0.5, gt=0)
    learning_rate_power: float = Field(default=1.1, ge=0)
    learning_rate_min: float = Field(default=5e-6, gt=0)
    learning_rate_max: float = Field(default=1e-4, gt=0)
    tau_scale: float = Field(default=0.05, ge=0)
    tau_power: float = Field(default=0.3, ge=0)
    max_grad_norm: float = Field(default=0.267, gt=0)
    ema: float | None = Field(default=0.999, ge=0, lt=1)

    @model_validator(mode="after")
    def valid_learning_rate_bounds(self) -> "AtaraxosMoveLearning":
        if self.learning_rate_min > self.learning_rate_max:
            raise ValueError("learning rate minimum exceeds maximum")
        return self

    def rates(self, iteration: int) -> tuple[float, float]:
        """One-based completed collection/update iteration, including skips."""
        if iteration < 1:
            raise ValueError("Ataraxos iteration must be positive")
        rate = self.learning_rate_scale / iteration**self.learning_rate_power
        return (
            min(self.learning_rate_max, max(self.learning_rate_min, rate)),
            self.tau_scale / iteration**self.tau_power,
        )


class Execution(Strict):
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32"] = "float32"
    workers: Literal[1] = 1
    threads: int = Field(default=1, ge=1, le=4)
    wall_seconds: float = Field(default=300, gt=0)
    memory_bytes: int = Field(default=32 * 1024**3, gt=0)


class Stage(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    execution: Execution = Execution()


class ImportPolicy(Stage):
    """Admit one published producer export and exact raw or EMA checkpoint."""

    operation: Literal["import_policy"]
    source_run: ArtifactReference
    source_stage: str
    checkpoint: ArtifactReference
    weights: Literal["raw", "ema"] = "raw"


class ProducerCost(Strict):
    """Sunk cost through the source checkpoint, never fresh execution cost."""

    run_id: str
    stage_id: str
    weights: Literal["raw", "ema"]
    cumulative_seconds: float = Field(ge=0)


class CollectSearch(Stage):
    operation: Literal["collect_search"]
    policy: Literal["uniform-prior-determinized-puct"] = (
        "uniform-prior-determinized-puct"
    )
    target: Literal["visit_distribution"] = "visit_distribution"
    games: int = Field(ge=2)
    simulations: int = Field(default=64, ge=1)
    worlds: int = Field(default=4, ge=1)
    max_steps: int = Field(default=2000, ge=1)


class CollectLocalUpdate(Stage):
    """Freeze an admitted raw/EMA policy for sampling-based local targets."""

    operation: Literal["collect_local_update"]
    policy: str
    weights: Literal["raw", "ema"] = "raw"
    games: int = Field(ge=2)
    max_steps: int = Field(default=2000, ge=1)
    search: LocalSearchConfig = LocalSearchConfig()
    sampler: str | None = None


class TrainSupervised(Stage):
    operation: Literal["train_supervised"]
    trainer: Literal["search_supervised"] = "search_supervised"
    optimizer: Literal["adam"] = "adam"
    trainable: Literal["policy"] = "policy"
    target: Literal[
        "visit_distribution", "local_soft", "local_argmax", "local_allocation"
    ] = "visit_distribution"
    datasets: list[str] = Field(min_length=1)
    initial: str | None = None
    # Omit the historical default so existing recipe identities stay unchanged.
    initial_weights: Literal["raw", "ema"] = Field(
        default="raw", exclude_if=lambda value: value == "raw"
    )
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)


class FrozenOpponent(Strict):
    """Exact admitted policy bytes used only for opponent-seat inference."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LearningStateImport(Strict):
    """Portable learner state; source bytes retain historical paths and identity.

    Paths here locate staged copies. Their hashes/sizes must match the parent
    TrainingRun. Game streams and all RNGs explicitly start afresh.
    """

    source_run: ArtifactReference
    source_stage: str
    raw: ArtifactReference
    ema: ArtifactReference
    optimizer: ArtifactReference


class LearningStateOrigin(Strict):
    """Absolute parent coordinates, separate from this segment's work/cost."""

    run_id: str
    stage_id: str
    iteration: int = Field(ge=1)
    games: int = Field(ge=0)
    environment_decisions: int = Field(ge=0)
    learner_transitions: int = Field(ge=0)
    optimizer_exposures: int = Field(ge=0)
    active_training_seconds: float = Field(ge=0)
    cumulative_seconds: float = Field(ge=0)
    streams: Literal["fresh"] = "fresh"


class TrainSelfPlay(Stage):
    operation: Literal["train_self_play"]
    trainer: Literal["net_opponent"] = "net_opponent"
    optimizer: Literal["adam"] = "adam"
    trainable: Literal["policy_value"] = "policy_value"
    behavior: Literal["current-self", "ema-self", "frozen"] = "current-self"
    opponent: FrozenOpponent | None = None
    initial: str | None = None
    learning_state: LearningStateImport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    updates: int = Field(default=2, ge=1)
    # Frozen v1 compatibility only; JobSpec rejects active-time endpoints.
    # An active-time endpoint completes after a whole collect/update iteration.
    # updates remains a safety ceiling; reaching it early is a failed attempt.
    active_seconds: float | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    streams: int = Field(default=4, ge=2)
    transitions: int = Field(default=256, ge=1)
    learning: Learning | AtaraxosMoveLearning = Learning()

    @model_validator(mode="after")
    def valid_behavior(self) -> "TrainSelfPlay":
        if self.learning_state is not None and (
            self.initial is not None
            or self.active_seconds is not None
            or not isinstance(self.learning, AtaraxosMoveLearning)
            or self.learning.ema is None
        ):
            raise ValueError(
                "learning-state continuation requires absolute-step Ataraxos with EMA"
            )
        if self.active_seconds is not None:
            if self.active_seconds >= self.execution.wall_seconds:
                raise ValueError("active endpoint requires additional watchdog reserve")
            if not isinstance(self.learning, AtaraxosMoveLearning):
                raise ValueError(
                    "active endpoint currently requires iteration-based Ataraxos learning"
                )
        if self.behavior == "ema-self" and self.learning.ema is None:
            raise ValueError("ema-self behavior requires an EMA rate")
        if (self.behavior == "frozen") != (self.opponent is not None):
            raise ValueError("frozen behavior requires exactly one frozen opponent")
        if self.behavior == "frozen" and self.streams % 2:
            raise ValueError(
                "frozen opponent training requires even streams for both decks"
            )
        return self


class SelectionGameSpec(Strict):
    """Population and split membership frozen before observing any outcome."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    seed: int = Field(ge=0, le=2**32 - 1)
    action_seed: int = Field(ge=0, le=2**32 - 1)
    assignment: Literal[0, 1]
    split: Literal["development", "held_out"]


class CollectSelection(Stage):
    """Diagnostic complete games from a fixed admitted self-play policy.

    source_run refers to VerifyStore, never an unverified run JSON export.
    Selection is recomputed per partition over its complete rows, not fitted.
    """

    operation: Literal["collect_selection"]
    policy: str
    source_run: str | None = None
    weights: Literal["raw", "ema"] = "raw"
    population: tuple[SelectionGameSpec, ...] = Field(min_length=4)
    max_steps: int = Field(default=2000, ge=1)
    learning: Learning | AtaraxosMoveLearning = Learning()
    bootstrap_samples: int = Field(default=500, ge=100, le=10000)
    bootstrap_seed: int = Field(default=93, ge=0)

    @model_validator(mode="after")
    def valid_population(self) -> "CollectSelection":
        if len({game.seed for game in self.population}) != len(self.population):
            raise ValueError("selection population requires unique deal seeds")
        for split in ("development", "held_out"):
            if {g.assignment for g in self.population if g.split == split} != {0, 1}:
                raise ValueError("each whole-game split requires both deck assignments")
        return self


class CollectBelief(Stage):
    """Freeze one admitted policy artifact before collecting private labels."""

    operation: Literal["collect_belief"]
    policy: str
    weights: Literal["raw", "ema"] = "raw"
    games: int = Field(ge=3)
    max_steps: int = Field(default=2000, ge=1)


class TrainBelief(Stage):
    """Fit only the sampler; policy weights are never updated by belief NLL."""

    operation: Literal["train_belief"]
    dataset: str
    steps: int = Field(default=32, ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_size: int = Field(default=32, ge=2)
    learning_rate: float = Field(default=0.001, gt=0)
    history_dropout: float = Field(default=0, ge=0, le=1)
    evaluation_samples: int = Field(default=32, ge=1)


class TrainCompound(Stage):
    """Complete-game self-play with explicit decoder credit units."""

    operation: Literal["train_compound"]
    initial: str | None = None
    updates: int = Field(default=1, ge=1)
    games_per_update: int = Field(default=2, ge=1)
    max_commands: int = Field(default=4000, ge=1)
    grouping: Literal["sequential", "grouped"] = "grouped"
    estimator: Literal["outcome", "bootstrapped"] = "outcome"
    skip_trivial: bool = True
    learning: Learning = Learning(retained_fraction=1, epochs=1)

    @model_validator(mode="after")
    def supported_learning(self) -> "TrainCompound":
        if (
            self.learning.filter_kind != "top_count"
            or self.learning.filter_scope != "actor_critic"
        ):
            raise ValueError(
                "compound stages support top-count actor-critic filtering only"
            )
        if self.learning.reference != "uniform" or self.learning.ema is not None:
            raise ValueError(
                "compound stages require conditional uniform reference and raw weights"
            )
        return self


Operation = Annotated[
    ImportPolicy
    | CollectSearch
    | CollectLocalUpdate
    | TrainSupervised
    | TrainSelfPlay
    | TrainCompound
    | CollectBelief
    | CollectSelection
    | TrainBelief,
    Field(discriminator="operation"),
]


class TrainingRegime(Strict):
    schema_version: Literal[1] = 1
    id: str
    world: str
    match: MatchHypers
    observation: ObservationSpaceHypers = ObservationSpaceHypers()
    agent: AgentSpec = AgentSpec()
    stages: list[Operation] = Field(min_length=1)
    wall_seconds: float = Field(default=900, gt=0)
    schedule_clock: Literal["run_elapsed_budget", "iteration_fraction"] = (
        "run_elapsed_budget"
    )
    recovery_max_microsteps: int | None = Field(default=None, ge=1, le=1_000_000)
    selection: Literal["last-complete-raw"] = "last-complete-raw"

    @model_validator(mode="after")
    def references(self) -> "TrainingRegime":
        if any(
            isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
            for stage in self.stages
        ) and (
            len(self.stages) != 1
            or self.recovery_max_microsteps is not None
            or self.schedule_clock != "iteration_fraction"
        ):
            raise ValueError(
                "learning-state continuation requires one step-target stage without process recovery"
            )
        if self.agent.recent_events and self.agent.semantic_pack is None:
            raise ValueError(
                "recent events require a semantic pack for public identities"
            )
        if self.agent.recent_events and not 1 <= self.observation.max_events <= 32:
            raise ValueError("public history v1 requires 1..32 event rows")
        # The model treatment owns this input contract; callers cannot silently
        # train history-on against the historical drained event tensor.
        self.observation = self.observation.model_copy(
            update={"policy_history_version": 1 if self.agent.recent_events else 0}
        )
        has_compound = any(isinstance(stage, TrainCompound) for stage in self.stages)
        has_import = any(isinstance(stage, ImportPolicy) for stage in self.stages)
        if (
            has_compound
            and not self.agent.compound_decisions
            or (self.agent.compound_decisions and not (has_compound or has_import))
        ):
            raise ValueError(
                "compound stages and compound Agent must be selected together"
            )
        if self.agent.compound_decisions and any(
            not isinstance(
                stage,
                (
                    TrainCompound,
                    ImportPolicy,
                    CollectBelief,
                    TrainBelief,
                    CollectLocalUpdate,
                ),
            )
            for stage in self.stages
        ):
            raise ValueError(
                "compound policies require compound policy stages throughout the run"
            )
        if self.recovery_max_microsteps is not None and (
            any(not isinstance(stage, TrainSelfPlay) for stage in self.stages)
            or self.schedule_clock != "iteration_fraction"
        ):
            raise ValueError(
                "recovery requires only self-play stages and iteration_fraction schedule"
            )
        if self.recovery_max_microsteps is not None and any(
            isinstance(stage, TrainSelfPlay) and stage.active_seconds is not None
            for stage in self.stages
        ):
            raise ValueError("active-time process recovery is unsupported")
        if any(stage.execution.device == "cuda" for stage in self.stages):
            if self.recovery_max_microsteps is not None:
                raise ValueError("CUDA process recovery is unsupported")
            if self.agent.belief_count_buckets or any(
                not isinstance(stage, TrainSelfPlay) or stage.opponent is not None
                for stage in self.stages
            ):
                raise ValueError("CUDA requires self-contained ordinary self-play")
        previous: dict[str, Stage] = {}
        latest_self_play = None
        latest_compound = None
        for stage in self.stages:
            if stage.id in previous:
                raise ValueError("stage IDs must be unique")
            if isinstance(stage, CollectSelection) and stage.source_run is None:
                policy = previous.get(stage.policy)
                if not isinstance(policy, TrainSelfPlay):
                    raise ValueError("selection requires an earlier self-play policy")
                if stage.weights == "ema" and policy.learning.ema is None:
                    raise ValueError("selection EMA dependency requires EMA weights")
            if isinstance(stage, (CollectBelief, CollectLocalUpdate)):
                policy = previous.get(stage.policy)
                if not isinstance(
                    policy,
                    (TrainSupervised, TrainSelfPlay, TrainCompound, ImportPolicy),
                ):
                    raise ValueError(
                        "belief policy must refer to an earlier policy stage"
                    )
                if isinstance(policy, ImportPolicy) and stage.weights != policy.weights:
                    raise ValueError("imported policy weights differ from consumer")
                if (
                    stage.weights == "ema"
                    and not isinstance(policy, ImportPolicy)
                    and (
                        not isinstance(policy, TrainSelfPlay)
                        or policy.learning.ema is None
                    )
                ):
                    raise ValueError(
                        "belief EMA dependency requires an EMA policy artifact"
                    )
            if isinstance(stage, CollectLocalUpdate):
                if (stage.search.sampling == "learned") != (stage.sampler is not None):
                    raise ValueError("learned local search requires a sampler stage")
                if stage.sampler is not None:
                    sampler = previous.get(stage.sampler)
                    if not isinstance(sampler, TrainBelief):
                        raise ValueError(
                            "local sampler must refer to an earlier train_belief stage"
                        )
                    data = previous.get(sampler.dataset)
                    if not isinstance(data, CollectBelief) or (
                        data.policy,
                        data.weights,
                    ) != (stage.policy, stage.weights):
                        raise ValueError(
                            "local sampler must use the same frozen policy and weights"
                        )
            if isinstance(stage, TrainBelief) and not isinstance(
                previous.get(stage.dataset), CollectBelief
            ):
                raise ValueError(
                    "belief dataset must refer to an earlier belief collection"
                )
            if isinstance(stage, TrainSupervised):
                if len(stage.datasets) != len(set(stage.datasets)):
                    raise ValueError("dataset references must be unique")
                for ref in stage.datasets:
                    if not isinstance(
                        previous.get(ref), (CollectSearch, CollectLocalUpdate)
                    ):
                        raise ValueError(
                            f"dataset {ref} must refer to an earlier collection"
                        )
                    if stage.target.startswith("local_") != isinstance(
                        previous[ref], CollectLocalUpdate
                    ):
                        raise ValueError(
                            "supervised target must match collection semantics"
                        )
            initial = getattr(stage, "initial", None)
            parent = previous.get(initial)
            if isinstance(stage, TrainSupervised):
                if isinstance(parent, ImportPolicy):
                    if stage.initial_weights != parent.weights:
                        raise ValueError(
                            "imported initial weights differ from consumer"
                        )
                elif stage.initial_weights != "raw":
                    raise ValueError("EMA initialization requires an imported policy")
            if isinstance(stage, TrainSupervised) and stage.target.startswith("local_"):
                if not isinstance(
                    parent, (TrainSelfPlay, TrainSupervised, ImportPolicy)
                ) or (
                    isinstance(parent, TrainSupervised)
                    and not parent.target.startswith("local_")
                ):
                    raise ValueError(
                        "local distillation requires an earlier signed-value policy"
                    )
            elif (
                initial
                and type(parent) is not type(stage)
                and not (
                    isinstance(stage, TrainSupervised)
                    and isinstance(parent, ImportPolicy)
                )
            ):
                raise ValueError(
                    "continuation requires an earlier stage of the same operation"
                )
            if isinstance(stage, TrainSelfPlay) and initial:
                if parent is not latest_self_play:
                    raise ValueError(
                        "live self-play continuation cannot branch from an older collector"
                    )
                if (
                    parent.execution.device != stage.execution.device
                    or parent.behavior != stage.behavior
                    or parent.streams != stage.streams
                    or parent.learning.ema != stage.learning.ema
                    or parent.opponent != stage.opponent
                    or parent.learning.gradient != stage.learning.gradient
                ):
                    raise ValueError(
                        "live self-play continuation must preserve streams, gradient, EMA clock and opponent"
                    )
            if isinstance(stage, TrainCompound):
                if initial and parent is not latest_compound:
                    raise ValueError(
                        "compound continuation cannot branch from older weights"
                    )
                latest_compound = stage
            if self.agent.value_kind == "categorical_wdl" and (
                isinstance(stage, TrainCompound)
                or isinstance(stage, TrainSupervised)
                and not stage.target.startswith("local_")
                or isinstance(stage, TrainSelfPlay)
                and not isinstance(stage.learning, AtaraxosMoveLearning)
            ):
                raise ValueError("categorical outcome training requires ataraxos_move")
            previous[stage.id] = stage
            if isinstance(stage, TrainSelfPlay):
                latest_self_play = stage
        return self


class NumericalFailure(Strict):
    """A rejected update is not a completed training diagnostic coordinate."""

    invariant: str
    iteration: int
    health: dict[str, int | float] = {}


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
    games: int = 0
    environment_decisions: int = 0
    learner_transitions: int = 0
    optimizer_exposures: int = 0
    sampled_peak_rss_bytes: int = 0
    cpu_seconds: float = 0
    inputs: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    rejected_artifacts: dict[str, dict] = {}
    diagnostics: list[dict] = []
    error: str | None = None


class TrainingCoordinates(Strict):
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


class TrainingRun(Strict):
    schema_version: Literal[1] = 1
    id: str
    regime_digest: str
    regime: TrainingRegime
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
    unobserved_seconds: float = 0
    recovery_seconds: float = 0
    prior_seconds: float = 0
    watchdog_seconds: float = 0
    prior_watchdog_seconds: float = 0
    selected_artifact: dict | None = None
    error: str | None = None
    monitoring_checkpoints: list[MonitoringCheckpoint] = []
    monitoring_export_seconds: float = 0
    monitoring_checkpoint_seconds: float | None = None
    allocation_deadline: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fixed_validation: FixedValidationCohort | None = None

    def updates_through(self, stage_id: str | None = None) -> int:
        """Completed learner iterations/epochs; collection records are not updates."""
        learning = {
            s.id
            for s in self.regime.stages
            if isinstance(s, (TrainSelfPlay, TrainSupervised, TrainCompound))
        }
        total = 0
        for stage in self.stages:
            if stage.id in learning:
                total += len(stage.diagnostics)
                if stage.learning_state_origin is not None:
                    total += stage.learning_state_origin.iteration
            if stage.id == stage_id:
                break
        return total
