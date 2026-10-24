"""Validated recipe definitions: stages, learning settings and execution admission."""

from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.training.state.game_errors import GameErrorPolicy


class ArtifactReference(TypedDict):
    """Immutable local bytes admitted by content digest."""

    path: str
    sha256: str
    bytes: int


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CheckpointCadence(Strict):
    """Prospective exports use absolute completed iterations, anchored at zero.

    Explicit seconds preserve frozen launch records. Omitting both fields chooses
    1,000 updates; this is an authoring default, never a throughput estimate.
    """

    checkpoint_updates: int | None = Field(
        default=1000, gt=0, strict=True, exclude_if=lambda value: value is None
    )
    checkpoint_seconds: float | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )

    @model_validator(mode="before")
    @classmethod
    def select_clock(cls, value: object) -> object:
        if isinstance(value, dict):
            if value.get("checkpoint_seconds") is not None:
                if "checkpoint_updates" not in value:
                    return {**value, "checkpoint_updates": None}
            elif value.get("checkpoint_updates") is None:
                return {**value, "checkpoint_updates": 1000}
        return value

    @model_validator(mode="after")
    def one_clock(self) -> "CheckpointCadence":
        if self.checkpoint_updates is not None and self.checkpoint_seconds is not None:
            raise ValueError(
                "choose checkpoint_updates or checkpoint_seconds, not both"
            )
        return self

    @model_serializer(mode="wrap")
    def retain_legacy_field_order(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        # Job/cohort identities hash serialized bytes, including field order.
        # All existing cadence owners placed seconds directly after monitoring.
        values: dict[str, object] = handler(self)
        if self.checkpoint_updates is None and "monitoring" in values:
            seconds = values.pop("checkpoint_seconds", None)
            if seconds is not None:
                ordered: dict[str, object] = {}
                for key, value in values.items():
                    ordered[key] = value
                    if key == "monitoring":
                        ordered["checkpoint_seconds"] = seconds
                return ordered
        return values

    def admit_regime(self, regime: "TrainingRegime") -> None:
        stage = regime.stages[0]
        if self.checkpoint_updates is not None and (
            len(regime.stages) != 1
            or not isinstance(stage, TrainSelfPlay)
            or stage.active_seconds is not None
            or (
                regime.schedule_clock != "iteration_fraction"
                and stage.learning.gradient != "damped_policy"
            )
        ):
            raise ValueError(
                "step cadence requires one step-target self-play stage with iteration schedules"
            )


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


class IterationSchedule(Strict):
    """Map a declared run horizon onto a reference iteration schedule.

    Both endpoints are preserved. A one-step run uses the initial rate. The
    mapping is deterministic in completed updates, never elapsed wall time;
    continuation must retain the original horizon rather than start a new one.
    """

    steps: int = Field(ge=1, strict=True)
    reference_steps: int = Field(default=10_000, ge=2, strict=True)

    def at(self, iteration: int) -> float:
        if iteration < 1:
            raise ValueError("iteration must be positive")
        if self.steps == 1:
            return 1.0
        progress = min(1.0, (iteration - 1) / (self.steps - 1))
        return 1 + progress * (self.reference_steps - 1)


class DampedPolicyLearning(Strict):
    """Unnormalized clipped policy update with reverse-KL regularization.

    Iteration schedules, one epoch and no advantage normalization belong to
    this method, rather than inheriting the PPO control's elapsed-time recipe.
    """

    gradient: Literal["damped_policy"]
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
    iteration_schedule: IterationSchedule | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_learning_rate_bounds(self) -> "DampedPolicyLearning":
        if self.learning_rate_min > self.learning_rate_max:
            raise ValueError("learning rate minimum exceeds maximum")
        return self

    def rates(self, iteration: int) -> tuple[float, float]:
        """One-based completed collection/update iteration, including skips."""
        if iteration < 1:
            raise ValueError("Damped policy iteration must be positive")
        position = (
            self.iteration_schedule.at(iteration)
            if self.iteration_schedule is not None
            else iteration
        )
        rate = self.learning_rate_scale / position**self.learning_rate_power
        return (
            min(self.learning_rate_max, max(self.learning_rate_min, rate)),
            self.tau_scale / position**self.tau_power,
        )


class Execution(Strict):
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32"] = "float32"
    # Coarse native observation producers; policy/learning remain on one device.
    workers: Literal[1, 2, 4] = 1
    threads: int = Field(default=1, ge=1, le=4)
    wall_seconds: float = Field(default=300, gt=0)
    memory_bytes: int = Field(default=32 * 1024**3, gt=0)
    profile_collection: bool = Field(default=False, exclude_if=lambda value: not value)


class Stage(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    execution: Execution = Execution()

    @model_validator(mode="after")
    def supported_workers(self) -> "Stage":
        if (
            self.execution.profile_collection
            and getattr(self, "operation", None) != "train_self_play"
        ):
            raise ValueError("collection profiling requires train_self_play")
        if (
            self.execution.workers != 1
            and getattr(self, "operation", None) != "train_self_play"
        ):
            raise ValueError(
                "multiple producers are supported only for train_self_play"
            )
        return self


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


class FrozenOpponent(Strict):
    """Exact admitted policy bytes used only for opponent-seat inference."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PretrainedPolicy(Strict):
    """Published weights a self-play stage starts from instead of random ones.

    The producer fixes what the weights mean, so its world, ActorCritic architecture
    and observation encoding must equal this regime's. How learning proceeds is
    this run's own choice: the learning rule and its schedules, streams,
    transitions, behavior, opponent and seed may all differ from the producer.

    Only the policy is inherited. Adam moments, the EMA average, game streams
    and the iteration that drives schedules all start fresh, so iteration 1 of
    this run uses the first value of its own schedule. Use LearningStateImport
    to extend one unchanged recipe instead.
    """

    source_run: ArtifactReference
    source_stage: str
    checkpoint: ArtifactReference
    weights: Literal["raw", "ema"] = "raw"


class LearningStateImport(Strict):
    """Portable learner state; source bytes retain historical paths and identity.

    Paths here locate staged copies. Their hashes/sizes must match the parent
    TrainingRun. Game streams and all RNGs explicitly start afresh.

    `source_match` declares a deck switch: the parent trained on exactly this
    match and this segment continues on the regime's. Both must name the
    agent's own semantic pack, so card identities and the learned programs
    keep their meaning; only deck/sideboard composition changes. Omitted, the
    parent's match must equal the regime's.
    """

    source_run: ArtifactReference
    source_stage: str
    raw: ArtifactReference
    ema: ArtifactReference
    optimizer: ArtifactReference
    source_match: MatchConfig | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


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
    optimizer_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )


class TrainSelfPlay(Stage):
    game_errors: GameErrorPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    operation: Literal["train_self_play"]
    trainer: Literal["net_opponent"] = "net_opponent"
    optimizer: Literal["adam"] = "adam"
    trainable: Literal["policy_value", "none"] = "policy_value"
    behavior: Literal["current-self", "ema-self", "frozen", "random"] = "current-self"
    # Omitted defaults preserve frozen historical regime identities.
    root: Literal["lethal-target-v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    opponent: FrozenOpponent | None = None
    initial: str | None = None
    learning_state: LearningStateImport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    pretrained: PretrainedPolicy | None = Field(
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
    games_per_stream: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    learning_seats: Literal["one", "both"] = "both"
    matchup_curriculum: Literal[
        "fixed", "cross-balanced", "mirrors-balanced", "elemental-uniform"
    ] = Field(default="fixed", exclude_if=lambda value: value == "fixed")
    learning: Learning | DampedPolicyLearning = Learning()

    @model_validator(mode="before")
    @classmethod
    def default_learning_seats(cls, value: object) -> object:
        if (
            isinstance(value, dict)
            and "learning_seats" not in value
            and value.get("behavior") in ("random", "frozen")
        ):
            return {**value, "learning_seats": "one"}
        return value

    @model_validator(mode="after")
    def valid_behavior(self) -> "TrainSelfPlay":
        if self.game_errors is not None and (
            self.root is not None or self.initial is not None
        ):
            raise ValueError(
                "game error recovery requires native fresh-stage collection"
            )
        if self.learning_seats == "both" and self.behavior not in (
            "current-self",
            "ema-self",
        ):
            raise ValueError("both learning seats require self-play behavior")
        if self.execution.workers > self.streams:
            raise ValueError("producer count cannot exceed streams")
        if self.execution.workers != 1 and self.root is not None:
            raise ValueError("diagnostic roots require one producer")
        if self.matchup_curriculum == "elemental-uniform":
            if (
                self.behavior != "current-self"
                or self.execution.workers != 1
                or self.root is not None
            ):
                raise ValueError(
                    "elemental-uniform requires current-self, one producer and ordinary roots"
                )
        elif self.matchup_curriculum != "fixed":
            width = 12 if self.matchup_curriculum == "mirrors-balanced" else 4
            if self.streams % width or self.behavior != "current-self":
                raise ValueError(
                    "balanced matchups require current-self and complete stream strata"
                )
        if self.pretrained is not None and (
            self.initial is not None or self.learning_state is not None
        ):
            raise ValueError(
                "pretrained weights cannot be combined with another starting state"
            )
        if self.learning_state is not None and (
            self.initial is not None
            or self.active_seconds is not None
            or not isinstance(self.learning, DampedPolicyLearning)
            or self.learning.ema is None
        ):
            raise ValueError(
                "learning-state continuation requires absolute-step damped policy with EMA"
            )
        if self.active_seconds is not None:
            if self.active_seconds >= self.execution.wall_seconds:
                raise ValueError("active endpoint requires additional watchdog reserve")
            if not isinstance(self.learning, DampedPolicyLearning):
                raise ValueError(
                    "active endpoint currently requires iteration-based damped policy learning"
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
    learning: Learning | DampedPolicyLearning = Learning()
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
    | TrainSelfPlay
    | TrainCompound
    | CollectBelief
    | CollectSelection
    | TrainBelief,
    Field(discriminator="operation"),
]


class RecoveryPolicy(Strict):
    """Bounded current-game replay and an awake elapsed-time allocation.

    Recovery artifacts are private compressed snapshots. All created snapshots
    remain retained; cadence bounds duplicated serialization and restart work.
    """

    max_game_microsteps: int = Field(default=40000, ge=1, le=1_000_000)
    checkpoint_updates: int = Field(default=128, ge=1)


class TrainingRegime(Strict):
    schema_version: Literal[1, 2] = 2
    id: str
    world: str
    match: MatchConfig
    observation: ObservationConfig = ObservationConfig()
    agent: ModelConfig = ModelConfig()
    stages: list[Operation] = Field(min_length=1)
    wall_seconds: float = Field(default=900, gt=0)
    schedule_clock: Literal["run_elapsed_budget", "iteration_fraction"] = (
        "run_elapsed_budget"
    )
    recovery_max_microsteps: int | None = Field(default=None, ge=1, le=20_000_000)
    recovery: RecoveryPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Cadence affects replayable progress only, never optimizer/collector updates.
    recovery_every_updates: int = Field(
        default=1, ge=1, le=1000, exclude_if=lambda value: value == 1
    )
    selection: Literal["last-complete-raw"] = "last-complete-raw"

    @model_validator(mode="before")
    @classmethod
    def read_one_seat_regime(cls, value: object) -> object:
        """Schema 1 omitted the one-seat choice; never relabel frozen evidence."""
        if isinstance(value, dict) and value.get("schema_version") == 1:
            stages = value.get("stages")
            if isinstance(stages, list):
                return {
                    **value,
                    "stages": [
                        {**stage, "learning_seats": stage.get("learning_seats", "one")}
                        if isinstance(stage, dict)
                        and stage.get("operation") == "train_self_play"
                        else stage
                        for stage in stages
                    ],
                }
        return value

    @model_serializer(mode="wrap")
    def retain_one_seat_regime(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        values: dict[str, object] = handler(self)
        if self.schema_version == 1:
            # Old hashes include omitted defaults. Two-seat treatments are always
            # explicit, including when authored from a schema-1 experiment.
            stages = values.get("stages")
            if isinstance(stages, list):
                for stage in stages:
                    if isinstance(stage, dict) and stage.get("learning_seats") == "one":
                        stage.pop("learning_seats")
        return values

    @property
    def recoverable(self) -> bool:
        return self.recovery is not None or self.recovery_max_microsteps is not None

    @model_validator(mode="after")
    def references(self) -> "TrainingRegime":
        tolerant = any(
            isinstance(stage, TrainSelfPlay) and stage.game_errors is not None
            for stage in self.stages
        )
        if tolerant and (
            len(self.stages) != 1 or self.recovery_max_microsteps is not None
        ):
            raise ValueError(
                "game error recovery requires one stage and current-game recovery snapshots"
            )
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
                "compound stages and compound ActorCritic must be selected together"
            )
        if self.agent.compound_decisions and any(
            not isinstance(
                stage, (TrainCompound, ImportPolicy, CollectBelief, TrainBelief)
            )
            for stage in self.stages
        ):
            raise ValueError(
                "compound policies require compound policy stages throughout the run"
            )
        if self.recovery is not None and self.recovery_max_microsteps is not None:
            raise ValueError("select one recovery contract")
        if self.recoverable and (
            any(not isinstance(stage, TrainSelfPlay) for stage in self.stages)
            or (
                self.schedule_clock != "iteration_fraction"
                and any(
                    stage.learning.gradient != "damped_policy" for stage in self.stages
                )
            )
        ):
            raise ValueError(
                "recovery requires self-play with iteration_fraction or absolute damped policy schedules"
            )
        if self.recoverable and any(stage.root is not None for stage in self.stages):
            raise ValueError("diagnostic roots do not support recovery")
        if self.recoverable and any(
            isinstance(stage, TrainSelfPlay) and stage.active_seconds is not None
            for stage in self.stages
        ):
            raise ValueError("active-time process recovery is unsupported")
        if any(stage.execution.device == "cuda" for stage in self.stages):
            if self.recoverable:
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
            if isinstance(stage, CollectBelief):
                policy = previous.get(stage.policy)
                if not isinstance(policy, (TrainSelfPlay, TrainCompound, ImportPolicy)):
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
            if isinstance(stage, TrainBelief) and not isinstance(
                previous.get(stage.dataset), CollectBelief
            ):
                raise ValueError(
                    "belief dataset must refer to an earlier belief collection"
                )
            initial = getattr(stage, "initial", None)
            parent = previous.get(initial)
            if initial and type(parent) is not type(stage):
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
                    or parent.execution.workers != stage.execution.workers
                    or parent.behavior != stage.behavior
                    or parent.streams != stage.streams
                    or parent.learning_seats != stage.learning_seats
                    or parent.games_per_stream != stage.games_per_stream
                    or parent.matchup_curriculum != stage.matchup_curriculum
                    or parent.learning.ema != stage.learning.ema
                    or parent.opponent != stage.opponent
                    or parent.root != stage.root
                    or parent.trainable != stage.trainable
                    or parent.learning.gradient != stage.learning.gradient
                ):
                    raise ValueError(
                        "live self-play continuation must preserve streams, curriculum, gradient, EMA clock, workers and opponent"
                    )
            if isinstance(stage, TrainCompound):
                if initial and parent is not latest_compound:
                    raise ValueError(
                        "compound continuation cannot branch from older weights"
                    )
                latest_compound = stage
            if self.agent.value_kind == "categorical_wdl" and (
                isinstance(stage, TrainCompound)
                or isinstance(stage, TrainSelfPlay)
                and not isinstance(stage.learning, DampedPolicyLearning)
            ):
                raise ValueError("categorical outcome training requires damped_policy")
            previous[stage.id] = stage
            if isinstance(stage, TrainSelfPlay):
                latest_self_play = stage
        return self
