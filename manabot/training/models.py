"""Validated training recipes and durable execution records."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.infra.hypers import AgentHypers, MatchHypers, ObservationSpaceHypers


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
    device: Literal["cpu"] = "cpu"
    precision: Literal["float32"] = "float32"
    workers: Literal[1] = 1
    threads: int = Field(default=1, ge=1, le=4)
    wall_seconds: float = Field(default=300, gt=0)
    memory_bytes: int = Field(default=32 * 1024**3, gt=0)


class Stage(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    execution: Execution = Execution()


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


class TrainSupervised(Stage):
    operation: Literal["train_supervised"]
    trainer: Literal["search_supervised"] = "search_supervised"
    optimizer: Literal["adam"] = "adam"
    trainable: Literal["policy"] = "policy"
    target: Literal["visit_distribution"] = "visit_distribution"
    datasets: list[str] = Field(min_length=1)
    initial: str | None = None
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)


class TrainSelfPlay(Stage):
    operation: Literal["train_self_play"]
    trainer: Literal["net_opponent"] = "net_opponent"
    optimizer: Literal["adam"] = "adam"
    trainable: Literal["policy_value"] = "policy_value"
    behavior: Literal["current-self"] = "current-self"
    initial: str | None = None
    updates: int = Field(default=2, ge=1)
    streams: int = Field(default=4, ge=2)
    transitions: int = Field(default=256, ge=1)
    learning: Learning | AtaraxosMoveLearning = Learning()


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


Operation = Annotated[
    CollectSearch | TrainSupervised | TrainSelfPlay | CollectBelief | TrainBelief,
    Field(discriminator="operation"),
]


class TrainingRegime(Strict):
    schema_version: Literal[1] = 1
    id: str
    world: str
    match: MatchHypers
    observation: ObservationSpaceHypers = ObservationSpaceHypers()
    agent: AgentHypers = AgentHypers()
    stages: list[Operation] = Field(min_length=1)
    wall_seconds: float = Field(default=900, gt=0)
    schedule_clock: Literal["run_elapsed_budget", "iteration_fraction"] = (
        "run_elapsed_budget"
    )
    recovery_max_microsteps: int | None = Field(default=None, ge=1, le=1_000_000)
    selection: Literal["last-complete-raw"] = "last-complete-raw"

    @model_validator(mode="after")
    def references(self) -> "TrainingRegime":
        if self.recovery_max_microsteps is not None and (
            len(self.stages) != 1
            or not isinstance(self.stages[0], TrainSelfPlay)
            or self.schedule_clock != "iteration_fraction"
        ):
            raise ValueError(
                "recovery requires one self-play stage and iteration_fraction schedule"
            )
        previous: dict[str, Stage] = {}
        latest_self_play = None
        for stage in self.stages:
            if stage.id in previous:
                raise ValueError("stage IDs must be unique")
            if isinstance(stage, CollectBelief):
                policy = previous.get(stage.policy)
                if not isinstance(policy, (TrainSupervised, TrainSelfPlay)):
                    raise ValueError(
                        "belief policy must refer to an earlier policy stage"
                    )
                if stage.weights == "ema" and (
                    not isinstance(policy, TrainSelfPlay) or policy.learning.ema is None
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
            if isinstance(stage, TrainSupervised):
                if len(stage.datasets) != len(set(stage.datasets)):
                    raise ValueError("dataset references must be unique")
                for ref in stage.datasets:
                    if not isinstance(previous.get(ref), CollectSearch):
                        raise ValueError(
                            f"dataset {ref} must refer to an earlier collection"
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
                    parent.streams != stage.streams
                    or parent.learning.ema != stage.learning.ema
                    or parent.learning.gradient != stage.learning.gradient
                ):
                    raise ValueError(
                        "live self-play continuation must preserve streams, gradient and EMA clock"
                    )
            if self.agent.value_kind == "categorical_wdl" and (
                isinstance(stage, TrainSupervised)
                or isinstance(stage, TrainSelfPlay)
                and not isinstance(stage.learning, AtaraxosMoveLearning)
            ):
                raise ValueError("categorical outcome training requires ataraxos_move")
            previous[stage.id] = stage
            if isinstance(stage, TrainSelfPlay):
                latest_self_play = stage
        return self


class StageRecord(Strict):
    id: str
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    seconds: float = 0
    cumulative_seconds: float | None = Field(default=None, ge=0)
    collection_seconds: float = 0
    learning_seconds: float = 0
    export_seconds: float = 0
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


class TrainingRun(Strict):
    schema_version: Literal[1] = 1
    id: str
    regime_digest: str
    regime: TrainingRegime
    seed: int
    seed_streams: dict[str, int]
    identities: dict
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = (
        "pending"
    )
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
