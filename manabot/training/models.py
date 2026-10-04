"""Validated training recipes and durable execution records."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.infra.hypers import AgentHypers, MatchHypers, ObservationSpaceHypers


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
    learning: Learning = Learning()


Operation = Annotated[
    CollectSearch | TrainSupervised | TrainSelfPlay, Field(discriminator="operation")
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
    schedule_clock: Literal["run_elapsed_budget"] = "run_elapsed_budget"
    selection: Literal["last-complete-raw"] = "last-complete-raw"

    @model_validator(mode="after")
    def references(self):
        previous: dict[str, Stage] = {}
        latest_self_play = None
        for stage in self.stages:
            if stage.id in previous:
                raise ValueError("stage IDs must be unique")
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
                ):
                    raise ValueError(
                        "live self-play continuation must preserve streams and EMA clock"
                    )
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
    selected_artifact: dict | None = None
    error: str | None = None
