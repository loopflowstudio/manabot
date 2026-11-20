"""Self-play learning with Etude owning run identity, saves and stopping.

`SelfPlay` describes the learning problem; `Counts` is the cumulative work
registered with the run. `step` collects and learns in place. `self_play`
registers raw weights, Adam, EMA and counts, then publishes immutable
policies for evaluation. No native game or RNG is persisted.

Model initialization uses the run seed s. At starting update u, native streams
use s + 10000 + u, sampling s + u, and minibatches s + 20000 + u. A fresh
run preserves the retired executor's offsets; resume starts fresh streams.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from itertools import zip_longest
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from etude.records import Clock
from etude.train.records import Time
from etude.train.tracking import TrainingRun
import numpy as np
import psutil
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch.checkpoint import (
    load_checkpoint,
    load_metadata,
    save_checkpoint as save_policy,
)
from manabot.backends.torch.learner import Learner, apply_matmul_precision, update_ema
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.random import seed_random
from manabot.backends.torch.rl import NumericalError
from manabot.env.failures import CaptureScope, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.run_settings import RunSettings
from manabot.model.world import (
    input_columns,
    validate_agent_decks,
    validate_agent_setup,
)
from manabot.policy.checkpoint import pretrained_memory_branch
from manabot.policy.readings import Readings, legacy_readings
from manabot.training.corpus import (
    ParentCheckpoint,
    log_corpus_outcomes,
    log_corpus_setup,
)
from manabot.training.decks import UniformDecks
from manabot.training.learning import (
    BatchClosure,
    DampedPolicyLearning,
    Learning,
    Precision,
)
from manabot.training.mps import mps_torch_error
from manabot.training.rollout.collection import (
    Collector,
    OperationScope,
    RolloutBatch,
    close_collector,
    collect,
    no_operation,
    open_collector,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    RecordAttempt,
    append_attempt,
    open_ledger,
)
from manabot.training.state.incidents import save_incident
from manabot.training.update_metrics import (
    metric_row,
    phase_seconds,
    time_since,
)
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from manabot.training.updates.selection import UpdateDiagnostics
import managym

# Offsets that separate a run seed's random streams (see the module docstring).
COLLECTION_SEED = 10_000
MINIBATCH_SEED = 20_000
CLOCKS = [
    "updates",
    Clock(
        name="learner_transitions", kind="step", meaning="collected learner transitions"
    ),
    Time("training_seconds", "active collection and learning seconds"),
]


# ---------------------------------------------------------------------------
# Settings


class SelfPlay(BaseModel):
    """What one self-play loop plays, how it learns and where it runs.

    Validation checks the configured match or deck source, observation and
    model against the installed runtime and the collector and learner.
    """

    model_config = ConfigDict(extra="forbid")

    # Settings recorded before the world label was removed carry it; ignored.
    world: str | None = Field(default=None, exclude_if=lambda value: value is None)
    match: MatchConfig | None = None
    deck_source: UniformDecks | None = None
    observation: ObservationConfig = ObservationConfig()
    model: ModelConfig = ModelConfig()
    learning: Learning | DampedPolicyLearning
    behavior: Literal["current-self", "ema-self", "random"] = "current-self"
    streams: int = Field(default=4, ge=2)
    transitions: int = Field(default=256, ge=1)
    games_per_stream: int | None = Field(default=None, ge=1)
    learning_seats: Literal["one", "both"] = "both"
    batch_closure: BatchClosure = "complete-games"
    matchup_curriculum: Literal[
        "fixed", "cross-balanced", "mirrors-balanced", "elemental-uniform"
    ] = "fixed"
    game_errors: GameErrorPolicy | None = None
    # Recording, diagnostics and statistics; excluded from the learner identity.
    run_settings: RunSettings = Field(
        default_factory=RunSettings, exclude_if=lambda value: value == RunSettings()
    )
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Precision = "float32"
    workers: Literal[1, 2, 4] = 1
    threads: int = Field(default=1, ge=1, le=4)
    bucketed_shapes: bool = False
    live_tokens: bool = False
    # Rows per forward in collection, bootstrap and learning. They bound how
    # many decision-memory prefixes are materialized at once, independently of
    # streams and of the optimizer's logical minibatch; None runs groups whole.
    collection_rows: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    bootstrap_rows: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    learner_rows: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    # A published policy to start from; only its weights are inherited. Its
    # model must equal this one, except that a history-enabled control may
    # seed a decision-memory encoder (`pretrained_memory_branch`).
    pretrained: Path | None = None
    # Stop after the update that reaches this many learner decisions; `updates`
    # stays a safety ceiling. Requires the damped decision schedule.
    learner_transitions: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    # Checkpoint at absolute updates 0, n, 2n, …; a continued run keeps the grid.
    checkpoint_every: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_settings(self) -> "SelfPlay":
        if isinstance(self.learning, Learning) and self.learning.horizon is None:
            raise ValueError("PPO self-play requires a learning horizon")
        if (self.match is None) == (self.deck_source is None):
            raise ValueError("self-play needs exactly one match or deck source")
        if self.model.value_kind == "categorical_wdl" and not isinstance(
            self.learning, DampedPolicyLearning
        ):
            raise ValueError("categorical outcome training requires damped_policy")
        if self.model.belief_count_buckets:
            raise ValueError("self-play does not produce belief inputs")
        if self.precision == "tf32" and self.device != "cuda":
            raise ValueError("precision tf32 requires CUDA")
        if self.precision == "bf16" and self.device == "cpu":
            raise ValueError("precision bf16 requires CUDA or MPS")
        if self.live_tokens and self.device != "cuda":
            raise ValueError("live tokens require CUDA")
        if self.device == "mps" and (error := mps_torch_error()) is not None:
            raise ValueError(error)
        if self.workers > self.streams:
            raise ValueError("producer count cannot exceed streams")
        if self.learning_seats == "both" and (
            self.behavior == "random" or self.batch_closure == "exact-quota"
        ):
            raise ValueError("both learning seats require complete-game self-play")
        if self.batch_closure == "exact-quota":
            if self.games_per_stream is not None:
                raise ValueError("games_per_stream requires complete games")
            if self.game_errors is not None and not self.game_errors.recover_capacity:
                raise ValueError("game error recovery requires complete games")
        if self.behavior == "ema-self" and self.learning.ema is None:
            raise ValueError("ema-self behavior requires an EMA rate")
        if self.matchup_curriculum == "elemental-uniform":
            if self.behavior != "current-self":
                raise ValueError("elemental-uniform requires current-self")
            elemental_roster(Match(root_match(self)))
            if self.model.world != "elemental-world-v2":
                raise ValueError(
                    "elemental-uniform requires complete Elemental policy input"
                )
        elif self.matchup_curriculum != "fixed":
            width = 12 if self.matchup_curriculum == "mirrors-balanced" else 4
            if self.streams % width or self.behavior != "current-self":
                raise ValueError(
                    "balanced matchups require current-self and complete stream strata"
                )
        if self.deck_source is not None:
            if self.behavior != "current-self" or self.matchup_curriculum != "fixed":
                raise ValueError(
                    "a deck source requires current-self and no curriculum"
                )
            self.deck_source.check()
            if self.model.world != managym.WORLD_SETUP:
                raise ValueError(
                    "a deck source requires the complete world policy vocabulary"
                )
        if self.model.recent_events and (
            self.model.world is None or not 1 <= self.observation.max_events <= 32
        ):
            raise ValueError("recent events require a world and 1..32 event rows")
        memory = (
            self.observation.max_decision_records,
            self.observation.max_decision_reference_details,
            self.observation.max_decision_references,
            self.observation.max_decision_members,
        )
        if self.model.decision_memory != "none" and not all(memory):
            raise ValueError("decision memory requires all four decision capacities")
        if memory[0] and None in (
            self.collection_rows,
            self.bootstrap_rows,
            self.learner_rows,
        ):
            raise ValueError(
                "decision-memory collection requires explicit collection_rows, "
                "bootstrap_rows and learner_rows"
            )
        if self.learner_transitions is not None and (
            not isinstance(self.learning, DampedPolicyLearning)
            or self.learning.decision_schedule is None
        ):
            raise ValueError(
                "a learner-decision target requires its damped decision schedule"
            )
        self.observation = self.observation.model_copy(
            update={"policy_history_version": 1 if self.model.recent_events else 0}
        )
        return self


def root_match(settings: SelfPlay) -> MatchConfig:
    """The authored match, or a deck source's representative opening pair."""
    if settings.deck_source is not None:
        return settings.deck_source.match_config()
    if settings.match is None:
        raise ValueError("self-play needs a match or a deck source")
    return settings.match


def learner_digest(settings: SelfPlay) -> str:
    """Identity of the learner `settings` train; machine fields excluded."""
    return canonical_sha256(
        settings.model_dump(
            mode="json",
            exclude={
                "device",
                "workers",
                "threads",
                "run_settings",
                "collection_rows",
                "bootstrap_rows",
                "learner_rows",
            },
        )
    )


def is_checkpoint(settings: SelfPlay, update: int) -> bool:
    return (
        settings.checkpoint_every is not None
        and update % settings.checkpoint_every == 0
    )


def self_play_collector(
    settings: SelfPlay,
    *,
    seed: int,
    start: int,
    record_attempt: RecordAttempt | None = None,
    operation: OperationScope = no_operation,
) -> Collector:
    """Native streams seeded from the run seed and the update they start at.

    The caller closes it with `close_collector`. `record_attempt` persists
    each settled game when `game_errors` is set.
    """
    collector = open_collector(
        ObservationSpace(settings.observation),
        Match(root_match(settings)),
        Reward(RewardConfig()),
        num_envs=settings.streams,
        execution_rows=settings.collection_rows,
        seed=seed + COLLECTION_SEED + start,
        opponent_mode="random" if settings.behavior == "random" else "self",
        learning_seats=settings.learning_seats,
        batch_closure=settings.batch_closure,
        games_per_stream=settings.games_per_stream,
        game_errors=settings.game_errors,
        record_attempt=record_attempt,
        operation=operation,
        matchup_curriculum=settings.matchup_curriculum,
        deck_source=settings.deck_source,
        workers=settings.workers,
        numerics=torch_backend.collection.CollectionSession(
            device=settings.device,
            seed=seed + COLLECTION_SEED + start,
            precision=settings.precision,
        ),
    )

    collector.profiler.enabled = True
    return collector


# ---------------------------------------------------------------------------
# Cumulative work, saved by Etude alongside the framework objects


class Counts(BaseModel):
    """Cumulative completed work, changed in place so Etude saves this object.

    Model settings and input readings identify the saved weights on resume.
    """

    updates: int = 0
    training_seconds: float = 0.0
    games: int = 0
    environment_decisions: int = 0
    learner_transitions: int = 0
    idle_stream_steps: int = 0
    optimizer_updates: int = 0
    optimizer_exposures: int = 0
    readings: dict[str, list[str]] | None = None
    model: dict[str, Any] | None = None

    def coordinates(self) -> dict[str, int | float]:
        """Cumulative work published in policy files and update metrics."""
        return self.model_dump(exclude={"readings", "model"})


def clocks(counts: Counts) -> dict[str, float]:
    """CLOCKS coordinates other than `updates`, which `run.steps` sets for rows."""
    return {
        "learner_transitions": counts.learner_transitions,
        "training_seconds": counts.training_seconds,
    }


def behavior(
    settings: SelfPlay, learner: Learner, ema: ActorCritic | None
) -> ActorCritic:
    """The EMA plays for ema-self; otherwise the raw learner plays."""
    if settings.behavior == "ema-self":
        assert ema is not None
        return ema
    return learner.agent


def pretrained_policy(settings: SelfPlay, space: ObservationSpace) -> ActorCritic:
    """The published policy, admitted against these settings.

    The run's model is built with its readings, so a warm start reads its
    inputs exactly as the published policy did, whatever the current rule.
    """
    assert settings.pretrained is not None
    with torch.random.fork_rng(devices=[]):
        published, published_space = load_checkpoint(str(settings.pretrained))
    if published_space.encoder.config != space.encoder.config:
        raise ValueError("pretrained policy has another observation config")
    if settings.deck_source is not None:
        validate_agent_decks(published, settings.deck_source.players())
    else:
        validate_agent_setup(published, Match(root_match(settings)).to_rust())
    if any(not torch.isfinite(v).all() for v in published.state_dict().values()):
        raise ValueError("pretrained policy contains nonfinite tensors")
    return published


def load_pretrained(
    agent: ActorCritic, weights: dict[str, torch.Tensor], source: ModelConfig
) -> None:
    """Copy published weights into `agent` in place; strict unless a memory branch.

    A memory encoder started from a same-layout control keeps its newly
    initialized encoder, gates and projections. Every other tensor must come
    from the published policy, with its exact shape.
    """
    if not pretrained_memory_branch(
        source, agent.config, agent.observation_space.encoder.config
    ):
        agent.load_state_dict(weights, strict=True)
        return
    fresh = agent.state_dict()
    memory = {
        name
        for name in fresh
        if name.startswith("decision_memory.")
        or name.endswith((".memory_gate", ".memory_projection.weight"))
    }
    if set(fresh) - set(weights) != memory or set(weights) - set(fresh):
        raise ValueError(
            "pretrained copy would leave non-memory parameters unused or uninitialized"
        )
    agent.load_state_dict({**fresh, **weights}, strict=True)


def parent_checkpoint(path: Path) -> ParentCheckpoint:
    """Identify the published policy a run starts from by its writer and bytes."""
    record = load_metadata(path)
    return ParentCheckpoint(
        run=(record.get("source") or {}).get("etude_run"),
        update=(record.get("bc") or {}).get("update"),
        sha256=file_sha256(path),
    )


def create(settings: SelfPlay, *, seed: int) -> tuple[Learner, ActorCritic | None]:
    """Fresh weights from `seed` (or the pretrained policy), EMA a copy, fresh Adam."""
    space = ObservationSpace(settings.observation)
    published = (
        pretrained_policy(settings, space) if settings.pretrained is not None else None
    )
    torch.manual_seed(seed)
    agent = ActorCritic(
        space,
        settings.model,
        readings=None if published is None else published.readings,
        bucketed_shapes=settings.bucketed_shapes,
        live_tokens=settings.live_tokens,
    ).to(settings.device)
    if published is not None:
        load_pretrained(agent, published.state_dict(), published.config)
    learner = Learner(
        agent,
        settings.device,
        minibatch_rows=settings.streams,
        precision=settings.precision,
        execution_rows=settings.learner_rows,
        bootstrap_rows=settings.bootstrap_rows,
    )
    ema = deepcopy(agent) if settings.learning.ema is not None else None
    return learner, ema


def resume_model(saved: dict[str, Any] | None, model: ModelConfig) -> dict[str, Any]:
    """Return the model settings to save, refusing a change to the learner."""
    if saved is not None:
        previous = ModelConfig.from_checkpoint(saved)
        changed = [
            f"{key}: {getattr(previous, key)!r} -> {getattr(model, key)!r}"
            for key in ModelConfig.model_fields
            if getattr(previous, key) != getattr(model, key)
        ]
        if changed:
            raise ValueError(
                f"resumed model settings differ from the saved run ({', '.join(changed)}); "
                "resume with the run's model settings or from the revision "
                "that started the run"
            )
    return model.checkpoint_payload()


def resume_readings(
    counts: Counts, readings: Readings, space: ObservationSpace
) -> dict[str, list[str]]:
    """Return the model's readings for run state, refusing incompatible saved weights."""
    built = {key: [str(value) for value in values] for key, values in readings.items()}
    saved = counts.readings
    if saved is None:
        if counts.updates == 0:
            return built
        # Runs predating the record used legacy readings over the model's prefix.
        legacy = legacy_readings(space)
        saved = {
            key: [str(value) for value in legacy[key][: len(values)]]
            for key, values in readings.items()
        }
    if saved == built:
        return built

    columns = input_columns(space)

    def name(key: str, index: int) -> str:
        names = columns.get(key, ())
        return f"{key}.{names[index]}" if index < len(names) else f"{key}[{index}]"

    differ = sorted(
        f"{name(key, index)}: {was} -> {now}"
        for key in saved.keys() | built.keys()
        for index, (was, now) in enumerate(
            zip_longest(saved.get(key, ()), built.get(key, ()))
        )
        if was != now
    )
    shown = 8
    listed = ", ".join(differ[:shown]) + (
        f", and {len(differ) - shown} more" if len(differ) > shown else ""
    )
    written = (
        "the saved run"
        if counts.readings is not None
        else "the saved run, which was written before input readings were recorded"
    )
    raise ValueError(
        f"resumed model reads its inputs differently from {written} "
        f"({listed}); resume from the revision that started the "
        "run, or warm start from its last published policy"
    )


def write_policy(
    model: ActorCritic,
    settings: SelfPlay,
    counts: Counts,
    path: Path,
    weights: Literal["raw", "ema"],
    *,
    seed: int,
) -> Path:
    if path.exists():
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    players = (
        None
        if settings.deck_source is not None
        else Match(root_match(settings)).to_rust()
    )
    averaging = {
        "clock": "collect-update-iteration",
        "iteration": counts.updates,
        "rate": settings.learning.ema,
    }
    # A fresh model constructed for admission must not draw from the loop's RNG.
    device = next(model.parameters()).device.type
    with torch.random.fork_rng(
        devices=[0] if device in {"cuda", "mps"} else [],
        device_type="mps" if device == "mps" else "cuda",
    ):
        save_policy(
            model,
            model.observation_space,
            temporary,
            player_configs=players,
            extra={
                "weights": weights,
                "update": counts.updates,
                "counts": counts.coordinates(),
                "seed": seed,
                "value_semantic": "signed_outcome",
                "averaging": averaging if weights == "ema" else None,
            },
        )
        load_checkpoint(str(temporary))
    os.replace(temporary, path)
    return path


def publish(
    settings: SelfPlay,
    run: TrainingRun,
    learner: Learner,
    ema: ActorCritic | None,
    counts: Counts,
) -> Path:
    """Keep immutable raw/EMA policies and record the policy used for evaluation."""
    directory = run.out / "policies"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{counts.updates:08d}"
    path = write_policy(
        learner.agent,
        settings,
        counts,
        directory / f"{name}-raw.safetensors",
        "raw",
        seed=run.seed,
    )
    if ema is not None:
        path = write_policy(
            ema,
            settings,
            counts,
            directory / f"{name}-ema.safetensors",
            "ema",
            seed=run.seed,
        )
    run.checkpoint(path, updates=counts.updates, **clocks(counts))
    return path


# ---------------------------------------------------------------------------
# One update


def learn(
    settings: SelfPlay,
    learner: Learner,
    ema: ActorCritic | None,
    counts: Counts,
    batch: RolloutBatch,
    rng: np.random.Generator,
) -> UpdateDiagnostics:
    """Learn from one batch in place: the update its learning record selects, then EMA.

    The update made is `counts.updates + 1`. Damped-policy schedules read that
    absolute iteration; PPO schedules read `counts.updates / learning.horizon`.
    Both keep the recipe horizon when a launch extends its target.
    """
    if isinstance(settings.learning, DampedPolicyLearning):
        metrics = update_damped_iteration(
            torch_backend,
            learner,
            batch,
            settings.learning,
            counts.updates + 1,
            bootstrap_agent=behavior(settings, learner, ema),
            learner_transitions=counts.learner_transitions,
        )
    else:
        assert settings.learning.horizon is not None
        metrics = update_iteration(
            torch_backend,
            learner,
            batch,
            settings.learning,
            counts.updates / settings.learning.horizon,
            rng,
            iteration=counts.updates + 1,
            bootstrap_agent=behavior(settings, learner, ema),
        )
    if ema is not None:
        assert settings.learning.ema is not None
        update_ema(ema, learner.agent, settings.learning.ema)
    return metrics


def step(
    settings: SelfPlay,
    learner: Learner,
    ema: ActorCritic | None,
    collector: Collector,
    counts: Counts,
    rng: np.random.Generator,
    *,
    out: Path,
) -> dict[str, object]:
    """Collect one batch, learn from it, and return its metric row; update learner, EMA and counts in place.

    The collector and learner profilers are the one phase measurement: their
    per-update deltas are the row's `time/` keys, and `time/collection` plus
    `time/learning` is the update's active training time. A numerical failure
    leaves a bounded incident under `out` before it propagates.
    """
    device = settings.device
    stats = collector.stats
    skipped, taps = stats.skipped_decisions, stats.priority_taps
    games, decisions, transitions, idle = (
        stats.games,
        stats.micro_steps,
        stats.learner_transitions,
        stats.idle_stream_steps,
    )
    before = phase_seconds(collector.profiler, learner.profiler)
    batch: RolloutBatch | None = None
    try:
        with collector.operation("collection") as collection:
            batch = collect(
                collector, behavior(settings, learner, ema), settings.transitions
            )
            collection.progress(stats.micro_steps - decisions)
        synchronize(device)
        with collector.operation("learning"), learner.profiler.track("Learning"):
            metrics = learn(settings, learner, ema, counts, batch, rng)
            if settings.model.decision_memory != "none":
                metrics["decision_memory"] = learner.agent.memory_gates()
            synchronize(device)
    except NumericalError as error:
        try:
            save_incident(
                out / "numerical-incident" / uuid4().hex,
                error,
                learner,
                ema,
                counts,
                collector,
                batch,
                rng,
                out=out,
                settings_digest=learner_digest(settings),
                backend=torch_backend.state,
                framework={"torch": str(torch.__version__), "cuda": torch.version.cuda},
            )
        except Exception as capture_error:
            error.add_note(f"incident capture failed: {capture_error}")
        raise
    phases = time_since(before, phase_seconds(collector.profiler, learner.profiler))
    # The collector's totals are this attempt's; counts span every attempt.
    counts.updates += 1
    seconds = phases["time/collection"] + phases["time/learning"]
    counts.training_seconds += seconds
    counts.games += stats.games - games
    counts.environment_decisions += stats.micro_steps - decisions
    counts.learner_transitions += stats.learner_transitions - transitions
    counts.idle_stream_steps += stats.idle_stream_steps - idle
    counts.optimizer_updates += metrics.get("numerical", {}).get("optimizer_steps", 0)
    counts.optimizer_exposures += metrics.get("optimizer_exposures", 0)
    rates: dict[str, float] = {}
    # With both seats learning complete games, retained rows count exactly
    # the surfaced choices of the games that contributed terminal targets.
    if settings.learning_seats == "both" and settings.batch_closure == "complete-games":
        completed = stats.games - games
        rows = batch.actions.size
        skips = stats.skipped_decisions - skipped
        rates = {
            "decisions_per_game": rows / completed,
            "skipped_share": skips / (skips + rows),
            "priority_tap_rate": (stats.priority_taps - taps)
            / (stats.micro_steps - decisions),
            "rows_per_second": rows / seconds,
            "updates_per_hour": 3600 / seconds,
            "games_per_hour": completed * 3600 / seconds,
        }
    return (
        metric_row(metrics)
        | rates
        | {f"coordinates/{k}": v for k, v in counts.coordinates().items()}
        | phases
        | resource_metrics(device)
    )


def resource_metrics(device: str) -> dict[str, float | int]:
    """Process RSS/load and device allocator observations, sampled after each update.

    CUDA values are process high-water marks since process start; MPS values
    are current tensor/driver allocations, not peaks. No sampling thread.
    """
    values: dict[str, float | int] = {
        "coordinates/rss_bytes": psutil.Process().memory_info().rss,
        "coordinates/host_load_1m": os.getloadavg()[0],
    }
    if device == "cuda":
        values |= {
            "memory/peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "memory/peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
        }
    elif device == "mps":
        values |= {
            "memory/allocated_mps_bytes": torch.mps.current_allocated_memory(),
            "memory/driver_allocated_mps_bytes": torch.mps.driver_allocated_memory(),
        }
    return values


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Running on a device and recording game failures


@contextmanager
def device_settings(settings: SelfPlay) -> Iterator[None]:
    """Threads and matmul precision for the loop, restored afterwards."""
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; CPU fallback is forbidden")
    if settings.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable; CPU fallback is forbidden")
    threads = torch.get_num_threads()
    torch.set_num_threads(settings.threads)
    apply_matmul_precision(settings.precision)
    try:
        yield
    finally:
        apply_matmul_precision("float32")
        torch.set_num_threads(threads)


def game_ledger(
    out: Path, policy: GameErrorPolicy
) -> tuple[GameErrorLedger, GameErrorState]:
    """This attempt's append-only ledger and the state earlier attempts settled.

    Raises `GameErrorLimit` when an earlier attempt already crossed the limit.
    """
    directory = out / "game-attempts"
    directory.mkdir(parents=True, exist_ok=True)
    earlier = sorted(directory.glob("*.jsonl"))
    ledger, errors = open_ledger(
        directory / f"{len(earlier):04d}.jsonl",
        policy,
        parents=[str(path) for path in earlier],
    )
    if errors.stop_reason:
        raise GameErrorLimit(errors.stop_reason)
    return ledger, errors


# ---------------------------------------------------------------------------
# The loop


@contextmanager
def self_play_games(
    settings: SelfPlay, run: TrainingRun, counts: Counts
) -> Iterator[Collector]:
    """Scope fresh native streams, append-only game ledgers and failure capture."""
    failures = CaptureScope(
        run.out / "failures",
        lambda: {"seed": run.seed, "update": counts.updates, "out": str(run.out)},
        settings=settings.run_settings,
    )
    record_attempt: RecordAttempt | None = None
    errors = GameErrorState()
    if settings.game_errors is not None:
        ledger, errors = game_ledger(run.out, settings.game_errors)

        def record_attempt(
            errors: GameErrorState, attempt: GameAttempt
        ) -> GameErrorState:
            return append_attempt(ledger, errors, attempt, failures.context())

    with capture_scope(failures):
        collector = self_play_collector(
            settings,
            seed=run.seed,
            start=counts.updates,
            record_attempt=record_attempt,
            operation=failures.operation,
        )
        collector.game_error_state = errors
        seed_random(run.seed + counts.updates)
        try:
            yield collector
        finally:
            close_collector(collector)


def self_play(
    settings: SelfPlay,
    run: TrainingRun,
    *,
    updates: int,
    kind: str | None = None,
    recipe: str | None = None,
) -> Path:
    """Train to the target; log report labels only when the recipe supplies them."""
    settings = SelfPlay.model_validate(settings.model_dump())
    if updates < 1:
        raise ValueError("updates must be positive")
    with device_settings(settings):
        learner, ema = create(settings, seed=run.seed)
        # No EMA object exists for recipes that do not average their policy.
        objects: dict[str, object] = {
            "learner": learner.agent,
            "optimizer": learner.optimizer,
        }
        if ema is not None:
            objects["ema"] = ema
        # Check the saved architecture and input contract before restoring tensors.
        saved_counts = run.resume(extra=Counts())
        saved_model = saved_counts.model
        if saved_model is None and saved_counts.updates > 0:
            initial = run.out / "policies/00000000-raw.safetensors"
            saved_model = load_metadata(initial)["hypers"]["agent_hypers"]
        model = resume_model(saved_model, settings.model)
        readings = resume_readings(
            saved_counts,
            learner.agent.readings,
            learner.agent.observation_space,
        )
        counts = run.resume(**objects, extra=Counts())
        counts.model = model
        counts.readings = readings
        rng = np.random.default_rng(run.seed + MINIBATCH_SEED + counts.updates)
        # Repair an unrecorded milestone without rewriting its immutable bytes.
        if counts.updates == 0 or is_checkpoint(settings, counts.updates):
            publish(settings, run, learner, ema, counts)
        with self_play_games(settings, run, counts) as collector:
            if counts.updates == 0:
                log_corpus_setup(
                    run,
                    collector,
                    behavior=settings.behavior,
                    curriculum=settings.matchup_curriculum,
                    parent=parent_checkpoint(settings.pretrained)
                    if settings.pretrained is not None
                    else None,
                    updates=0,
                    **clocks(counts),
                )
            for update in run.steps("updates", to=updates):
                row = step(settings, learner, ema, collector, counts, rng, out=run.out)
                if kind is not None:
                    row["run/kind"] = kind
                if recipe is not None:
                    row["run/recipe"] = recipe
                run.log(row, **clocks(counts))
                log_corpus_outcomes(run, collector, **clocks(counts))
                run.save()
                if is_checkpoint(settings, update):
                    publish(settings, run, learner, ema, counts)
                if (
                    settings.learner_transitions is not None
                    and counts.learner_transitions >= settings.learner_transitions
                ):
                    break
            else:
                if settings.learner_transitions is not None:
                    raise RuntimeError(
                        "update safety ceiling reached before learner-decision target"
                    )
        return publish(settings, run, learner, ema, counts)
