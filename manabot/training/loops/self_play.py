"""Self-play learning with Etude owning run identity, saves and stopping.

`SelfPlay` describes the learning problem; `Counts` is the cumulative work
registered with the run. `step` collects and learns in place. `self_play`
saves canonical weights, Adam, EMA and counts, then publishes immutable
policies for evaluation. Snapshot runs also save root recipes and scheduler RNG.

Fresh streams at starting update u use run seed s + 10000 + u, framework
sampling s + u, and minibatches s + 20000 + u. A host continuation
reinstalls the saved streams and RNGs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import zip_longest
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Literal, Protocol
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

from manabot.env.decision_memory import MEMORY_FIELDS
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
from manabot.policy.protocol import BackendPolicy
from manabot.policy.readings import Readings, legacy_readings
from manabot.training.combat_probe import combat_probe
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
from manabot.training.rollout.collection import (
    Collector,
    CollectorSnapshot,
    OperationScope,
    RolloutBatch,
    close_collector,
    collect,
    no_operation,
    open_collector,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.snapshot_vector import (
    SnapshotVector,
    SnapshotVectorFactory,
)
from manabot.training.rollout.snapshots import (
    CapturePredicate,
    SnapshotBank,
    SnapshotBankState,
    SnapshotConfig,
    SnapshotScorer,
)
from manabot.training.state.backend import (
    Continuation,
    SelfPlayBackend,
    SelfPlayLearner,
    require_continuation,
)
from manabot.training.state.continuation import (
    RestoredContinuation,
    export_continuation,
    restore_continuation,
    restore_host_rng,
)
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
    complete_game_rates,
    metric_row,
    phase_seconds,
    time_since,
)
from manabot.training.updates.backend import ppo_rl
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from manabot.training.updates.selection import UpdateDiagnostics
from manabot.training.updates.targets import BatchTargets, compute_batch_targets
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
    snapshot: SnapshotConfig | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    game_errors: GameErrorPolicy | None = None
    # Recording, diagnostics and statistics; excluded from the learner identity.
    run_settings: RunSettings = Field(
        default_factory=RunSettings, exclude_if=lambda value: value == RunSettings()
    )
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Precision = "float32"
    workers: int = Field(default=1, ge=1)
    threads: int = Field(default=1, ge=1)
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
    checkpoint_weights: Literal["ema", "raw"] = Field(
        default="ema", exclude_if=lambda value: value == "ema"
    )

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
        if self.workers > self.streams:
            raise ValueError("producer count cannot exceed streams")
        if self.snapshot is not None and self.learning_seats != "both":
            raise ValueError("snapshot collection requires both learning seats")
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


def default_backend() -> SelfPlayBackend:
    """Torch for existing self-play entry points called without a backend."""
    from manabot.backends import resolve

    return resolve("torch")


def self_play_collector(
    settings: SelfPlay,
    *,
    seed: int,
    start: int,
    record_attempt: RecordAttempt | None = None,
    operation: OperationScope = no_operation,
    backend: SelfPlayBackend | None = None,
    snapshot: CollectorSnapshot | None = None,
    snapshot_vector: SnapshotVector | SnapshotVectorFactory | None = None,
    snapshot_bank: SnapshotBank | None = None,
    snapshot_state: SnapshotBankState | None = None,
    snapshot_scorer: SnapshotScorer | None = None,
    capture_predicate: CapturePredicate | None = None,
) -> Collector:
    """Open saved native roots or seed new streams from the run seed and update.

    The caller closes it with `close_collector`. `record_attempt` persists
    each settled game when `game_errors` is set.
    """
    backend = backend or default_backend()
    if snapshot_bank is not None:
        if snapshot_bank.config != settings.snapshot:
            raise ValueError("snapshot bank differs from self-play settings")
        if snapshot_scorer is not None and snapshot_scorer is not snapshot_bank.scorer:
            raise ValueError("supplied snapshot bank already defines its scorer")
        snapshot_scorer = snapshot_bank.scorer
    if settings.snapshot is None:
        if snapshot_scorer is not None or capture_predicate is not None:
            raise ValueError(
                "snapshot_scorer and capture_predicate require snapshot settings"
            )
    elif snapshot_scorer is None:
        if settings.snapshot.signal == "action_value_spread":
            raise ValueError(
                "self-play collects no action values for action_value_spread; supply a custom scorer"
            )
        if (
            settings.snapshot.signal == "wdl_entropy"
            and settings.model.value_kind != "categorical_wdl"
        ):
            raise ValueError(
                "wdl_entropy requires a categorical_wdl critic or a custom scorer"
            )
    if snapshot_state is not None:
        if settings.snapshot is None:
            raise ValueError("resuming a snapshot bank requires its snapshot settings")
        snapshot_bank = SnapshotBank.from_state(
            settings.snapshot, snapshot_state, scorer=snapshot_scorer
        )
    elif settings.snapshot is not None and snapshot_bank is None:
        snapshot_bank = SnapshotBank(settings.snapshot, scorer=snapshot_scorer)
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
        snapshot=snapshot,
        snapshot_vector=snapshot_vector,
        snapshot_bank=snapshot_bank,
        capture_predicate=capture_predicate,
        workers=settings.workers,
        numerics=backend.collection.CollectionSession(
            device=settings.device,
            seed=seed + COLLECTION_SEED + start,
            precision=settings.precision,
        ),
    )

    collector.profiler.enabled = True
    return collector


# ---------------------------------------------------------------------------
# Cumulative work, saved by Etude alongside the continuation tensors


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
    snapshot_bank: SnapshotBankState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    def coordinates(self) -> dict[str, int | float]:
        """Cumulative work published in policy files and update metrics."""
        return self.model_dump(exclude={"readings", "model", "snapshot_bank"})


def clocks(counts: Counts) -> dict[str, float]:
    """CLOCKS coordinates other than `updates`, which `run.steps` sets for rows."""
    return {
        "learner_transitions": counts.learner_transitions,
        "training_seconds": counts.training_seconds,
    }


def behavior(
    settings: SelfPlay, learner: SelfPlayLearner, ema: BackendPolicy | None
) -> BackendPolicy:
    """The EMA plays for ema-self; otherwise the raw learner plays."""
    if settings.behavior == "ema-self":
        assert ema is not None
        return ema
    return learner.agent


def pretrained_policy(
    settings: SelfPlay, space: ObservationSpace, backend: SelfPlayBackend
) -> BackendPolicy:
    """The published policy, admitted against these settings.

    The run's model is built with its readings, so a warm start reads its
    inputs exactly as the published policy did, whatever the current rule.
    """
    assert settings.pretrained is not None
    with backend.runtime.preserve_rng(settings.device):
        published, published_space = backend.load_policy(
            settings.pretrained,
            bucketed_shapes=settings.bucketed_shapes,
            live_tokens=settings.live_tokens,
        )
    if published_space.config != space.config:
        raise ValueError("pretrained policy has another observation config")
    if settings.deck_source is not None:
        validate_agent_decks(published, settings.deck_source.players())
    else:
        validate_agent_setup(published, Match(root_match(settings)).to_rust())
    return published


def parent_checkpoint(path: Path, backend: SelfPlayBackend) -> ParentCheckpoint:
    """Identify the published policy a run starts from by its writer and bytes."""
    record = backend.metadata(path)
    return ParentCheckpoint(
        run=(record.get("source") or {}).get("etude_run"),
        update=(record.get("bc") or {}).get("update"),
        sha256=file_sha256(path),
    )


def create(
    settings: SelfPlay, *, seed: int, backend: SelfPlayBackend | None = None
) -> tuple[SelfPlayLearner, BackendPolicy | None]:
    """Fresh weights from `seed` (or the pretrained policy), EMA a copy, fresh Adam."""
    backend = backend or default_backend()
    if isinstance(settings.learning, Learning):
        ppo_rl(backend.rl)
    space = ObservationSpace(settings.observation)
    published = (
        pretrained_policy(settings, space, backend)
        if settings.pretrained is not None
        else None
    )
    backend.seed(seed)
    agent = backend.runtime.create_policy(
        space,
        settings.model,
        pretrained=published,
        precision=settings.precision,
        bucketed_shapes=settings.bucketed_shapes,
        live_tokens=settings.live_tokens,
    ).to(settings.device)
    learner = backend.runtime.create_learner(
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
    if space.config.max_decision_records and saved.keys() == built.keys() - set(
        MEMORY_FIELDS
    ):
        # Older run states omitted the separately prepared memory fields.
        saved = {
            **saved,
            **{key: ["own"] * space.shapes[key][-1] for key in MEMORY_FIELDS},
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
    model: BackendPolicy,
    settings: SelfPlay,
    counts: Counts,
    path: Path,
    weights: Literal["raw", "ema"],
    *,
    seed: int,
    backend: SelfPlayBackend | None = None,
) -> Path:
    backend = backend or default_backend()
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
    with backend.runtime.preserve_rng(settings.device):
        backend.save_policy(
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
        backend.load_policy(temporary)
    os.replace(temporary, path)
    return path


def write_policies(
    settings: SelfPlay,
    run: RunLocation,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    counts: Counts,
    backend: SelfPlayBackend,
) -> Path:
    """Write both immutable milestones and return the one selected for evaluation."""
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
        backend=backend,
    )
    if ema is not None:
        ema_path = write_policy(
            ema,
            settings,
            counts,
            directory / f"{name}-ema.safetensors",
            "ema",
            seed=run.seed,
            backend=backend,
        )
        if settings.checkpoint_weights == "ema":
            path = ema_path
    return path


def publish(
    settings: SelfPlay,
    run: TrainingRun,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    counts: Counts,
    *,
    backend: SelfPlayBackend | None = None,
) -> Path:
    backend = backend or default_backend()
    path = write_policies(settings, run, learner, ema, counts, backend)
    run.checkpoint(path, updates=counts.updates, **clocks(counts))
    return path


# ---------------------------------------------------------------------------
# One update


def learn(
    settings: SelfPlay,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    counts: Counts,
    batch: RolloutBatch,
    rng: np.random.Generator,
    *,
    backend: SelfPlayBackend | None = None,
    targets: BatchTargets | None = None,
) -> UpdateDiagnostics:
    """Learn from one batch in place: the update its learning record selects, then EMA.

    The update made is `counts.updates + 1`. Damped-policy schedules read that
    absolute iteration; PPO schedules read `counts.updates / learning.horizon`.
    Both keep the recipe horizon when a launch extends its target.
    """
    backend = backend or default_backend()
    if isinstance(settings.learning, DampedPolicyLearning):
        metrics = update_damped_iteration(
            backend,
            learner,
            batch,
            settings.learning,
            counts.updates + 1,
            bootstrap_agent=behavior(settings, learner, ema),
            targets=targets,
            learner_transitions=counts.learner_transitions,
        )
    else:
        assert settings.learning.horizon is not None
        metrics = update_iteration(
            backend,
            learner,
            batch,
            settings.learning,
            counts.updates / settings.learning.horizon,
            rng,
            iteration=counts.updates + 1,
            bootstrap_agent=behavior(settings, learner, ema),
            targets=targets,
        )
    if ema is not None and metrics.get("skipped") != "zero sample weight":
        assert settings.learning.ema is not None
        backend.learner.update_ema(ema, learner.agent, settings.learning.ema)
    return metrics


def step(
    settings: SelfPlay,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
    collector: Collector,
    counts: Counts,
    rng: np.random.Generator,
    *,
    out: Path,
    backend: SelfPlayBackend | None = None,
) -> dict[str, object]:
    """Collect one batch, learn from it, and return its metric row; update learner, EMA and counts in place.

    The collector and learner profilers are the one phase measurement: their
    per-update deltas are the row's `time/` keys, and `time/collection` plus
    `time/learning` is the update's active training time. A numerical failure
    leaves a bounded incident under `out` before it propagates.
    """
    backend = backend or default_backend()
    device = settings.device
    stats = collector.stats
    skipped, taps = stats.skipped_decisions, stats.priority_taps
    fresh_rows = stats.fresh_learner_transitions
    snapshot_before = asdict(stats.snapshot) if stats.snapshot is not None else {}
    bank = collector.snapshot_bank
    games, decisions, transitions, idle = (
        stats.games,
        stats.micro_steps,
        stats.learner_transitions,
        stats.idle_stream_steps,
    )
    before = phase_seconds(collector.profiler, learner.profiler)
    batch: RolloutBatch | None = None
    try:
        policy = behavior(settings, learner, ema)
        with bank.begin_batch() if bank is not None else nullcontext() as plan:
            with collector.operation("collection") as collection:
                batch = collect(
                    collector, policy, settings.transitions, snapshot_plan=plan
                )
                collection.progress(stats.micro_steps - decisions)
            backend.runtime.synchronize(device)
            with collector.operation("learning"), learner.profiler.track("Learning"):
                targets = None
                if plan is not None:
                    targets = compute_batch_targets(
                        backend,
                        learner,
                        backend.rollout.AdmittedRollout(learner.agent, batch),
                        settings.learning,
                        policy,
                    )
                    assert plan.receipt is not None
                    prepared = plan.bank.prepare_commit(plan.receipt, targets.signals)
                metrics = learn(
                    settings,
                    learner,
                    ema,
                    counts,
                    batch,
                    rng,
                    backend=backend,
                    targets=targets,
                )
                if settings.model.decision_memory != "none":
                    metrics["decision_memory"] = learner.agent.memory_gates()
                backend.runtime.synchronize(device)
                if plan is not None:
                    plan.bank.publish(prepared)
                    counts.snapshot_bank = plan.bank.state()
    except backend.rl.NumericalError as error:
        state = getattr(backend, "state", None)
        if not isinstance(state, Continuation):
            raise
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
                backend=state,
                framework=state.framework(),
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
    probe: dict[str, float | int] = {}
    every = settings.run_settings.combat_probe_every
    if every and counts.updates % every == 0:
        with collector.operation("combat_probe"), learner.forward_precision():
            started = perf_counter()
            probe = combat_probe(
                learner.agent,
                batch,
                limit=settings.run_settings.combat_probe_rows,
                execution_rows=settings.bootstrap_rows,
            )
            backend.runtime.synchronize(device)
            probe["time/combat_probe"] = perf_counter() - started
    rates: dict[str, float] = {}
    if settings.learning_seats == "both" and settings.batch_closure == "complete-games":
        rates = complete_game_rates(
            games=stats.games - games,
            fresh_rows=stats.fresh_learner_transitions - fresh_rows,
            skips=stats.skipped_decisions - skipped,
            rows=batch.actions.size,
            taps=stats.priority_taps - taps,
            decisions=stats.micro_steps - decisions,
            seconds=seconds,
        )
    if stats.snapshot is not None:
        rates.update(
            {
                f"snapshot/{key}": value - snapshot_before[key]
                for key, value in asdict(stats.snapshot).items()
            }
        )
    return (
        metric_row(metrics)
        | rates
        | {f"coordinates/{k}": v for k, v in counts.coordinates().items()}
        | phases
        | probe
        | resource_metrics(device, backend=backend)
        | {
            "execution/backend": backend.name,
            "execution/device": learner.agent.device_type,
            "execution/precision": settings.precision,
        }
    )


def resource_metrics(
    device: str, *, backend: SelfPlayBackend | None = None
) -> dict[str, float | int]:
    """Process RSS/load and device allocator observations, sampled after each update.

    CUDA values are process high-water marks since process start; MPS values
    are current tensor/driver allocations, not peaks. No sampling thread.
    """
    backend = backend or default_backend()
    values: dict[str, float | int] = {
        "coordinates/rss_bytes": psutil.Process().memory_info().rss,
        "coordinates/host_load_1m": os.getloadavg()[0],
    }
    return values | backend.runtime.memory_metrics(device)


# ---------------------------------------------------------------------------
# Running on a device and recording game failures


@contextmanager
def device_settings(
    settings: SelfPlay, *, backend: SelfPlayBackend | None = None
) -> Iterator[None]:
    """Use the framework device and precision for this loop's lifetime."""
    backend = backend or default_backend()
    with backend.runtime.scoped(settings.threads, settings.precision, settings.device):
        yield


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


class RunLocation(Protocol):
    @property
    def out(self) -> Path: ...
    @property
    def seed(self) -> int: ...


@dataclass(frozen=True)
class FreshRun:
    out: Path
    seed: int


@contextmanager
def self_play_games(
    settings: SelfPlay,
    run: RunLocation,
    counts: Counts,
    *,
    backend: SelfPlayBackend | None = None,
    snapshot: CollectorSnapshot | None = None,
    snapshot_vector: SnapshotVector | SnapshotVectorFactory | None = None,
    snapshot_bank: SnapshotBank | None = None,
    snapshot_scorer: SnapshotScorer | None = None,
    capture_predicate: CapturePredicate | None = None,
) -> Iterator[Collector]:
    """Scope native streams, append-only game ledgers and failure capture."""
    backend = backend or default_backend()
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
            backend=backend,
            snapshot_state=counts.snapshot_bank,
            snapshot_scorer=snapshot_scorer,
            snapshot=snapshot,
            snapshot_vector=snapshot_vector,
            snapshot_bank=snapshot_bank,
            capture_predicate=capture_predicate,
        )
        collector.game_error_state = errors
        try:
            sampling_seed = run.seed + counts.updates
            random.seed(sampling_seed)
            np.random.seed(sampling_seed)
            backend.seed(sampling_seed, deterministic=True)
            yield collector
        finally:
            close_collector(collector)


def resume_training(
    settings: SelfPlay,
    run: TrainingRun,
    backend: SelfPlayBackend,
    learner: SelfPlayLearner,
    ema: BackendPolicy | None,
) -> tuple[Counts, RestoredContinuation]:
    """Register the counts and admit saved tensors before opening the collector."""
    state = require_continuation(backend)
    counts = Counts()
    restored = RestoredContinuation()
    model = settings.model.checkpoint_payload()
    readings = resume_readings(
        counts, learner.agent.readings, learner.agent.observation_space
    )

    def restore(folder: Path) -> None:
        nonlocal restored, readings
        saved_model = counts.model
        if (
            saved_model is None
            and counts.updates > 0
            and (folder / "learner.pt").is_file()
        ):
            initial = run.out / "policies/00000000-raw.safetensors"
            saved_model = backend.metadata(initial)["hypers"]["agent_hypers"]
        resume_model(saved_model, settings.model)
        readings = resume_readings(
            counts, learner.agent.readings, learner.agent.observation_space
        )
        restored = restore_continuation(
            folder,
            state=state,
            learner=learner,
            ema=ema,
            counts=counts,
            settings=settings,
        )

    def unopened_export(folder: Path) -> None:
        raise RuntimeError("cannot export self-play before opening the collector")

    run.resume(export=unopened_export, restore=restore, extra=counts)
    counts.model = model
    counts.readings = readings
    return counts, restored


def self_play(
    settings: SelfPlay,
    run: TrainingRun,
    *,
    updates: int,
    kind: str | None = None,
    recipe: str | None = None,
    backend: SelfPlayBackend | None = None,
    snapshot_vector: SnapshotVector | SnapshotVectorFactory | None = None,
    snapshot_bank: SnapshotBank | None = None,
    snapshot_scorer: SnapshotScorer | None = None,
    capture_predicate: CapturePredicate | None = None,
) -> Path:
    """Train to the target. A supplied bank seeds a fresh run.

    Resume uses the saved roots and scheduler with the supplied runtime scorer.
    The collector releases cached handles before closing its vector.
    """
    backend = backend or default_backend()
    state = require_continuation(backend)
    settings = SelfPlay.model_validate(settings.model_dump())
    if updates < 1:
        raise ValueError("updates must be positive")
    with device_settings(settings, backend=backend):
        learner, ema = create(settings, seed=run.seed, backend=backend)
        counts, restored = resume_training(settings, run, backend, learner, ema)
        host = restored.host
        rng = np.random.default_rng(run.seed + MINIBATCH_SEED + counts.updates)
        with self_play_games(
            settings,
            run,
            counts,
            backend=backend,
            snapshot=host.collector if host is not None else None,
            snapshot_vector=snapshot_vector,
            snapshot_bank=snapshot_bank,
            snapshot_scorer=snapshot_scorer,
            capture_predicate=capture_predicate,
        ) as collector:
            # Validate every replayed root before repairing an unrecorded milestone.
            if counts.updates == 0 or is_checkpoint(settings, counts.updates):
                publish(settings, run, learner, ema, counts, backend=backend)
            if counts.updates == 0:
                log_corpus_setup(
                    run,
                    collector,
                    behavior=settings.behavior,
                    curriculum=settings.matchup_curriculum,
                    parent=parent_checkpoint(settings.pretrained, backend)
                    if settings.pretrained is not None
                    else None,
                    updates=0,
                    **clocks(counts),
                )
            if host is not None:
                restore_host_rng(host, rng)
            if restored.sampling is not None:
                state.restore_sampling(collector.numerics, restored.sampling)

            def export(folder: Path) -> None:
                export_continuation(
                    folder,
                    state=state,
                    learner=learner,
                    ema=ema,
                    counts=counts,
                    settings=settings,
                    collector=collector,
                    rng=rng,
                )

            run.save(export=export)
            first_update = counts.updates + 1
            for update in run.steps("updates", to=updates):
                row = step(
                    settings,
                    learner,
                    ema,
                    collector,
                    counts,
                    rng,
                    out=run.out,
                    backend=backend,
                )
                if update == first_update:
                    if restored.sampling_reseeded:
                        row["run/sampling_reseeded"] = 1
                    if restored.adam_steps_aligned:
                        row["run/adam_steps_aligned"] = restored.adam_steps_aligned
                if kind is not None:
                    row["run/kind"] = kind
                if recipe is not None:
                    row["run/recipe"] = recipe
                run.log(row, **clocks(counts))
                log_corpus_outcomes(run, collector, **clocks(counts))
                run.save()
                if is_checkpoint(settings, update):
                    publish(settings, run, learner, ema, counts, backend=backend)
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
        return publish(settings, run, learner, ema, counts, backend=backend)


@dataclass(frozen=True)
class FreshIteration:
    """Live policies and measurements at the first portable milestone."""

    raw: BackendPolicy
    ema: BackendPolicy | None
    counts: Counts
    metrics: dict[str, object]
    checkpoint: Path


def fresh_iteration(
    settings: SelfPlay, *, backend: SelfPlayBackend, seed: int, out: Path
) -> FreshIteration:
    """Run one ordinary collect/update iteration and export it into a new directory."""
    settings = SelfPlay.model_validate(settings.model_dump())
    out.mkdir(parents=True, exist_ok=False)
    run = FreshRun(out, seed)
    counts = Counts()
    with device_settings(settings, backend=backend):
        learner, ema = create(settings, seed=seed, backend=backend)
        rng = np.random.default_rng(seed + MINIBATCH_SEED)
        with self_play_games(settings, run, counts, backend=backend) as collector:
            row = step(
                settings, learner, ema, collector, counts, rng, out=out, backend=backend
            )
        checkpoint = write_policies(settings, run, learner, ema, counts, backend)
    return FreshIteration(learner.agent, ema, counts, row, checkpoint)
