"""Self-play learning with Etude owning run identity, saves and stopping.

`SelfPlay` describes the learning problem; `Counts` is the cumulative work
registered with the run. `step` collects and learns in place. `self_play`
registers raw weights, Adam, EMA and counts, then publishes immutable,
world-bound policies for evaluation. No native game or RNG is persisted.

Model initialization uses the run seed s. At starting update u, native streams
use s + 10000 + u, sampling s + u, and minibatches s + 20000 + u. A fresh
run preserves the retired executor's offsets; resume starts fresh streams.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from etude.train.records import Time
from etude.train.tracking import Run
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
from manabot.infra.identity import canonical_sha256
from manabot.model.world import (
    validate_agent_decks,
    validate_agent_setup,
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
    RolloutBatch,
    close_collector,
    collect,
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
    OUTCOME_TABLE,
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
    Time("training_seconds", "active collection and learning seconds"),
]


# ---------------------------------------------------------------------------
# Settings


class SelfPlay(BaseModel):
    """What one self-play loop plays, how it learns and where it runs.

    The match (or deck source), observation and model configs bind the world
    once; a world other than the installed native runtime fails validation,
    as do combinations the collector or learner cannot run.
    """

    model_config = ConfigDict(extra="forbid")

    world: str = Field(default_factory=lambda: managym.WORLD_VERSION)
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
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Precision = "float32"
    workers: Literal[1, 2, 4] = 1
    threads: int = Field(default=1, ge=1, le=4)
    bucketed_shapes: bool = False
    live_tokens: bool = False
    # A published policy to start from; only its weights are inherited.
    pretrained: Path | None = None
    # Checkpoint at absolute updates 0, n, 2n, …; a continued run keeps the grid.
    checkpoint_every: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_settings(self) -> "SelfPlay":
        if isinstance(self.learning, Learning) and self.learning.horizon is None:
            raise ValueError("PPO self-play requires a learning horizon")
        if self.world != managym.WORLD_VERSION:
            raise ValueError("self-play world differs from native runtime")
        if (self.match is None) == (self.deck_source is None):
            raise ValueError("self-play needs exactly one match or deck source")
        if self.model.value_kind == "categorical_wdl" and not isinstance(
            self.learning, DampedPolicyLearning
        ):
            raise ValueError("categorical outcome training requires damped_policy")
        if self.model.belief_count_buckets:
            raise ValueError("self-play does not produce belief inputs")
        if self.precision != "float32" and self.device != "cuda":
            raise ValueError(f"precision {self.precision} requires CUDA")
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
        settings.model_dump(mode="json", exclude={"device", "workers", "threads"})
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
        seed=seed + COLLECTION_SEED + start,
        opponent_mode="random" if settings.behavior == "random" else "self",
        learning_seats=settings.learning_seats,
        batch_closure=settings.batch_closure,
        games_per_stream=settings.games_per_stream,
        game_errors=settings.game_errors,
        record_attempt=record_attempt,
        matchup_curriculum=settings.matchup_curriculum,
        deck_source=settings.deck_source,
        device=settings.device,
        workers=settings.workers,
        precision=settings.precision,
    )

    collector.profiler.enabled = True
    return collector


# ---------------------------------------------------------------------------
# Cumulative work, saved by Etude alongside the framework objects


class Counts(BaseModel):
    """Cumulative completed work, changed in place so Etude saves this object."""

    updates: int = 0
    training_seconds: float = 0.0
    games: int = 0
    environment_decisions: int = 0
    learner_transitions: int = 0
    idle_stream_steps: int = 0
    optimizer_updates: int = 0
    optimizer_exposures: int = 0


def behavior(
    settings: SelfPlay, learner: Learner, ema: ActorCritic | None
) -> ActorCritic:
    """The EMA plays for ema-self; otherwise the raw learner plays."""
    if settings.behavior == "ema-self":
        assert ema is not None
        return ema
    return learner.agent


def pretrained_weights(
    settings: SelfPlay, space: ObservationSpace
) -> dict[str, torch.Tensor]:
    """The published weights to start from, admitted against these settings."""
    assert settings.pretrained is not None
    with torch.random.fork_rng(devices=[]):
        published, published_space = load_checkpoint(str(settings.pretrained))
    if (
        published.config != settings.model
        or published_space.encoder.config != space.encoder.config
    ):
        raise ValueError("pretrained policy has other model or observation configs")
    if settings.deck_source is not None:
        validate_agent_decks(published, settings.deck_source.players())
    else:
        validate_agent_setup(published, Match(root_match(settings)).to_rust())
    weights = published.state_dict()
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("pretrained policy contains nonfinite tensors")
    return weights


def create(settings: SelfPlay, *, seed: int) -> tuple[Learner, ActorCritic | None]:
    """Fresh weights from `seed` (or the pretrained policy), EMA a copy, fresh Adam."""
    space = ObservationSpace(settings.observation)
    torch.manual_seed(seed)
    agent = ActorCritic(
        space,
        settings.model,
        bucketed_shapes=settings.bucketed_shapes,
        live_tokens=settings.live_tokens,
    ).to(settings.device)
    if settings.pretrained is not None:
        agent.load_state_dict(pretrained_weights(settings, space))
    learner = Learner(
        agent,
        settings.device,
        minibatch_rows=settings.streams,
        precision=settings.precision,
    )
    ema = deepcopy(agent) if settings.learning.ema is not None else None
    return learner, ema


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
    # A deck-source run binds the world (`None`); a fixed match binds its setups.
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
                "counts": counts.model_dump(mode="json"),
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
    run: Run,
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
        directory / f"{name}-raw.pt",
        "raw",
        seed=run.seed,
    )
    if ema is not None:
        path = write_policy(
            ema, settings, counts, directory / f"{name}-ema.pt", "ema", seed=run.seed
        )
    run.checkpoint(
        path, updates=counts.updates, training_seconds=counts.training_seconds
    )
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
    games, decisions, transitions, idle = (
        stats.games,
        stats.micro_steps,
        stats.learner_transitions,
        stats.idle_stream_steps,
    )
    before = phase_seconds(collector.profiler, learner.profiler)
    batch: RolloutBatch | None = None
    try:
        batch = collect(
            collector, behavior(settings, learner, ema), settings.transitions
        )
        synchronize(device)
        with learner.profiler.track("Learning"):
            metrics = learn(settings, learner, ema, counts, batch, rng)
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
            )
        except Exception as capture_error:
            error.add_note(f"incident capture failed: {capture_error}")
        raise
    phases = time_since(before, phase_seconds(collector.profiler, learner.profiler))
    # The collector's totals are this attempt's; counts span every attempt.
    counts.updates += 1
    counts.training_seconds += phases["time/collection"] + phases["time/learning"]
    counts.games += stats.games - games
    counts.environment_decisions += stats.micro_steps - decisions
    counts.learner_transitions += stats.learner_transitions - transitions
    counts.idle_stream_steps += stats.idle_stream_steps - idle
    counts.optimizer_updates += metrics.get("numerical", {}).get("optimizer_steps", 0)
    counts.optimizer_exposures += metrics.get("optimizer_exposures", 0)
    return (
        metric_row(metrics)
        | {f"coordinates/{k}": v for k, v in counts.model_dump().items()}
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
    settings: SelfPlay, run: Run, counts: Counts
) -> Iterator[Collector]:
    """Scope fresh native streams, append-only game ledgers and failure capture."""
    failures = CaptureScope(
        run.out / "failures",
        lambda: {"seed": run.seed, "update": counts.updates, "out": str(run.out)},
    )
    record_attempt: RecordAttempt | None = None
    errors = GameErrorState()
    if settings.game_errors is not None:
        ledger, errors = game_ledger(run.out, settings.game_errors)

        def record_attempt(
            errors: GameErrorState, attempt: GameAttempt
        ) -> GameErrorState:
            return append_attempt(ledger, errors, attempt, failures.context())

    collector = self_play_collector(
        settings, seed=run.seed, start=counts.updates, record_attempt=record_attempt
    )
    collector.game_error_state = errors
    seed_random(run.seed + counts.updates)
    try:
        with capture_scope(failures):
            yield collector
    finally:
        close_collector(collector)


def self_play(settings: SelfPlay, run: Run, *, updates: int) -> Path:
    """Train to the run's update target and return its final world-bound policy."""
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
        counts = run.resume(**objects, extra=Counts())
        rng = np.random.default_rng(run.seed + MINIBATCH_SEED + counts.updates)
        # Repair an unrecorded milestone without rewriting its immutable bytes.
        if counts.updates == 0 or is_checkpoint(settings, counts.updates):
            publish(settings, run, learner, ema, counts)
        with self_play_games(settings, run, counts) as collector:
            for update in run.steps("updates", to=updates):
                row = step(settings, learner, ema, collector, counts, rng, out=run.out)
                run.log(row, training_seconds=counts.training_seconds)
                if settings.behavior != "random":
                    run.log_table(
                        OUTCOME_TABLE,
                        collector.stats.self_play.drain(),
                        training_seconds=counts.training_seconds,
                    )
                run.save()
                if is_checkpoint(settings, update):
                    publish(settings, run, learner, ema, counts)
        return publish(settings, run, learner, ema, counts)
