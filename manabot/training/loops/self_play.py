"""The self-play loop every current recipe uses, written to be read and copied.

`self_play(settings, out, seed=, updates=, run=)` plays games with the current
learner, learns from each batch and saves the learner to `out / "state.pt"`
atomically. Running it again with the same arguments resumes from that file; a
finished loop returns its final policy at once. Nothing else drives it: there
is no executor, attempt record or lease here. A tracker (`Tracker`, which
Etude's run satisfies) receives rows, checkpoints and the stop signal.

`SelfPlay` holds validated settings; `LoopState` carries the learner.
`step` collects and learns, returning the next state and its metric row.
`new_state`, `load_state`, `save_state` and `save_checkpoint` own the state
file and published policies.

Seeds. With run seed `s` and starting update `u` (0 for a fresh loop): model
initialization uses `s`; native streams `s + 10000 + u`; Torch, NumPy and
Python sampling `s + u`; minibatch order `s + 20000 + u`. At `u = 0` these are
the streams the retired regime executor used for a fresh one-stage self-play
regime. This preserves seed offsets; numerical parity also depends on the
collector and update implementation. A resumed loop starts fresh, reproducible
streams at the update it resumes from.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import time
from typing import Literal, Protocol
from uuid import uuid4

from etude.stopping import Stopped
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
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
    checkpoint_source,
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
    CollectorStats,
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
from manabot.training.update_metrics import OUTCOME_TABLE, metric_row
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from manabot.training.updates.selection import UpdateDiagnostics
import managym

# Offsets that separate a run seed's random streams (see the module docstring).
COLLECTION_SEED = 10_000
MINIBATCH_SEED = 20_000


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
    learning: Learning | DampedPolicyLearning = Learning()
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
    save_every: int = Field(default=1, ge=1)
    # Checkpoint at absolute updates 0, n, 2n, …; a continued run keeps the grid.
    checkpoint_every: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_settings(self) -> "SelfPlay":
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


def learner_settings(settings: SelfPlay) -> dict[str, JsonValue]:
    """Resume identity fields; device, workers and threads may change between attempts."""
    return settings.model_dump(mode="json", exclude={"device", "workers", "threads"})


def learner_digest(settings: SelfPlay) -> str:
    """Identity of the learner `settings` train; machine fields excluded."""
    return canonical_sha256(learner_settings(settings))


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
    return open_collector(
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


# ---------------------------------------------------------------------------
# What a loop asks of a tracker


class Tracker(Protocol):
    """Etude's run satisfies this; pass `None` to a loop to train untracked."""

    def log(self, values: dict[str, object], /, **clocks: float) -> None: ...

    def log_table(
        self, name: str, rows: list[dict[str, object]], /, **clocks: float
    ) -> object: ...

    def checkpoint(
        self, path: Path, /, *, tags: Sequence[str] = (), **clocks: float
    ) -> object: ...

    def stopping(self) -> bool: ...


# ---------------------------------------------------------------------------
# The learner a loop carries


@dataclass(frozen=True)
class Counts:
    """Cumulative work across every attempt of the loop."""

    games: int = 0
    environment_decisions: int = 0
    learner_transitions: int = 0
    idle_stream_steps: int = 0
    optimizer_updates: int = 0
    optimizer_exposures: int = 0


@dataclass(frozen=True)
class LoopState:
    """Raw learner and Adam, EMA, the absolute update count and cumulative counts.

    The learner, its optimizer and the EMA are Torch objects that `step`
    updates in place; the update count and counts are replaced, so each
    `step` returns the next state.
    """

    settings: SelfPlay
    seed: int
    space: ObservationSpace
    learner: Learner
    ema: ActorCritic | None
    update: int = 0
    counts: Counts = Counts()


def behavior(state: LoopState) -> ActorCritic:
    """The policy that plays the games: the EMA for ema-self, else the raw learner."""
    if state.settings.behavior == "ema-self":
        assert state.ema is not None
        return state.ema
    return state.learner.agent


def create_model(settings: SelfPlay, space: ObservationSpace) -> ActorCritic:
    return ActorCritic(
        space,
        settings.model,
        bucketed_shapes=settings.bucketed_shapes,
        live_tokens=settings.live_tokens,
    ).to(settings.device)


def create_learner(settings: SelfPlay, agent: ActorCritic) -> Learner:
    return Learner(
        agent,
        settings.device,
        minibatch_rows=settings.streams,
        precision=settings.precision,
    )


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


def new_state(settings: SelfPlay, seed: int) -> LoopState:
    """Fresh weights from `seed` (or the pretrained policy), EMA a copy, fresh Adam."""
    space = ObservationSpace(settings.observation)
    torch.manual_seed(seed)
    agent = create_model(settings, space)
    if settings.pretrained is not None:
        agent.load_state_dict(pretrained_weights(settings, space))
    learner = create_learner(settings, agent)
    ema = deepcopy(agent) if settings.learning.ema is not None else None
    return LoopState(settings, seed, space, learner, ema)


def setting_differences(
    saved: dict[str, JsonValue], current: dict[str, JsonValue], prefix: str = ""
) -> list[str]:
    """Describe changed learner fields, including nested learning/model settings."""
    differences: list[str] = []
    for key in sorted(saved.keys() | current.keys()):
        name = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            differences.append(
                f"{name}: missing in saved state, requested {current[key]!r}"
            )
        elif key not in current:
            differences.append(
                f"{name}: saved {saved[key]!r}, missing in requested settings"
            )
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            differences.extend(setting_differences(saved[key], current[key], name))
        elif saved[key] != current[key]:
            differences.append(
                f"{name}: saved {saved[key]!r}, requested {current[key]!r}"
            )
    return differences


def load_state(path: Path, settings: SelfPlay, seed: int) -> LoopState | None:
    """The learner saved at `path`, or None when there is none.

    Refuses a state written for other settings, another seed or another world,
    naming what differs.
    """
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for name, expected in (
        ("world", settings.world),
        ("seed", seed),
    ):
        if saved[name] != expected:
            raise ValueError(
                f"{path} was written for another {name}: {saved[name]!r}, not {expected!r}"
            )
    if saved["settings"] != learner_digest(settings):
        stored = TypeAdapter(dict[str, JsonValue]).validate_python(
            saved["learner_settings"]
        )
        current = learner_settings(settings)
        differences = setting_differences(stored, current)
        raise ValueError(
            f"{path} has different learner settings: " + "; ".join(differences)
        )
    space = ObservationSpace(settings.observation)
    agent = create_model(settings, space)
    agent.load_state_dict(saved["raw"])
    learner = create_learner(settings, agent)
    learner.optimizer.load_state_dict(saved["optimizer"])
    ema = None
    if saved["ema"] is not None:
        ema = deepcopy(learner.agent)
        ema.load_state_dict(saved["ema"])
    return LoopState(
        settings,
        seed,
        space,
        learner,
        ema,
        update=int(saved["update"]),
        counts=Counts(**saved["counts"]),
    )


def save_state(path: Path, state: LoopState) -> None:
    """Write the learner to a temporary file, then rename it over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "format": "manabot.self-play-state/v1",
            "world": state.settings.world,
            "seed": state.seed,
            "settings": learner_digest(state.settings),
            "learner_settings": learner_settings(state.settings),
            "source_commit": checkpoint_source()["commit"],
            "update": state.update,
            "counts": asdict(state.counts),
            "raw": state.learner.agent.state_dict(),
            "ema": None if state.ema is None else state.ema.state_dict(),
            "optimizer": state.learner.optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def write_policy(
    model: ActorCritic, state: LoopState, path: Path, weights: Literal["raw", "ema"]
) -> Path:
    if path.exists():
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    # A deck-source run binds the world (`None`); a fixed match binds its setups.
    players = (
        None
        if state.settings.deck_source is not None
        else Match(root_match(state.settings)).to_rust()
    )
    averaging = {
        "clock": "collect-update-iteration",
        "iteration": state.update,
        "rate": state.settings.learning.ema,
    }
    # A fresh model constructed for admission must not draw from the loop's RNG.
    device = next(model.parameters()).device.type
    with torch.random.fork_rng(
        devices=[0] if device in {"cuda", "mps"} else [],
        device_type="mps" if device == "mps" else "cuda",
    ):
        save_policy(
            model,
            state.space,
            temporary,
            player_configs=players,
            extra={
                "weights": weights,
                "update": state.update,
                "seed": state.seed,
                "value_semantic": "signed_outcome",
                "averaging": averaging if weights == "ema" else None,
            },
        )
        load_checkpoint(str(temporary))
    os.replace(temporary, path)
    return path


def save_checkpoint(directory: Path, state: LoopState) -> Path:
    """Write immutable raw/EMA policies; return EMA when kept, otherwise raw."""
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{state.update:08d}"
    raw = write_policy(state.learner.agent, state, directory / f"{name}-raw.pt", "raw")
    if state.ema is not None:
        return write_policy(state.ema, state, directory / f"{name}-ema.pt", "ema")
    return raw


# ---------------------------------------------------------------------------
# One update


def learn(
    state: LoopState, batch: RolloutBatch, rng: np.random.Generator, *, updates: int
) -> UpdateDiagnostics:
    """Learn from one batch in place: the update its learning record selects, then EMA.

    The update made is `state.update + 1`. Damped-policy schedules read that
    absolute iteration; PPO schedules read the fraction `state.update /
    updates`, so for PPO the target matters.
    """
    settings = state.settings
    if isinstance(settings.learning, DampedPolicyLearning):
        metrics = update_damped_iteration(
            torch_backend,
            state.learner,
            batch,
            settings.learning,
            state.update + 1,
            bootstrap_agent=behavior(state),
        )
    else:
        metrics = update_iteration(
            torch_backend,
            state.learner,
            batch,
            settings.learning,
            state.update / updates,
            rng,
            iteration=state.update + 1,
            bootstrap_agent=behavior(state),
        )
    if state.ema is not None:
        assert settings.learning.ema is not None
        update_ema(state.ema, state.learner.agent, settings.learning.ema)
    return metrics


def counted(
    counts: Counts, base: Counts, stats: CollectorStats, metrics: UpdateDiagnostics
) -> Counts:
    """Counts after one more update.

    `stats` are this attempt's collector totals, added to `base`, the counts
    saved by earlier attempts; optimizer counts accumulate per update.
    """
    return Counts(
        games=base.games + stats.games,
        environment_decisions=base.environment_decisions + stats.micro_steps,
        learner_transitions=base.learner_transitions + stats.learner_transitions,
        idle_stream_steps=base.idle_stream_steps + stats.idle_stream_steps,
        optimizer_updates=counts.optimizer_updates
        + metrics.get("numerical", {}).get("optimizer_steps", 0),
        optimizer_exposures=counts.optimizer_exposures
        + metrics.get("optimizer_exposures", 0),
    )


def step(
    state: LoopState,
    collector: Collector,
    rng: np.random.Generator,
    *,
    updates: int,
    base: Counts,
    out: Path | None = None,
) -> tuple[LoopState, dict[str, object]]:
    """Collect one batch, learn from it, and return the next state and its metric row."""
    device = state.settings.device
    started = time.perf_counter()
    batch: RolloutBatch | None = None
    try:
        batch = collect(collector, behavior(state), state.settings.transitions)
        synchronize(device)
        collected = time.perf_counter()
        metrics = learn(state, batch, rng, updates=updates)
        synchronize(device)
    except NumericalError as error:
        if out is not None:
            try:
                save_incident(
                    out / "numerical-incident" / uuid4().hex,
                    error,
                    state,
                    collector,
                    batch,
                    rng,
                    out=out,
                    settings_digest=learner_digest(state.settings),
                )
            except Exception as capture_error:
                error.add_note(f"incident capture failed: {capture_error}")
        raise
    state = replace(
        state,
        update=state.update + 1,
        counts=counted(state.counts, base, collector.stats, metrics),
    )
    row = (
        metric_row(metrics)
        | {f"coordinates/{k}": v for k, v in asdict(state.counts).items()}
        | {
            "time/collection": collected - started,
            "time/learning": time.perf_counter() - collected,
        }
    )
    return state, row


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


def self_play(
    settings: SelfPlay,
    out: Path,
    *,
    seed: int,
    updates: int,
    run: Tracker | None = None,
) -> Path:
    """Train to the absolute update `updates` and return the final policy's path.

    Saves `out / "state.pt"` every `settings.save_every` updates and when it
    stops; raises `Stopped` when `run.stopping()` ends it early. Run it again
    to carry on.
    """
    settings = SelfPlay.model_validate(settings.model_dump())
    if updates < 1:
        raise ValueError("updates must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with device_settings(settings):
        state = load_state(out / "state.pt", settings, seed)
        if state is None:
            if any((out / "checkpoints").glob("*.pt")):
                raise ValueError(
                    "checkpoints exist without state.pt; choose a fresh output directory"
                )
            state = new_state(settings, seed)
        if state.update >= updates:
            return record_checkpoint(out, state, run, final=True)

        # Only the learner is durable. Each attempt starts fresh random streams
        # and adds its collector's totals to the counts saved before it.
        base = state.counts
        rng = np.random.default_rng(seed + MINIBATCH_SEED + state.update)
        # Reads `state` when a failure is captured, so it names the current update.
        failures = CaptureScope(
            out / "failures",
            lambda: {"seed": seed, "update": state.update, "out": str(out)},
        )
        record_attempt: RecordAttempt | None = None
        errors = GameErrorState()
        if settings.game_errors is not None:
            ledger, errors = game_ledger(out, settings.game_errors)

            def record_attempt(
                errors: GameErrorState, attempt: GameAttempt
            ) -> GameErrorState:
                return append_attempt(ledger, errors, attempt, failures.context())

        collector = self_play_collector(
            settings, seed=seed, start=state.update, record_attempt=record_attempt
        )
        collector.game_error_state = errors
        # After the collector, which may draw from these streams while it starts.
        seed_random(seed + state.update)
        try:
            with capture_scope(failures):
                if state.update == 0 and is_checkpoint(settings, 0):
                    save_state(out / "state.pt", state)
                    record_checkpoint(out, state, run)
                while state.update < updates and not (run and run.stopping()):
                    state, row = step(
                        state, collector, rng, updates=updates, base=base, out=out
                    )
                    if run:
                        run.log(row, updates=state.update)
                        if settings.behavior != "random":
                            outcomes = collector.stats.self_play.drain()
                            run.log_table(OUTCOME_TABLE, outcomes, updates=state.update)
                    checkpoint_due = state.update < updates and is_checkpoint(
                        settings, state.update
                    )
                    # Commit resume state once, before publishing immutable policy bytes.
                    if checkpoint_due or state.update % settings.save_every == 0:
                        save_state(out / "state.pt", state)
                    if checkpoint_due:
                        record_checkpoint(out, state, run)
        finally:
            close_collector(collector)
        save_state(out / "state.pt", state)
        if state.update < updates:
            raise Stopped(f"stopped at update {state.update}; run again to resume")
        return record_checkpoint(out, state, run, final=True)


def record_checkpoint(
    out: Path, state: LoopState, run: Tracker | None, *, final: bool = False
) -> Path:
    path = save_checkpoint(out / "checkpoints", state)
    if run:
        run.checkpoint(path, updates=state.update, tags=("final",) if final else ())
    return path
