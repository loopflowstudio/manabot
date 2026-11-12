"""The self-play loop every current recipe uses, written to be read and copied.

`self_play(settings, out, seed=, updates=, run=)` plays games with the current
learner, learns from each batch with the update its learning record selects,
and saves its learner to `out / "state.pt"` atomically. Running it again with
the same arguments resumes from that file; a finished loop returns its final
policy at once. Nothing else drives it: there is no executor, attempt record or
lease here. A tracker (`Tracker`, which Etude's run satisfies) receives rows,
checkpoints and the stop signal.

Principal types: `SelfPlay` (settings, a Pydantic model so a recipe config that
holds one serializes), `LoopState` (the learner a loop carries), `Tracker`,
`Stopped` and `Every`. The pieces `new_state`, `load_state`, `save_state`,
`save_checkpoint` and `save_final` are what the loop body calls.

Seeds. With run seed `s` and starting update `u` (0 for a fresh loop): model
initialization uses `s`; native streams `s + 10000 + u`; Torch, NumPy and
Python sampling `s + u`; minibatch order `s + 20000 + u`. At `u = 0` these are
the streams the retired regime executor used for a fresh one-stage self-play
regime, so the same seed produces the same updates and checkpoints. A resumed
loop starts fresh, reproducible streams at the update it resumes from.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import time
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

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
from manabot.infra.runtime import seed_random
from manabot.infra.source import source_commit
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint as save_policy
from manabot.model.world import validate_agent_setup
from manabot.training.decks import UniformDecks
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_ema, update_iteration
from manabot.training.learning.selection import UpdateDiagnostics
from manabot.training.precision import Precision, apply_matmul_precision
from manabot.training.regime import (
    BatchClosure,
    DampedPolicyLearning,
    Execution,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
)
from manabot.training.update_metrics import OUTCOME_TABLE, metric_row
import managym

# Offsets that separate a run seed's random streams (see the module docstring).
COLLECTION_SEED = 10_000
MINIBATCH_SEED = 20_000
# Settings that describe the machine, not the learner; a resumed loop may change them.
_MACHINE_FIELDS = frozenset({"device", "workers", "threads"})


# ---------------------------------------------------------------------------
# Settings


class Every(BaseModel):
    """Every `every`-th absolute update, including update 0 (the initialization)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    every: int = Field(ge=1)

    def __contains__(self, update: object) -> bool:
        return isinstance(update, int) and update % self.every == 0


def every(n: int) -> Every:
    """Milestones at absolute updates 0, n, 2n, …; a continued run keeps the grid."""
    return Every(every=n)


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
    milestones: tuple[int, ...] | Every = ()

    @model_validator(mode="after")
    def _runs_here(self) -> "SelfPlay":
        from manabot.training.execution import validate_regime

        validate_regime(self.regime(id="self-play", updates=1))
        return self

    def regime(self, *, id: str, updates: int) -> TrainingRegime:
        """The one-stage saved-regime form of these settings.

        Validation reuses the regime's checks, and the parity test runs the
        retired executor on it. Wall and memory limits are left to the tracker.
        """
        stage = TrainSelfPlay(
            id="self-play",
            operation="train_self_play",
            behavior=self.behavior,
            updates=updates,
            streams=self.streams,
            transitions=self.transitions,
            games_per_stream=self.games_per_stream,
            learning_seats=self.learning_seats,
            batch_closure=self.batch_closure,
            matchup_curriculum=self.matchup_curriculum,
            learning=self.learning.model_copy(deep=True),
            game_errors=self.game_errors,
            execution=Execution(
                device=self.device,
                precision=self.precision,
                workers=self.workers,
                threads=self.threads,
                wall_seconds=1e9,
                bucketed_shapes=self.bucketed_shapes,
                live_tokens=self.live_tokens,
            ),
        )
        return TrainingRegime(
            id=id,
            world=self.world,
            match=self.match,
            deck_source=self.deck_source,
            observation=self.observation,
            agent=self.model,
            stages=[stage],
            wall_seconds=1e9,
            schedule_clock="iteration_fraction",
        )

    @property
    def root_match(self) -> MatchConfig:
        """The authored match, or a deck source's representative opening pair."""
        if self.deck_source is not None:
            return self.deck_source.match_config()
        if self.match is None:
            raise ValueError("self-play needs a match or a deck source")
        return self.match

    def learner_digest(self) -> str:
        """Identity of the learner these settings train; machine fields excluded."""
        return canonical_sha256(
            self.model_dump(mode="json", exclude=set(_MACHINE_FIELDS))
        )

    def collector(
        self,
        *,
        seed: int,
        start: int,
        record_attempt: Callable[[GameAttempt], GameErrorState] | None = None,
    ) -> SeatRoutedCollector:
        """Native streams seeded from the run seed and the update they start at.

        The caller closes it. `record_attempt` receives each settled game when
        `game_errors` is set.
        """
        return SeatRoutedCollector(
            ObservationSpace(self.observation),
            Match(self.root_match),
            Reward(RewardConfig()),
            num_envs=self.streams,
            seed=seed + COLLECTION_SEED + start,
            opponent_mode="random" if self.behavior == "random" else "self",
            learning_seats=self.learning_seats,
            batch_closure=self.batch_closure,
            games_per_stream=self.games_per_stream,
            game_errors=self.game_errors,
            record_attempt=record_attempt,
            matchup_curriculum=self.matchup_curriculum,
            deck_source=self.deck_source,
            device=self.device,
            workers=self.workers,
            precision=self.precision,
        )

    def update(
        self,
        learner: Learner,
        batch: RolloutBatch,
        *,
        rng: np.random.Generator,
        iteration: int,
        updates: int,
        bootstrap: ActorCritic,
    ) -> UpdateDiagnostics:
        """Learn from one batch; `iteration` is the absolute update being made.

        Damped-policy schedules read `iteration`; PPO schedules read the
        fraction `(iteration - 1) / updates`, so for PPO the target matters.
        """
        return update_iteration(
            learner,
            batch,
            self.learning,
            (iteration - 1) / updates,
            rng,
            iteration=iteration,
            bootstrap_agent=bootstrap,
        )


# ---------------------------------------------------------------------------
# What a loop asks of a tracker


class Tracker(Protocol):
    """Etude's run satisfies this; pass `None` to a loop to train untracked."""

    def log(self, values: dict[str, object], /, **clocks: float) -> None: ...

    def log_table(
        self, name: str, rows: list[dict[str, object]], /, **clocks: float
    ) -> object: ...

    def checkpoint(self, path: Path, /, **clocks: float) -> object: ...

    def stopping(self) -> bool: ...


class Stopped(Exception):
    """The loop saved its state at `update` and stopped before its target."""

    def __init__(self, update: int) -> None:
        super().__init__(f"stopped at update {update}; run again to resume")
        self.update = update


# ---------------------------------------------------------------------------
# The learner a loop carries


@dataclass
class Counts:
    """Cumulative work across every attempt of the loop."""

    games: int = 0
    environment_decisions: int = 0
    learner_transitions: int = 0
    idle_stream_steps: int = 0
    optimizer_updates: int = 0
    optimizer_exposures: int = 0


@dataclass
class LoopState:
    """Raw learner and Adam, EMA, the absolute update count and cumulative counts."""

    settings: SelfPlay
    seed: int
    space: ObservationSpace
    learner: Learner
    ema: ActorCritic | None
    update: int = 0
    counts: Counts = field(default_factory=Counts)

    @property
    def behavior(self) -> ActorCritic:
        if self.settings.behavior == "ema-self":
            assert self.ema is not None
            return self.ema
        return self.learner.agent

    def advance(
        self, collector: SeatRoutedCollector, metrics: UpdateDiagnostics, base: Counts
    ) -> None:
        """Count one completed update; `collector` stats are this attempt's totals."""
        self.update += 1
        self.counts.optimizer_updates += metrics.get("numerical", {}).get(
            "optimizer_steps", 0
        )
        self.counts.optimizer_exposures += metrics.get("optimizer_exposures", 0)
        stats = collector.stats
        self.counts.games = base.games + stats.games
        self.counts.environment_decisions = (
            base.environment_decisions + stats.micro_steps
        )
        self.counts.learner_transitions = (
            base.learner_transitions + stats.learner_transitions
        )
        self.counts.idle_stream_steps = base.idle_stream_steps + stats.idle_stream_steps


def _model(settings: SelfPlay, space: ObservationSpace) -> ActorCritic:
    return ActorCritic(
        space,
        settings.model,
        bucketed_shapes=settings.bucketed_shapes,
        live_tokens=settings.live_tokens,
    ).to(settings.device)


def _learner(settings: SelfPlay, agent: ActorCritic) -> Learner:
    return Learner(
        agent,
        settings.device,
        minibatch_rows=settings.streams,
        precision=settings.precision,
    )


def new_state(settings: SelfPlay, seed: int) -> LoopState:
    """Fresh weights from `seed` (or the pretrained policy), EMA a copy, fresh Adam."""
    space = ObservationSpace(settings.observation)
    torch.manual_seed(seed)
    agent = _model(settings, space)
    if settings.pretrained is not None:
        with torch.random.fork_rng(devices=[]):
            published, published_space = load_checkpoint(str(settings.pretrained))
        if (
            published.config != settings.model
            or published_space.encoder.config != space.encoder.config
        ):
            raise ValueError("pretrained policy has other model or observation configs")
        validate_agent_setup(published, Match(settings.root_match).to_rust())
        if any(not torch.isfinite(v).all() for v in published.state_dict().values()):
            raise ValueError("pretrained policy contains nonfinite tensors")
        agent.load_state_dict(published.state_dict())
    learner = _learner(settings, agent)
    ema = deepcopy(agent) if settings.learning.ema is not None else None
    return LoopState(settings, seed, space, learner, ema)


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
        ("settings", settings.learner_digest()),
    ):
        if saved[name] != expected:
            raise ValueError(
                f"{path} was written for another {name}: {saved[name]!r}, not {expected!r}"
            )
    space = ObservationSpace(settings.observation)
    agent = _model(settings, space)
    agent.load_state_dict(saved["raw"])
    learner = _learner(settings, agent)
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
            "settings": state.settings.learner_digest(),
            "source_commit": source_commit(),
            "update": state.update,
            "counts": vars(state.counts),
            "raw": state.learner.agent.state_dict(),
            "ema": None if state.ema is None else state.ema.state_dict(),
            "optimizer": state.learner.optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _player_configs(settings: SelfPlay) -> list[managym.PlayerConfig] | None:
    """A deck-source run binds the world (`None`); a fixed match binds its setups."""
    return (
        None
        if settings.deck_source is not None
        else Match(settings.root_match).to_rust()
    )


def _save_policy(
    model: ActorCritic, state: LoopState, path: Path, weights: Literal["raw", "ema"]
) -> Path:
    temporary = path.with_name(f".{path.name}.tmp")
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
            player_configs=_player_configs(state.settings),
            extra={
                "weights": weights,
                "update": state.update,
                "seed": state.seed,
                "value_semantic": "signed_outcome",
                "averaging": {
                    "clock": "collect-update-iteration",
                    "iteration": state.update,
                    "rate": state.settings.learning.ema,
                }
                if weights == "ema"
                else None,
            },
        )
        load_checkpoint(str(temporary))
    os.replace(temporary, path)
    return path


def save_checkpoint(directory: Path, state: LoopState) -> Path:
    """World-bound raw (returned) and EMA policies at this update, named by update."""
    directory.mkdir(parents=True, exist_ok=True)
    raw = _save_policy(
        state.learner.agent, state, directory / f"{state.update:08d}-raw.pt", "raw"
    )
    if state.ema is not None:
        _save_policy(state.ema, state, directory / f"{state.update:08d}-ema.pt", "ema")
    return raw


def save_final(path: Path, state: LoopState) -> Path:
    """The EMA policy (raw without EMA) at the last completed update."""
    if state.ema is not None:
        return _save_policy(state.ema, state, path, "ema")
    return _save_policy(state.learner.agent, state, path, "raw")


# ---------------------------------------------------------------------------
# Running on a device and recording game failures


@contextmanager
def _device(settings: SelfPlay) -> Iterator[None]:
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


def _ledger(out: Path, policy: GameErrorPolicy) -> GameErrorLedger:
    """This attempt's append-only ledger, counting every earlier attempt's games."""
    directory = out / "game-attempts"
    directory.mkdir(parents=True, exist_ok=True)
    earlier = sorted(directory.glob("*.jsonl"))
    return GameErrorLedger(
        directory / f"{len(earlier):04d}.jsonl",
        policy,
        parents=[str(path) for path in earlier],
    )


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


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
    out = Path(out)
    final = out / "final.pt"
    if updates < 1:
        raise ValueError("updates must be positive")
    out.mkdir(parents=True, exist_ok=True)
    with _device(settings):
        state = load_state(out / "state.pt", settings, seed)
        if state is None:
            if final.exists():
                raise ValueError(f"{final} has no state.pt to validate or resume")
            state = new_state(settings, seed)
        if state.update >= updates:
            return final if final.exists() else save_final(final, state)
        # A later target invalidates the earlier endpoint, including during a stop.
        final.unlink(missing_ok=True)
        # Only the learner is durable. Each invocation starts fresh random streams
        # and adds this collector's totals to the counts saved by its predecessor.
        base = replace(state.counts)
        rng = np.random.default_rng(seed + MINIBATCH_SEED + state.update)
        ledger = _ledger(out, settings.game_errors) if settings.game_errors else None
        failures = CaptureScope(
            out / "failures",
            lambda: {"seed": seed, "update": state.update, "out": str(out)},
        )
        collector = settings.collector(
            seed=seed,
            start=state.update,
            record_attempt=None
            if ledger is None
            else lambda attempt: ledger.record(attempt, failures.context()),
        )
        try:
            if ledger is not None:
                collector.game_error_state = ledger.state
                if ledger.state.stop_reason:
                    raise GameErrorLimit(ledger.state.stop_reason)
            seed_random(seed + state.update)
            with capture_scope(failures):
                if state.update == 0 and 0 in settings.milestones:
                    _milestone(out, state, run)
                while state.update < updates and not (run and run.stopping()):
                    started = time.perf_counter()
                    batch = collector.collect(state.behavior, settings.transitions)
                    _synchronize(settings.device)
                    collected = time.perf_counter()
                    metrics = settings.update(
                        state.learner,
                        batch,
                        rng=rng,
                        iteration=state.update + 1,
                        updates=updates,
                        bootstrap=state.behavior,
                    )
                    if state.ema is not None:
                        assert settings.learning.ema is not None
                        update_ema(
                            state.ema, state.learner.agent, settings.learning.ema
                        )
                    _synchronize(settings.device)
                    state.advance(collector, metrics, base)
                    if run:
                        run.log(
                            metric_row(metrics)
                            | {
                                f"coordinates/{k}": v
                                for k, v in vars(state.counts).items()
                            }
                            | {
                                "time/collection": collected - started,
                                "time/learning": time.perf_counter() - collected,
                            },
                            updates=state.update,
                        )
                        if settings.behavior != "random":
                            run.log_table(
                                OUTCOME_TABLE,
                                collector.stats.self_play.drain(),
                                updates=state.update,
                            )
                    if state.update % settings.save_every == 0:
                        save_state(out / "state.pt", state)
                    if state.update in settings.milestones:
                        _milestone(out, state, run)
        finally:
            collector.close()
        save_state(out / "state.pt", state)
        if state.update < updates:
            raise Stopped(state.update)
        return save_final(final, state)


def _milestone(out: Path, state: LoopState, run: Tracker | None) -> None:
    checkpoint = save_checkpoint(out / "checkpoints", state)
    if run:
        run.checkpoint(checkpoint, updates=state.update)
