"""Private native failure evidence, independent of training error tolerance.

ScalarCapture retains the current game's explicit inputs; CapturedVector retains
one prefix per native stream, including masked calls and native reset strides.
Strict calls always re-raise the original exception after fsync. Semantic calls
also journal pending Commands before execution. CaptureScope adds the owning
run's live coordinates without adding fields to frozen training recipes.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

import managym


class PlayerSetup(BaseModel):
    name: str
    decklist: dict[str, int]
    sideboard: dict[str, int]
    content_pack: str | None

    @classmethod
    def retain(cls, player: managym.PlayerConfig) -> "PlayerSetup":
        return cls(
            name=player.name,
            decklist=player.decklist,
            sideboard=player.sideboard,
            content_pack=player.content_pack,
        )

    def native(self) -> managym.PlayerConfig:
        return managym.PlayerConfig(**self.model_dump())


class GameInputs(BaseModel):
    stream: int = 0
    seed: int
    players: list[PlayerSetup]
    actions: list[int | str] = []
    actor: int | None = None
    frame: str | None = None

    def restart(self, seed: int) -> None:
        """Keep the stream's setup, discard all inputs from its previous game."""
        self.seed = seed
        self.actions = []
        self.actor = None
        self.frame = None


class BufferSpec(BaseModel):
    shape: list[int]
    dtype: str


class SearchInputs(BaseModel):
    method: Literal["rollout_pool", "flat_mc_scores"]
    arguments: list[int]
    operations: list[list[int] | Literal["finish_random"]] = []
    buffers: dict[str, BufferSpec] = {}
    capacity: int = 0


class FailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["native-failure-v1"] = Field(
        default="native-failure-v1", alias="schema"
    )
    mode: Literal["scalar", "vector", "search"]
    search: SearchInputs | None = None
    operation: Literal["reset", "step"]
    games: list[GameInputs]
    skip_trivial: bool = True
    opponent_policy: str = "none"
    seed_stride: int = 1
    identities: dict[str, object]
    context: dict[str, object]
    seconds: float
    error_class: str | None = None
    error: str | None = None
    traceback: str | None = None
    execution_outcome: Literal[
        "failed", "execution_outcome_unknown", "last_command_succeeded"
    ] = "execution_outcome_unknown"


@dataclass
class CaptureScope:
    directory: Path
    context: Callable[[], dict[str, object]] = dict
    paths: list[Path] = field(default_factory=list)


_scope: ContextVar[CaptureScope | None] = ContextVar("game_failure_scope", default=None)


@contextmanager
def capture_scope(scope: CaptureScope) -> Iterator[None]:
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


@cache
def runtime_identities() -> dict[str, object]:
    """Bind source/native bytes once, when this process first constructs a capture."""
    root = Path(__file__).resolve().parents[2]
    sources = hashlib.sha256()
    for directory, suffix in (
        (root / "manabot", "*.py"),
        (root / "managym" / "src", "*.rs"),
    ):
        for path in sorted(directory.rglob(suffix)):
            sources.update(str(path.relative_to(root)).encode())
            sources.update(path.read_bytes())
    extension = Path(managym._managym.__file__)
    return {
        "world": managym.WORLD_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": sources.hexdigest(),
        "engine_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }


def _append(path: Path, row: dict[str, object], *, create: bool = False) -> None:
    flags = os.O_WRONLY | (os.O_CREAT | os.O_EXCL if create else os.O_APPEND)
    with os.fdopen(os.open(path, flags, 0o600), "w") as output:
        output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    if create:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class _Capture:
    def __init__(self) -> None:
        self.scope = _scope.get()
        self.started = time.perf_counter()
        self.identities = runtime_identities().copy()
        self.context: dict[str, object] = {}

    def _retain(self, evidence: FailureEvidence) -> Path:
        """Create and register private evidence only after its bytes are durable."""
        scope = _scope.get() or self.scope
        directory = scope.directory if scope else Path(".runs/game-failures")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{uuid4().hex}.jsonl"
        _append(path, evidence.model_dump(mode="json", by_alias=True), create=True)
        if scope:
            scope.paths.append(path)
        return path

    def _coordinates(self) -> dict[str, object]:
        """Read live owner coordinates equally for failures and pending Commands."""
        scope = _scope.get() or self.scope
        return {
            "context": {**(scope.context() if scope else {}), **self.context},
            "seconds": time.perf_counter() - self.started,
        }

    def _evidence(self, **fields: object) -> FailureEvidence:
        return FailureEvidence.model_validate(
            {
                **fields,
                "identities": self.identities,
                **self._coordinates(),
            }
        )

    @contextmanager
    def _guard(self, evidence: Callable[[], FailureEvidence]) -> Iterator[None]:
        """Capture once when nested boundaries observe the same failure."""
        try:
            yield
        except BaseException as error:
            if not any(
                note.startswith("Private native failure evidence:")
                for note in getattr(error, "__notes__", [])
            ):
                self._failure(error, evidence)
            raise

    def _failure(
        self, error: BaseException, evidence: Callable[[], FailureEvidence]
    ) -> None:
        # Capture failures must never disguise the original native failure.
        try:
            retained = evidence()
            retained.execution_outcome = "failed"
            retained.error_class = type(error).__name__
            retained.error = str(error)
            retained.traceback = "".join(traceback.format_exception(error))
            path = self._retain(retained)
            error.add_note(f"Private native failure evidence: {path}")
        except Exception as capture_error:
            error.add_note(f"Failure evidence could not be retained: {capture_error}")


class ScalarCapture(_Capture):
    def __init__(self, seed: int, skip_trivial: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.skip_trivial = skip_trivial
        self.inputs = GameInputs(seed=seed, players=[])
        self.journal: Path | None = None

    def reset(self, env: managym.Env, players: list[managym.PlayerConfig]) -> Any:
        # The heterogeneous native observation/info tuple stays at this boundary.
        self.started = time.perf_counter()
        self.inputs = GameInputs(
            seed=self.seed, players=[PlayerSetup.retain(p) for p in players]
        )
        self.journal = None
        with self._guard(lambda: self.evidence("reset")):
            result = env.reset(players)
            self.identities["content_manifest"] = env.content_pack_manifest()
            return result

    def evidence(self, operation: Literal["reset", "step"] = "step") -> FailureEvidence:
        return self._evidence(
            mode="scalar",
            operation=operation,
            games=[self.inputs],
            skip_trivial=self.skip_trivial,
        )

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Retain policy/cap/deadline failures outside the native call too."""
        with self._guard(self.evidence):
            yield

    def step(self, env: managym.Env, action: int) -> Any:
        self.inputs.actions.append(int(action))
        self.inputs.actor = env.current_agent_index()
        self.inputs.frame = env.semantic_decision_frame_json()
        with self.guard():
            return env.step(action)

    def command(
        self, env: managym.Env, command: str, *, transition_only: bool = True
    ) -> Any:
        """Fsync exact pending Command and full offer frame before native execution."""
        self.inputs.actor = env.current_agent_index()
        self.inputs.frame = env.semantic_decision_frame_json()
        if self.journal is None:
            self.journal = self._retain(self.evidence())
        _append(
            self.journal,
            {
                "pending": command,
                "frame": self.inputs.frame,
                "actor": self.inputs.actor,
                **self._coordinates(),
            },
        )
        self.inputs.actions.append(command)
        try:
            result = (
                env.execute_semantic_command_json(command)
                if transition_only
                else env.step_semantic_command(command)
            )
        except BaseException as error:
            try:
                _append(
                    self.journal,
                    {
                        "error_class": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(traceback.format_exception(error)),
                    },
                )
                error.add_note(f"Private native failure evidence: {self.journal}")
            except Exception as capture_error:
                error.add_note(
                    f"Failure evidence could not be retained: {capture_error}"
                )
            raise
        _append(self.journal, {"success": True})
        return result


class CapturedVector(_Capture):
    """Instrument the strict native vector API without isolating or retrying rows.

    On batch failure every active row retains its pre-call prefix. Native parallel
    execution may already have advanced other rows; their outcome is unknown,
    not a crashed/completed count. Replay tests each retained row independently.
    Other native APIs, including opt-in isolation, retain their existing owner.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int = 0,
        skip_trivial: bool = True,
        opponent_policy: str = "none",
        seed_stride: int | None = None,
        stream_offset: int = 0,
    ) -> None:
        super().__init__()
        self.stride = seed_stride or num_envs
        self.skip_trivial = skip_trivial
        self.opponent_policy = opponent_policy
        self.native = managym.VectorEnv(
            num_envs=num_envs,
            seed=seed,
            skip_trivial=skip_trivial,
            opponent_policy=opponent_policy,
            seed_stride=self.stride,
        )
        self.games = [
            GameInputs(
                stream=stream_offset + i, seed=min(seed + i, 2**64 - 1), players=[]
            )
            for i in range(num_envs)
        ]
        self.next_seeds = [game.seed for game in self.games]
        self._roster: list[PlayerSetup] = []

    def __getattr__(self, name: str) -> Any:
        # PyO3 buffer/inspection APIs are untyped; mutation is explicit below.
        return getattr(self.native, name)

    def set_buffers(self, buffers: dict[str, Any]) -> None:
        # NumPy owns these buffers; preserve their native encoding ABI as evidence.
        self.identities["input_buffers"] = {
            key: {"shape": list(value.shape[1:]), "dtype": str(value.dtype)}
            for key, value in buffers.items()
        }
        self.native.set_buffers(buffers)

    def capture_failure(self, error: BaseException) -> None:
        self._failure(error, lambda: self._evidence_for(self.games, "step"))

    def _evidence_for(
        self, games: list[GameInputs], operation: Literal["reset", "step"]
    ) -> FailureEvidence:
        return self._evidence(
            mode="vector",
            operation=operation,
            games=games,
            skip_trivial=self.skip_trivial,
            opponent_policy=self.opponent_policy,
            seed_stride=self.stride,
        )

    def set_uniform_roster(self, roster: list[managym.PlayerConfig]) -> None:
        self.native.set_uniform_roster(roster)
        self._roster = [PlayerSetup.retain(player) for player in roster]

    def set_training_decks(
        self, decks: list[managym.PlayerConfig], digest: str
    ) -> None:
        """Replace decks for later roots; evidence keeps each game's actual setup."""
        self.native.set_training_decks(decks)
        self._roster = [PlayerSetup.retain(player) for player in decks]
        self.identities["training_decks"] = {
            "digest": digest,
            "sampling_policy": "uniform-distinct-v1",
            "deck_ids": [player.name for player in decks],
        }

    def reset_all_into_buffers(self, players: list[managym.PlayerConfig]) -> None:
        self.reset_each_into_buffers([players for _ in self.games])

    def reset_each_into_buffers(
        self, players: list[list[managym.PlayerConfig]]
    ) -> None:
        for game, setup, seed in zip(self.games, players, self.next_seeds, strict=True):
            game.players = [PlayerSetup.retain(p) for p in setup]
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self.native.reset_each_into_buffers(players)
        self._synchronize(reset=True)

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        self.next_seeds = list(seeds)
        for game, seed in zip(self.games, seeds, strict=True):
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self.native.reset_seeds_into_buffers(seeds)
        self._synchronize(reset=True)

    def _synchronize(self, *, reset: bool = False) -> None:
        seeds = self.native.current_game_seeds()
        if self._roster and (
            reset or any(game.seed != seed for game, seed in zip(self.games, seeds))
        ):
            # Sampled roots record the native setup actually created.
            for game, players in zip(
                self.games, self.native.current_player_configs(), strict=True
            ):
                game.players = [PlayerSetup.retain(player) for player in players]
        for game, seed in zip(self.games, seeds, strict=True):
            if game.seed != seed:
                game.restart(seed)
        self.next_seeds = [
            min(game.seed + self.stride, 2**64 - 1) for game in self.games
        ]

    def step_into_buffers(
        self, actions: list[int], active: list[bool] | None = None
    ) -> None:
        self._step(actions, active, isolated=False)

    def step_isolated_into_buffers(
        self, actions: list[int], active: list[bool] | None = None
    ) -> None:
        self._step(actions, active, isolated=True)

    def replace_failed_into_buffers(self, active: list[bool]) -> None:
        self.native.replace_failed_into_buffers(active)
        self._synchronize()

    def _step(
        self, actions: list[int], active: list[bool] | None, *, isolated: bool
    ) -> None:
        enabled = active if active is not None else [True] * len(self.games)
        actors = self.native.current_agent_indices()
        frames = self.native.semantic_decision_frames_json(active)
        attempted = []
        for game, action, selected, actor, frame in zip(
            self.games, actions, enabled, actors, frames, strict=True
        ):
            if selected:
                game.actions.append(int(action))
                game.actor = actor
                game.frame = frame
                attempted.append(game)
        with self._guard(lambda: self._evidence_for(attempted, "step")):
            if isolated:
                self.native.step_isolated_into_buffers(actions, active)
            else:
                self.native.step_into_buffers(actions, active)
        self._synchronize()


class CapturedSearch(_Capture):
    """Replayable native search rooted in an actual game's retained prefix.

    Pool slot actions include the sampled policy inputs, so replay needs no model
    or RNG for policy sampling. Native determinization/random tails use the saved
    search seed. This owns evidence only, never retries or labels game outcomes.
    """

    def __init__(
        self,
        root: ScalarCapture,
        env: managym.Env,
        method: Literal["rollout_pool", "flat_mc_scores"],
        arguments: list[int],
    ) -> None:
        super().__init__()
        self.scope = _scope.get() or root.scope
        self.context = root.context.copy()
        self.identities = root.identities.copy()
        self.game = root.inputs.model_copy(deep=True)
        self.game.frame = env.semantic_decision_frame_json()
        self.skip_trivial = root.skip_trivial
        self.search = SearchInputs(method=method, arguments=arguments)
        with self._guard(self.evidence):
            self.native = getattr(env, method)(*arguments)

    def evidence(self) -> FailureEvidence:
        return self._evidence(
            mode="search",
            operation="step",
            games=[self.game],
            search=self.search,
            skip_trivial=self.skip_trivial,
        )

    def __getattr__(self, name: str) -> Any:
        # Native pool counters and observations are read-only PyO3 APIs.
        return getattr(self.native, name)

    def set_buffers(self, buffers: dict[str, Any], capacity: int) -> None:
        self.search.buffers = {
            key: BufferSpec(shape=list(value.shape), dtype=str(value.dtype))
            for key, value in buffers.items()
        }
        self.search.capacity = capacity
        self.native.set_buffers(buffers, capacity)

    def step_active(self, actions: list[int]) -> list[int]:
        self.search.operations.append(list(actions))
        with self._guard(self.evidence):
            return self.native.step_active(actions)

    def finish_random(self) -> None:
        self.search.operations.append("finish_random")
        with self._guard(self.evidence):
            self.native.finish_random()


def read_failure(path: Path) -> FailureEvidence:
    with path.open() as source:
        evidence = FailureEvidence.model_validate_json(next(source))
        for line in source:
            row = json.loads(line)
            if "pending" in row:
                evidence.execution_outcome = "execution_outcome_unknown"
                evidence.games[0].actions.append(row["pending"])
                evidence.games[0].frame = row["frame"]
                evidence.games[0].actor = row["actor"]
                evidence.context = row["context"]
                evidence.seconds = row["seconds"]
            elif "success" in row:
                evidence.execution_outcome = "last_command_succeeded"
            elif "error_class" in row:
                evidence.execution_outcome = "failed"
                evidence.error_class = row["error_class"]
                evidence.error = row["error"]
                evidence.traceback = row["traceback"]
    return evidence
