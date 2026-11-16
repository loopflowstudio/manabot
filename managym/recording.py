"""Private native replay evidence and optional durable pending-call recording.

ScalarCapture and CapturedVector retain exact roots and chosen inputs. Callers
supply destinations and context; no training run or phase is discovered here.
Journals publish intent before native reads, then acknowledge observed returns.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any, Literal, TypedDict, Unpack
from uuid import uuid4
import warnings

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

import managym
from managym.journal import DurableJournal, json_line
from managym.statistics import GameStatistics


class RecordingOptions(TypedDict, total=False):
    journal_path: Path | None
    context: dict[str, object] | None
    context_provider: Callable[[], dict[str, object]]
    failure_directory: Path
    on_retained: Callable[[Path], None] | None


class PlayerSetup(BaseModel):
    name: str
    decklist: dict[str, int]
    sideboard: dict[str, int]
    # Earlier evidence names the selector `content_pack`.
    world: str | None = Field(
        default=None, validation_alias=AliasChoices("world", "content_pack")
    )
    disclosure: Literal["open", "closed"] = Field(
        default="open", exclude_if=lambda value: value == "open"
    )

    @classmethod
    def retain(cls, player: managym.PlayerConfig) -> "PlayerSetup":
        return cls(
            name=player.name,
            decklist=player.decklist,
            sideboard=player.sideboard,
            world=player.world,
            disclosure=player.disclosure,
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


class NativeBoundary(BaseModel):
    """Durable call state; cached games never imply an unobserved reset setup."""

    operation: str
    phase: Literal["preparing", "executing", "synchronizing", "ready"]
    sequence: int
    timestamp: float
    returned: bool = False
    active: list[bool] = []
    pending_actions: list[int | str] = []
    committed_counts: list[int] = []
    setup_known: bool = True
    proposed_seeds: list[int] = []
    proposed_players: list[list[PlayerSetup] | None] = []
    last_known_frames: list[str | None] = []
    frame_prefix_lengths: list[int | None] = []


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
    # Only retained receipts use this field; new vector captures name both seats.
    opponent_policy: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    player_policies: tuple[str, str] | None = None
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
    boundary: NativeBoundary | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def resolved_player_policies(self) -> tuple[str, str]:
        """Read either receipt spelling without deprecating retained evidence."""
        return self.player_policies or _retained_player_policies(self.opponent_policy)


class GameChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    replacement: GameInputs | None = None
    keep: int = Field(default=0, ge=0)
    append: list[int | str] = []
    actor: int | None = None
    frame: str | None = None


class JournalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["training-game-checkpoint-v1"] = Field(
        default="training-game-checkpoint-v1", alias="schema"
    )
    sequence: int = Field(ge=0)
    evidence: FailureEvidence


class JournalChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["training-game-change-v1"] = Field(
        default="training-game-change-v1", alias="schema"
    )
    sequence: int = Field(ge=1)
    operation: Literal["reset", "step"]
    boundary: NativeBoundary
    context_updates: dict[str, object] = {}
    context_removed: list[str] = []
    seconds: float
    identities: dict[str, object] | None = None
    execution_outcome: Literal["execution_outcome_unknown", "last_command_succeeded"]
    games: list[GameChange] = []

    def apply(self, evidence: FailureEvidence) -> None:
        for change in self.games:
            if change.index >= len(evidence.games):
                raise ValueError("journal game index outside checkpoint")
            if change.replacement is not None:
                evidence.games[change.index] = change.replacement
            else:
                game = evidence.games[change.index]
                if change.keep > len(game.actions):
                    raise ValueError("journal prefix outside retained actions")
                del game.actions[change.keep :]
                game.actions.extend(change.append)
                game.actor, game.frame = change.actor, change.frame
        evidence.operation = self.operation
        evidence.boundary = self.boundary
        for key in self.context_removed:
            evidence.context.pop(key, None)
        evidence.context.update(deepcopy(self.context_updates))
        evidence.seconds = self.seconds
        evidence.execution_outcome = self.execution_outcome
        if self.identities is not None:
            evidence.identities = deepcopy(self.identities)


class _GameJournal:
    """One independently copyable checkpoint and sequenced changes per producer.

    Retain a projection for compaction and action-list identity for root changes.
    Phase-only writes never copy or serialize the retained action prefixes.
    """

    def __init__(self, path: Path) -> None:
        self.writer = DurableJournal(path)
        self.evidence: FailureEvidence | None = None
        self.sequence = 0
        self.checkpoint_bytes = 0
        self.appended_bytes = 0
        self.actions: list[list[int | str]] = []

    def publish(
        self,
        evidence: FailureEvidence,
        boundary: NativeBoundary,
        counts: list[int] | None = None,
    ) -> None:
        executing = boundary.phase == "executing"
        outcome: Literal["execution_outcome_unknown", "last_command_succeeded"] = (
            "last_command_succeeded"
            if boundary.returned and evidence.mode == "scalar"
            else "execution_outcome_unknown"
        )
        if self.evidence is None:
            # Reconstruct typed fields without recursively deepcopying prefixes;
            # opaque caller metadata must retain its Python representation.
            retained = FailureEvidence.model_validate(
                {
                    **evidence.model_dump(
                        mode="python", by_alias=True, exclude={"context", "identities"}
                    ),
                    "context": deepcopy(evidence.context),
                    "identities": deepcopy(evidence.identities),
                }
            )
            retained.boundary = boundary.model_copy(deep=True)
            retained.execution_outcome = outcome
            if not executing:
                for game in retained.games:
                    game.frame = None
            row = JournalCheckpoint(sequence=0, evidence=retained)
            self.checkpoint_bytes = self.writer.replace(
                row.model_dump(mode="json", by_alias=True)
            )
            self.evidence = retained
            self.actions = [game.actions for game in evidence.games]
            return
        changes: list[GameChange] = []
        for index, (game, old) in enumerate(
            zip(evidence.games, self.evidence.games, strict=True)
        ):
            count = len(game.actions) if counts is None else counts[index]
            frame = game.frame if executing else None
            if (
                self.actions[index] is not game.actions
                or game.seed != old.seed
                or game.players != old.players
            ):
                replacement = game.model_copy(
                    update={"actions": game.actions[:count], "frame": frame}, deep=False
                )
                changes.append(GameChange(index=index, replacement=replacement))
            else:
                keep = min(count, len(old.actions))
                addition = game.actions[keep:count]
                if (
                    count != len(old.actions)
                    or addition
                    or game.actor != old.actor
                    or frame != old.frame
                ):
                    changes.append(
                        GameChange(
                            index=index,
                            keep=keep,
                            append=addition,
                            actor=game.actor,
                            frame=frame,
                        )
                    )
        change = JournalChange(
            sequence=self.sequence + 1,
            operation=evidence.operation,
            boundary=boundary.model_copy(deep=True),
            context_updates={
                key: value
                for key, value in evidence.context.items()
                if key not in self.evidence.context
                or value != self.evidence.context[key]
            },
            context_removed=[
                key for key in self.evidence.context if key not in evidence.context
            ],
            seconds=evidence.seconds,
            execution_outcome=outcome,
            games=changes,
            identities=evidence.identities.copy()
            if evidence.identities != self.evidence.identities
            else None,
        )
        # The delta must be durable before mutating the compaction projection.
        self.appended_bytes += self.writer.append(
            change.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        change.apply(self.evidence)
        self.sequence = change.sequence
        self.actions = [game.actions for game in evidence.games]
        if self.appended_bytes >= max(self.checkpoint_bytes, 64 * 1024):
            row = JournalCheckpoint(sequence=self.sequence, evidence=self.evidence)
            self.checkpoint_bytes = self.writer.replace(
                row.model_dump(mode="json", by_alias=True)
            )
            self.appended_bytes = 0


@cache
def runtime_identities() -> dict[str, object]:
    """Bind source/native bytes once, when this process first constructs a capture."""
    root = Path(__file__).resolve().parent
    sources = hashlib.sha256()
    paths = [*root.glob("*.py"), *(root / "src").rglob("*.rs"), root / "Cargo.lock"]
    for path in sorted(paths):
        name = str(path.relative_to(root)).encode()
        data = path.read_bytes()
        sources.update(len(name).to_bytes(4, "big"))
        sources.update(name)
        sources.update(len(data).to_bytes(8, "big"))
        sources.update(data)
    extension = Path(managym._managym.__file__)
    return {
        "world": managym.WORLD_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "engine_recording_source_sha256": sources.hexdigest(),
        "engine_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }


def _jsonl(row: dict[str, object]) -> str:
    """One serialization for receipt identities and durable journal writes."""
    return json_line(row).decode()


def _append(path: Path, row: dict[str, object], *, create: bool = False) -> None:
    DurableJournal(path).append(row, create=create)


def atomic_private_json(path: Path, row: dict[str, object]) -> None:
    """Replace a complete private snapshot, including durable directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    DurableJournal(path).replace(row)


class _Capture:
    def __init__(
        self,
        *,
        journal_path: Path | None = None,
        context: dict[str, object] | None = None,
        context_provider: Callable[[], dict[str, object]] = dict,
        failure_directory: Path = Path(".runs/game-failures"),
        on_retained: Callable[[Path], None] | None = None,
    ) -> None:
        self.started = time.perf_counter()
        self._failed = False
        self.identities = runtime_identities().copy()
        self.context = dict(context or {})
        self.context_provider = context_provider
        self.failure_directory = failure_directory
        self.on_retained = on_retained
        self._boundary: NativeBoundary | None = None
        self._call_evidence: FailureEvidence | None = None
        self._game_journal = _GameJournal(journal_path) if journal_path else None
        self.frozen_context: dict[str, object] | None = None

    def _start_cursor(self, name: str) -> None:
        if self._game_journal is not None:
            cursor = self._game_journal.writer.path
            cursor.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(cursor.parent, 0o700)
            for path in cursor.parent.glob(f".{cursor.name}.*.tmp"):
                path.unlink(missing_ok=True)

    def record_policy_actions(
        self, count: int, kind: str, capacity: int, game: GameInputs
    ) -> None:
        """Consumer hook; policy cutoff accounting is not engine recording."""

    def flush_policy_actions(self) -> None:
        """Consumer hook for optional policy accounting at failure/close."""

    def _publish(
        self,
        evidence: FailureEvidence,
        phase: Literal["preparing", "executing", "synchronizing", "ready"],
        counts: list[int] | None = None,
    ) -> None:
        if self._game_journal is None:
            return
        assert self._boundary is not None
        self._boundary.phase = phase
        self._boundary.timestamp = time.time()
        if phase == "executing":
            self._boundary.last_known_frames = [game.frame for game in evidence.games]
            self._boundary.frame_prefix_lengths = [
                len(game.actions) - 1 if game.frame is not None else None
                for game in evidence.games
            ]
        self._game_journal.publish(evidence, self._boundary, counts)

    def _begin_boundary(
        self,
        evidence: FailureEvidence,
        operation: str,
        *,
        actions: list[int | str] | None = None,
        active: list[bool] | None = None,
        setup_known: bool = True,
        seeds: list[int] | None = None,
        players: list[list[PlayerSetup] | None] | None = None,
    ) -> None:
        if self._game_journal is None:
            return
        # Freeze fields, retaining action-list references and their pre-call lengths.
        # Actions append in place; root resets replace the list.
        self._call_evidence = evidence.model_copy(
            update={"games": [game.model_copy() for game in evidence.games]}
        )
        self._boundary = NativeBoundary(
            operation=operation,
            phase="preparing",
            timestamp=time.time(),
            sequence=1 if self._boundary is None else self._boundary.sequence + 1,
            pending_actions=actions or [],
            active=active or [],
            committed_counts=[len(game.actions) for game in evidence.games],
            setup_known=setup_known,
            proposed_seeds=seeds or [],
            proposed_players=players or [],
            last_known_frames=[game.frame for game in evidence.games],
            frame_prefix_lengths=[
                len(game.actions) - 1 if game.frame is not None else None
                for game in evidence.games
            ],
        )
        self._publish(evidence, "preparing")

    def _executing(self) -> None:
        """Publish a reset mutation using its cached, previously observed inputs."""
        if self._game_journal is not None:
            assert self._boundary is not None and self._call_evidence is not None
            self._publish(
                self._call_evidence, "executing", self._boundary.committed_counts
            )

    def _returned(self) -> None:
        if self._game_journal is not None:
            assert self._boundary is not None and self._call_evidence is not None
            self._boundary.returned = True
            try:
                self._publish(
                    self._call_evidence,
                    "synchronizing",
                    self._boundary.committed_counts,
                )
            except OSError as error:
                error.add_note(
                    "Native operation returned; its durable journal acknowledgment failed"
                )
                raise

    def _retain(self, evidence: FailureEvidence) -> Path:
        """Create and register private evidence only after its bytes are durable."""
        directory = self.failure_directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{uuid4().hex}.jsonl"
        _append(path, evidence.model_dump(mode="json", by_alias=True), create=True)
        if self.on_retained is not None:
            self.on_retained(path)
        return path

    def _coordinates(self) -> dict[str, object]:
        """Read live owner coordinates equally for failures and pending Commands."""
        context = self.frozen_context
        if context is None:
            context = self.context_provider()
        return {
            "context": {**context, **self.context},
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
        self._failed = True
        # Capture failures must never disguise the original native failure.
        try:
            self.flush_policy_actions()
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
    def __init__(
        self,
        seed: int,
        skip_trivial: bool = True,
        **recording: Unpack[RecordingOptions],
    ) -> None:
        super().__init__(**recording)
        self.seed = seed
        self.skip_trivial = skip_trivial
        self.inputs = GameInputs(seed=seed, players=[])
        self.journal: Path | None = None
        self._start_cursor("scalar")

    def reset(self, env: managym.Env, players: list[managym.PlayerConfig]) -> Any:
        # The heterogeneous native observation/info tuple stays at this boundary.
        self.started = time.perf_counter()
        if self._game_journal is not None:
            self._begin_boundary(
                self.evidence(),
                "reset",
                seeds=[self.seed],
                players=[[PlayerSetup.retain(p) for p in players]],
            )
        self.inputs = GameInputs(
            seed=self.seed, players=[PlayerSetup.retain(p) for p in players]
        )
        self.journal = None
        with self._guard(lambda: self.evidence("reset")):
            self._executing()
            result = env.reset(players)
            self._returned()
            self.identities["content_manifest"] = env.content_pack_manifest()
            if self._game_journal is not None:
                self._publish(self.evidence("reset"), "ready")
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
        if self._game_journal is not None:
            self._begin_boundary(
                self.evidence(), "step", actions=[int(action)], active=[True]
            )
        self.inputs.actions.append(int(action))
        self.inputs.actor = env.current_agent_index()
        self.inputs.frame = env.semantic_decision_frame_json()
        with self.guard():
            if self._game_journal is not None:
                self._publish(self.evidence(), "executing")
            result = env.step(action)
            self._returned()
            if self._game_journal is not None:
                self._publish(self.evidence(), "ready")
            return result

    def command(
        self, env: managym.Env, command: str, *, transition_only: bool = True
    ) -> Any:
        """Fsync exact pending Command and full offer frame before native execution."""
        if self._game_journal is not None:
            self._begin_boundary(
                self.evidence(), "command", actions=[command], active=[True]
            )
            self.inputs.actor = env.current_agent_index()
            self.inputs.frame = env.semantic_decision_frame_json()
            self.inputs.actions.append(command)
            with self.guard():
                self._publish(self.evidence(), "executing")
                result = (
                    env.execute_semantic_command_json(command)
                    if transition_only
                    else env.step_semantic_command(command)
                )
                self._returned()
                self._publish(self.evidence(), "ready")
                return result
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


# Native vector reads CapturedVector delegates; everything else it owns or rejects.
_VECTOR_READS = frozenset(
    {
        "buffered_call_timings",
        "current_action_summaries",
        "current_agent_indices",
        "current_exposure",
        "current_game_generations",
        "current_game_seeds",
        "current_matchup_indices",
        "current_player_configs",
        "get_last_info",
        "semantic_decision_frames_json",
        "skip_trivial_counts",
    }
)


def _retained_player_policies(opponent_policy: str | None) -> tuple[str, str]:
    """Translate retained names; native construction validates the resulting seats."""
    return (
        "agent",
        "agent" if opponent_policy in (None, "none") else opponent_policy,
    )


def resolve_player_policies(
    opponent_policy: str | None, player_policies: tuple[str, str] | None
) -> tuple[str, str]:
    """Per-seat chooser policies from either spelling of the argument.

    The legacy ``opponent_policy`` warns (removal in 0.3.0): ``none`` surfaces
    both seats and any other name plays seat 1 natively. Native construction
    rejects unknown per-seat names.
    """
    if opponent_policy is None:
        return player_policies or ("agent", "agent")
    if opponent_policy not in ("none", "random", "heuristic"):
        raise ValueError(f"unsupported opponent_policy: {opponent_policy}")
    if player_policies is not None:
        raise ValueError("use player_policies or opponent_policy, not both")
    warnings.warn(
        "opponent_policy is deprecated; use player_policies; removal in 0.3.0",
        DeprecationWarning,
        stacklevel=3,
    )
    return _retained_player_policies(opponent_policy)


class CapturedVector(_Capture):
    """Instrument the strict native vector API without isolating or retrying rows.

    On batch failure every active row retains its pre-call prefix. Native batch
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
        opponent_policy: str | None = None,
        player_policies: tuple[str, str] | None = None,
        seed_stride: int | None = None,
        stream_offset: int = 0,
        **recording: Unpack[RecordingOptions],
    ) -> None:
        super().__init__(**recording)
        self.stride = seed_stride or num_envs
        self.skip_trivial = skip_trivial
        self.player_policies = resolve_player_policies(opponent_policy, player_policies)
        self.native = managym.VectorEnv(
            num_envs=num_envs,
            seed=seed,
            skip_trivial=skip_trivial,
            player_policies=self.player_policies,
            seed_stride=self.stride,
        )
        self._closed = False
        self.games = [
            GameInputs(
                stream=stream_offset + i, seed=min(seed + i, 2**64 - 1), players=[]
            )
            for i in range(num_envs)
        ]
        self.next_seeds = [game.seed for game in self.games]
        self._generations = self.native.current_game_generations()
        # Sampled roots pick setups natively at reset; evidence reads them back.
        self._sampled_roots = False
        self._action_cutoff: int | None = None
        self._start_cursor(f"vector-{stream_offset}")

    def __getattr__(self, name: str) -> Any:
        # Only reads delegate: an undelegated mutation would leave the retained
        # replay prefixes behind the native games.
        if name in _VECTOR_READS:
            return getattr(self.__dict__["native"], name)
        raise AttributeError(
            f"CapturedVector does not expose {name}; use its read methods or "
            "reset_*_into_buffers / step_*_into_buffers calls so replay "
            "evidence stays aligned with the games"
        )

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
    ) -> list[managym.ObservationRead]:
        """Write viewer-safe NumPy rows; discard every output row after an error.

        Execution and capture buffers stay unchanged. Seats are native indices.
        """
        return self.native.write_observations_for_players(
            players, buffers, action_overflow=action_overflow
        )

    def hand_snapshots(self, players: list[int]) -> list[managym.HandSnapshot]:
        """Read privileged current hands without adding them to failure evidence."""
        return self.native.hand_snapshots(players)

    def set_buffers(
        self, buffers: dict[str, Any], action_overflow: str = "error"
    ) -> None:
        # NumPy owns these buffers; preserve their native encoding ABI as evidence.
        self.identities["input_buffers"] = {
            key: {"shape": list(value.shape[1:]), "dtype": str(value.dtype)}
            for key, value in buffers.items()
        }
        self.native.set_buffers(buffers, action_overflow)
        self._action_cutoff = (
            buffers["actions"].shape[1] if action_overflow == "first-legal" else None
        )
        if self._action_cutoff is not None:
            self.identities["action_overflow"] = action_overflow

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._retain_exposure(current=not self._failed)
        finally:
            self.flush_policy_actions()

    def _retain_exposure(self, *, current: bool = False) -> None:
        """Consumer hook at synchronized boundaries and close."""

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
            player_policies=self.player_policies,
            seed_stride=self.stride,
        )

    def set_uniform_roster(self, roster: list[managym.PlayerConfig]) -> None:
        self.native.set_uniform_roster(roster)
        self._sampled_roots = True

    def set_training_decks(
        self,
        decks: list[managym.PlayerConfig],
        digest: str,
        sampling: Literal[
            "uniform-distinct-v1", "uniform-independent-v1"
        ] = "uniform-distinct-v1",
    ) -> None:
        """Replace decks for later roots; evidence keeps each game's actual setup."""
        self.native.set_training_decks(
            decks, distinct=sampling == "uniform-distinct-v1"
        )
        self._sampled_roots = True
        self.identities["deck_source"] = {
            "digest": digest,
            "sampling_policy": sampling,
            "deck_ids": [player.name for player in decks],
        }

    def reset_all_into_buffers(self, players: list[managym.PlayerConfig]) -> None:
        self.reset_each_into_buffers([players for _ in self.games])

    def reset_each_into_buffers(
        self, players: list[list[managym.PlayerConfig]]
    ) -> None:
        self._reset_each(players, lambda: self.native.reset_each_into_buffers(players))

    def reset_all(self, players: list[managym.PlayerConfig]) -> Any:
        return self._reset_each(
            [players for _ in self.games], lambda: self.native.reset_all(players)
        )

    def _reset_each(
        self, players: list[list[managym.PlayerConfig]], call: Callable[[], Any]
    ) -> Any:
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reset_each",
                setup_known=not self._sampled_roots,
                seeds=self.next_seeds,
                players=[[PlayerSetup.retain(p) for p in setup] for setup in players],
            )
        for game, setup, seed in zip(self.games, players, self.next_seeds, strict=True):
            game.players = [PlayerSetup.retain(p) for p in setup]
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self._executing()
            result = call()
        self._returned()
        self._synchronize()
        return result

    def reassign_roots_into_buffers(
        self, players: list[list[managym.PlayerConfig] | None]
    ) -> None:
        """Retain new setups only after native atomic fresh-root validation."""
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reassign_roots",
                players=[
                    None if setup is None else [PlayerSetup.retain(p) for p in setup]
                    for setup in players
                ],
            )
        proposed = [game.model_copy(deep=True) for game in self.games]
        for game, setup, seed in zip(
            proposed, players, self.native.current_game_seeds(), strict=True
        ):
            if setup is not None:
                game.players = [PlayerSetup.retain(p) for p in setup]
                game.restart(seed)
        with self._guard(lambda: self._evidence_for(proposed, "reset")):
            self._executing()
            self.native.reassign_roots_into_buffers(players)
        self._returned()
        self.games = proposed
        self._synchronize()

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reset_seeds",
                setup_known=not self._sampled_roots,
                seeds=seeds,
            )
        self.next_seeds = list(seeds)
        for game, seed in zip(self.games, seeds, strict=True):
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self._executing()
            self.native.reset_seeds_into_buffers(seeds)
        self._returned()
        self._synchronize()

    def _synchronize(self) -> None:
        self._retain_exposure()
        seeds = self.native.current_game_seeds()
        # Generations, not seeds, mark a new game: saturated seed succession and
        # same-seed resets repeat a seed. They are process-local, never evidence.
        generations = self.native.current_game_generations()
        restarted = [
            index
            for index, (old, new) in enumerate(
                zip(self._generations, generations, strict=True)
            )
            if old != new
        ]
        self._generations = generations
        if self._sampled_roots and restarted:
            # Sampled roots record the native setup actually created.
            players = self.native.current_player_configs()
            for index in restarted:
                self.games[index].players = [
                    PlayerSetup.retain(player) for player in players[index]
                ]
        for index in restarted:
            self.games[index].restart(seeds[index])
        self.next_seeds = [
            min(game.seed + self.stride, 2**64 - 1) for game in self.games
        ]
        if self._boundary is not None:
            self._boundary.setup_known = True
        if self._game_journal is not None:
            self._publish(self._evidence_for(self.games, "step"), "ready")

    def enable_statistics(self) -> None:
        self.native.enable_statistics()

    def statistics(self) -> list[GameStatistics | None]:
        return [
            None
            if row is None
            else row.model_copy(update={"stream": self.games[i].stream})
            for i, row in enumerate(self.native.statistics())
        ]

    def take_statistics(self) -> list[GameStatistics]:
        return [
            row.model_copy(update={"stream": self.games[row.stream].stream})
            for row in self.native.take_statistics()
        ]

    def step(self, actions: list[int]) -> Any:
        return self._step(actions, None, isolated=False, buffered=False)

    def step_into_buffers(
        self, actions: list[int], active: list[bool] | None = None
    ) -> None:
        self._step(actions, active, isolated=False)

    def step_isolated_into_buffers(
        self,
        actions: list[int],
        active: list[bool] | None = None,
        recover_capacity: bool = False,
    ) -> None:
        self._step(actions, active, isolated=True, recover_capacity=recover_capacity)

    def replace_failed_into_buffers(self, active: list[bool]) -> None:
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "replace_failed",
                active=active,
                setup_known=not self._sampled_roots,
            )
        self._executing()
        self.native.replace_failed_into_buffers(active)
        self._returned()
        self._synchronize()

    def _step(
        self,
        actions: list[int],
        active: list[bool] | None,
        *,
        isolated: bool,
        recover_capacity: bool = False,
        buffered: bool = True,
    ) -> Any:
        enabled = active if active is not None else [True] * len(self.games)
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "step"),
                "step_isolated" if isolated else "step",
                actions=[int(action) for action in actions],
                active=enabled,
                setup_known=not self._sampled_roots,
            )
        actors = self.native.current_agent_indices()
        frames = self.native.semantic_decision_frames_json(active)
        summaries = (
            self.native.current_action_summaries() if self._action_cutoff else []
        )
        attempted = []
        for index, (game, action, selected, actor, frame) in enumerate(
            zip(self.games, actions, enabled, actors, frames, strict=True)
        ):
            if selected:
                game.actor = actor
                game.frame = frame
                if self._action_cutoff is not None:
                    count, kind = summaries[index]
                    self.record_policy_actions(count, kind, self._action_cutoff, game)
                game.actions.append(int(action))
                attempted.append(game)
        with self._guard(lambda: self._evidence_for(attempted, "step")):
            if self._game_journal is not None:
                self._publish(self._evidence_for(self.games, "step"), "executing")
            result = None
            if not buffered:
                result = self.native.step(actions)
            elif isolated:
                self.native.step_isolated_into_buffers(
                    actions, enabled, recover_capacity
                )
            else:
                self.native.step_into_buffers(actions, active)
        self._returned()
        self._synchronize()
        return result


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
        **recording: Unpack[RecordingOptions],
    ) -> None:
        recording.setdefault("context_provider", root.context_provider)
        recording.setdefault("failure_directory", root.failure_directory)
        recording.setdefault("on_retained", root.on_retained)
        super().__init__(**recording)
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
    with path.open("rb") as source:
        first = next(source, b"")
        row = json.loads(first)
        if row.get("schema") == "training-game-checkpoint-v1":
            if not first.endswith(b"\n"):
                raise ValueError("incomplete initial training checkpoint")
            checkpoint = JournalCheckpoint.model_validate(row)
            evidence = checkpoint.evidence
            sequence = checkpoint.sequence
            for line in source:
                if not line.endswith(b"\n"):
                    warnings.warn(
                        "Incomplete final training journal tail; projected last complete boundary",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
                change = JournalChange.model_validate_json(line)
                if change.sequence != sequence + 1:
                    raise ValueError("training journal sequence gap")
                change.apply(evidence)
                sequence = change.sequence
            return evidence
        evidence = FailureEvidence.model_validate(row)
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
