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
from importlib.metadata import version
from itertools import chain
import json
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any, BinaryIO, Literal, TypedDict, Unpack
from uuid import uuid4
import warnings

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

import managym
from managym.bots import BeliefSummary
from managym.card_capture import CapturedCards, captured_cards
from managym.decision import Command, DecisionFrame
from managym.disclosure import DecklistDisclosure
from managym.journal import DurableJournal, json_line
from managym.statistics import GameStatistics, Reducer, Reduction


def semantic_frame_sha256(frame: str) -> str:
    """Hash legal-frame JSON independently of whitespace and object key order."""
    body = json.dumps(json.loads(frame), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


class RecordingOptions(TypedDict, total=False):
    games_path: Path | None
    statistics_path: Path | None
    record_fraction: float
    journal_path: Path | None
    context: dict[str, object] | None
    context_provider: Callable[[], dict[str, object]]
    failure_directory: Path
    on_retained: Callable[[Path], None] | None


class CaptureCoordinates(TypedDict):
    context: dict[str, object]
    seconds: float


class PlayerSetup(BaseModel):
    name: str
    decklist: dict[str, int]
    sideboard: dict[str, int]
    mana_taps: managym.ManaTaps = Field(
        default="off", exclude_if=lambda value: value == "off"
    )
    # Earlier evidence names the selector `content_pack`.
    world: str | None = Field(
        default=None, validation_alias=AliasChoices("world", "content_pack")
    )
    disclosure: DecklistDisclosure = Field(
        default="open", exclude_if=lambda value: value == "open"
    )

    @classmethod
    def retain(
        cls,
        player: managym.PlayerConfig,
        decklists: DecklistDisclosure,
        mana_taps: managym.ManaTaps = "off",
    ) -> "PlayerSetup":
        return cls(
            name=player.name,
            decklist=player.decklist,
            sideboard=player.sideboard,
            world=player.world,
            disclosure=decklists,
            mana_taps=mana_taps,
        )


def _game_disclosure(players: list[PlayerSetup]) -> DecklistDisclosure | None:
    """The one game value retained in per-seat spelling; None without seats."""
    values = {player.disclosure for player in players}
    if len(values) > 1:
        raise ValueError("retained per-seat disclosure values disagree")
    return values.pop() if values else None


def _retain_players(players: managym.GameSetup, seed: int) -> list[PlayerSetup]:
    disclosure = managym.resolve_decklists(players, seed)
    setup = managym.GameSetup(players)
    return [
        PlayerSetup.retain(player, disclosure, taps)
        for player, taps in zip(setup, setup.mana_taps)
    ]


class DecisionAnnotation(BaseModel):
    """Optional runner evidence in the legal offer order, never recomputed."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    probabilities: list[float] | None = None
    value: float | None = None
    update: int | None = None
    search_scores: list[float | None] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    belief: BeliefSummary | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    extra: dict[str, object] = {}


class GameEnd(BaseModel):
    status: Literal["complete", "interrupted", "failed"]
    winner: int | None = None
    winner_retained: bool = True
    reason: str | None = None
    digest: str | None = None


class GameInputs(BaseModel):
    stream: int = 0
    seed: int
    players: list[PlayerSetup]
    actions: list[int | str] = []
    committed: int | None = Field(default=None, ge=0)
    digests: list[str] = []
    annotations: list[DecisionAnnotation | None] = []
    command_frame_hashes: list[str | None] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    end: GameEnd | None = None
    actor: int | None = None
    frame: str | None = None

    @model_validator(mode="after")
    def uniform_disclosure(self) -> "GameInputs":
        _game_disclosure(self.players)
        return self

    def setup(self) -> managym.GameSetup:
        """Rebuild retained seats with one game value, rejecting disagreement."""
        decklists = _game_disclosure(self.players)
        if decklists is None:
            raise ValueError("retained game has no player setups")
        return managym.GameSetup(
            [
                managym.PlayerConfig(
                    **player.model_dump(exclude={"disclosure", "mana_taps"})
                )
                for player in self.players
            ],
            decklists=decklists,
            mana_taps=(
                self.players[0].mana_taps,
                self.players[1].mana_taps,
            ),
        )

    def restart(self, seed: int) -> None:
        """Keep the stream's setup, discard all inputs from its previous game."""
        self.seed = seed
        self.actions = []
        self.committed = 0
        self.digests = []
        self.annotations = []
        self.command_frame_hashes = []
        self.end = None
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


class GameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["managym-game-v1", "native-failure-v1"] = Field(
        default="managym-game-v1", alias="schema"
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
    def needs_reproduction(self) -> bool:
        """A failed, pending or search call needs exact-runtime reproduction.

        Acknowledged prefixes are viewed with the walker on the same world;
        reproduction may attempt the unacknowledged call, including its failure.
        """
        return (
            self.error is not None
            or self.error_class is not None
            or self.execution_outcome == "failed"
            or self.mode == "search"
            or (self.boundary is not None and self.boundary.phase != "ready")
            or (
                self.boundary is None
                and self.execution_outcome != "last_command_succeeded"
                and any(
                    game.frame is not None
                    and (game.committed is None or game.committed < len(game.actions))
                    for game in self.games
                )
            )
        )

    @property
    def sha256(self) -> str:
        """Identity of the canonical private row, including its ending and annotations."""
        return hashlib.sha256(
            json_line(self.model_dump(mode="json", by_alias=True))
        ).hexdigest()

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
    committed: int | None = Field(default=None, ge=0)
    digests: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    annotations: list[DecisionAnnotation | None] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    command_frame_hashes: list[str | None] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    end: GameEnd | None = None
    actor: int | None = None
    frame: str | None = None


class JournalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["training-game-checkpoint-v1"] = Field(
        default="training-game-checkpoint-v1", alias="schema"
    )
    sequence: int = Field(ge=0)
    evidence: GameRecord


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

    def apply(self, evidence: GameRecord) -> None:
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
                game.digests = game.digests[: change.keep] + change.digests
                game.annotations = game.annotations[: change.keep] + change.annotations
                if game.command_frame_hashes or change.command_frame_hashes:
                    frames = game.command_frame_hashes[: change.keep]
                    game.command_frame_hashes = (
                        frames
                        + [None] * (change.keep - len(frames))
                        + change.command_frame_hashes
                    )
                game.end = change.end
                game.committed = change.committed
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
        self.evidence: GameRecord | None = None
        self.sequence = 0
        self.checkpoint_bytes = 0
        self.appended_bytes = 0
        self.actions: list[list[int | str]] = []

    def publish(
        self,
        evidence: GameRecord,
        boundary: NativeBoundary,
        counts: list[int] | None = None,
    ) -> None:
        executing = boundary.phase == "executing"
        outcome: Literal["execution_outcome_unknown", "last_command_succeeded"] = (
            "last_command_succeeded"
            if boundary.returned and evidence.mode == "scalar"
            else "execution_outcome_unknown"
        )
        if (
            self.evidence is None
            or evidence.error_class is not None
            or self.evidence.error_class is not None
        ):
            # Reconstruct typed fields without recursively deepcopying prefixes;
            # opaque caller metadata must retain its Python representation.
            retained = GameRecord.model_validate(
                {
                    **evidence.model_dump(
                        mode="python", by_alias=True, exclude={"context", "identities"}
                    ),
                    "context": deepcopy(evidence.context),
                    "identities": deepcopy(evidence.identities),
                }
            )
            retained.boundary = boundary.model_copy(deep=True)
            retained.execution_outcome = (
                "failed" if evidence.error_class is not None else outcome
            )
            if not executing:
                for game in retained.games:
                    game.frame = None
            self.sequence = 0
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
                    or game.end != old.end
                    or game.committed != old.committed
                ):
                    changes.append(
                        GameChange(
                            index=index,
                            keep=keep,
                            append=addition,
                            committed=game.committed,
                            digests=game.digests[keep:count],
                            annotations=game.annotations[keep:count],
                            command_frame_hashes=game.command_frame_hashes[keep:count],
                            end=game.end,
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
        "managym_version": version("managym"),
        "card_digest": managym.world_identity()["card_digest"],
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
        games_path: Path | None = None,
        statistics_path: Path | None = None,
        record_fraction: float = 1.0,
        journal_path: Path | None = None,
        context: dict[str, object] | None = None,
        context_provider: Callable[[], dict[str, object]] = dict,
        failure_directory: Path = Path(".runs/game-failures"),
        on_retained: Callable[[Path], None] | None = None,
    ) -> None:
        if not 0 <= record_fraction <= 1:
            raise ValueError("record_fraction must be in [0, 1]")
        self.producer = "capture"
        self.games_path = games_path
        self.record_fraction = record_fraction
        self.statistics_path = statistics_path
        self._outputs: dict[Path, BinaryIO] = {}
        self._statistics_generations: dict[int, int] = {}
        self.started = time.perf_counter()
        self._failed = False
        self._raised: GameRecord | None = None
        self.identities = runtime_identities().copy()
        self.context = dict(context or {})
        self.context_provider = context_provider
        self.failure_directory = failure_directory
        self.on_retained = on_retained
        self._boundary: NativeBoundary | None = None
        self._call_evidence: GameRecord | None = None
        self._game_journal = _GameJournal(journal_path) if journal_path else None
        self._automatic_journal = False
        self._journal_retained = False
        self.frozen_context: dict[str, object] | None = None
        self._pending_journal_path: Path | None = None
        self._profiler_requested = False

    def _snapshot(self) -> GameRecord:
        raise NotImplementedError

    def evidence(self) -> GameRecord:
        """Deprecated spelling of snapshot, removed in 0.4.0."""
        warnings.warn(
            "evidence() is deprecated; use snapshot(); removal in 0.4.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.snapshot()

    def snapshot(self) -> GameRecord:
        """Copy cached cursors only; never enter the native library from a watchdog."""
        evidence = self._snapshot()
        return evidence.model_copy(update={"boundary": self._boundary}, deep=True)

    def enable_journal(self, path: Path) -> None:
        # A watchdog can run during a GIL-free native call. Do not change its
        # in-flight journal protocol: activate at the next owner-thread boundary.
        self._profiler_requested = True
        if self._game_journal is None:
            self._pending_journal_path = path

    def _activate_journal(self, native: Any) -> None:
        if self._pending_journal_path is not None:
            self._game_journal = _GameJournal(self._pending_journal_path)
            self._pending_journal_path = None
        if self._profiler_requested:
            native.enable_profiler()
            self._profiler_requested = False

    @property
    def _recording(self) -> bool:
        return self.record_fraction > 0 and (
            self.games_path is not None
            or (self._game_journal is not None and not self._automatic_journal)
        )

    def _selected(self, game: GameInputs) -> bool:
        if not self._recording:
            return False
        # A journal enabled mid-game cannot invent earlier state digests. Keep
        # that root as diagnostic inputs; record from the next complete root.
        if game.actions and not game.digests:
            return False
        value = hashlib.sha256(f"{game.seed}:{game.stream}".encode()).digest()
        return int.from_bytes(value[:8], "big") < self.record_fraction * 2**64

    def _write_game(self, game: GameInputs, end: GameEnd) -> None:
        if not self._selected(game) or game.end is not None or not game.players:
            return
        game.end = end
        record = self._snapshot()
        record = record.model_copy(update={"games": [game], "boundary": None})
        if self.games_path is not None:
            self._append_row(
                self.games_path, record.model_dump(mode="json", by_alias=True)
            )

    def _append_row(self, path: Path, row: object) -> None:
        output = self._outputs.get(path)
        if output is None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.fchmod(descriptor, 0o600)
            output = self._outputs[path] = os.fdopen(descriptor, "ab")
        output.write(json_line(row))
        output.flush()

    def _write_statistics(self, rows: list[GameStatistics]) -> None:
        if self.statistics_path is None:
            return
        for row in rows:
            # Native drains omit the running generation. Close and root
            # replacement also read it; remember the latest generation per
            # stream so a repeated close writes no second row.
            if self._statistics_generations.get(row.stream) == row.generation:
                continue
            self._append_row(
                self.statistics_path,
                {
                    **row.model_dump(mode="json"),
                    "card_digest": self.identities["card_digest"],
                    "context": self._coordinates()["context"],
                },
            )
            self._statistics_generations[row.stream] = row.generation

    def _flush_statistics(self, *, partial: bool = False) -> None:
        pass

    def _close_output(self) -> None:
        try:
            for output in self._outputs.values():
                output.flush()
                os.fsync(output.fileno())
        finally:
            for output in self._outputs.values():
                output.close()
            self._outputs.clear()

    def _start_cursor(self, name: str) -> None:
        self.producer = name
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
        evidence: GameRecord,
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
        evidence: GameRecord,
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
        self._raised = None
        self._failed = False
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

    def _retain(self, evidence: GameRecord) -> Path:
        """Create and register private evidence only after its bytes are durable."""
        directory = self.failure_directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{uuid4().hex}.jsonl"
        _append(path, evidence.model_dump(mode="json", by_alias=True), create=True)
        if self.on_retained is not None:
            self.on_retained(path)
        return path

    def _coordinates(self) -> CaptureCoordinates:
        """Read live owner coordinates equally for failures and pending Commands."""
        context = self.frozen_context
        if context is None:
            context = self.context_provider()
        return {
            "context": {**context, **self.context},
            "seconds": time.perf_counter() - self.started,
        }

    def _evidence(self, **fields: object) -> GameRecord:
        return GameRecord.model_validate(
            {
                **fields,
                **(
                    {
                        "error_class": self._raised.error_class,
                        "error": self._raised.error,
                        "traceback": self._raised.traceback,
                        "execution_outcome": "failed",
                    }
                    if self._raised is not None
                    else {"execution_outcome": "last_command_succeeded"}
                    if self._boundary is not None and self._boundary.returned
                    else {}
                ),
                "identities": self.identities,
                **self._coordinates(),
            }
        )

    @contextmanager
    def _guard(self, evidence: Callable[[], GameRecord]) -> Iterator[None]:
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
        self, error: BaseException, evidence: Callable[[], GameRecord]
    ) -> None:
        self._failed = True
        # Capture failures must never disguise the original native failure.
        try:
            self.flush_policy_actions()
            # No statistics row here: the game may still finish after a raised
            # failure. Only close or root replacement writes a partial row.
            retained = evidence()
            retained.execution_outcome = "failed"
            retained.error_class = type(error).__name__
            retained.error = str(error)
            retained.traceback = "".join(traceback.format_exception(error))
            self._raised = retained
            if self._automatic_journal:
                assert self._game_journal is not None and self._boundary is not None
                self._publish(retained, self._boundary.phase)
                path = self._game_journal.writer.path
                # A failure is immutable evidence, no longer a live cursor that
                # a later reset or close may compact over.
                self._game_journal = None
                self._automatic_journal = False
                self._journal_retained = False
            else:
                path = self._retain(retained)
                if self._game_journal is not None and self._boundary is not None:
                    self._publish(retained, self._boundary.phase)
            error.add_note(f"Private native failure evidence: {path}")
        except Exception as capture_error:
            error.add_note(f"Failure evidence could not be retained: {capture_error}")


class ScalarCapture(_Capture):
    def __init__(
        self,
        seed: int,
        skip_trivial: bool = True,
        *,
        reducers: list[Reducer[Any]] | None = None,
        **recording: Unpack[RecordingOptions],
    ) -> None:
        super().__init__(**recording)
        self.reducers = reducers or []
        self._reduction = Reduction(self.reducers)
        self._reducers_finished = True
        self._env: managym.Env | None = None
        self.seed = seed
        self.skip_trivial = skip_trivial
        self.inputs = GameInputs(seed=seed, players=[])
        self._start_cursor("scalar")

    def reset(self, env: managym.Env, players: managym.GameSetup) -> Any:
        # The heterogeneous native observation/info tuple stays at this boundary.
        if self.statistics_path is not None and self._env is not env:
            self._flush_statistics(partial=True)
            env.enable_statistics()
        if self.reducers:
            if self._env is None:
                env.enable_exposure_capture()
            else:
                self._finish_reducers(
                    env, GameEnd(status="interrupted", reason="reset")
                )
        self.started = time.perf_counter()
        self._activate_journal(env)
        if self._game_journal is not None:
            self._begin_boundary(
                self._snapshot(),
                "reset",
                seeds=[self.seed],
                players=[_retain_players(players, self.seed)],
            )
        self._write_game(self.inputs, GameEnd(status="interrupted", reason="reset"))
        self.inputs = GameInputs(
            seed=self.seed,
            committed=0,
            players=_retain_players(players, self.seed),
        )
        with self._guard(lambda: self._snapshot("reset")):
            self._executing()
            result = env.reset(players)
            self._env = env
            self._flush_statistics()
            self._reducers_finished = False
            self._reduction.start(self.inputs.players)
            self._returned()
            self.identities["content_manifest"] = env.content_pack_manifest()
            if self._game_journal is not None:
                self._publish(self._snapshot("reset"), "ready")
            return result

    def _snapshot(self, operation: Literal["reset", "step"] = "step") -> GameRecord:
        return self._evidence(
            mode="scalar",
            operation=operation,
            games=[self.inputs],
            skip_trivial=self.skip_trivial,
        )

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Retain policy/cap/deadline failures outside the native call too."""
        with self._guard(self._snapshot):
            yield

    def rows(self) -> list[list[Any]]:
        """Reducer tables for the games completed or closed so far."""
        return self._reduction.tables()

    @property
    def analysis_error(self) -> str | None:
        return self._reduction.error

    def _finish_reducers(self, env: managym.Env, end: GameEnd) -> None:
        if self._reducers_finished or not self.reducers:
            return
        from managym.replay import ReplayStop

        try:
            captured = captured_cards(env)
        except Exception as exc:
            self._reduction.fail(f"card facts: {type(exc).__name__}: {exc}")
            self._reducers_finished = True
            return
        result = ReplayStop(
            "complete" if end.status == "complete" else "interrupted",
            len(self.inputs.actions),
            end.reason,
            end.winner,
        )
        self._reduction.end(result, captured)
        self._reducers_finished = True

    def _flush_statistics(self, *, partial: bool = False) -> None:
        if self.statistics_path is not None and self._env is not None:
            rows = self._env.take_statistics()
            if partial and (current := self._env.statistics()) is not None:
                rows.append(current)
            self._write_statistics(rows)

    def close(self, *, reason: str = "capture closed") -> None:
        if self._env is not None:
            self._finish_reducers(
                self._env, GameEnd(status="interrupted", reason=reason)
            )
        self._write_game(
            self.inputs,
            GameEnd(
                status="failed" if self._failed else "interrupted",
                reason=reason,
                digest=(
                    self._env.state_digest()
                    if self._env is not None
                    and not self._failed
                    and self.inputs.end is None
                    and self._selected(self.inputs)
                    else None
                ),
            ),
        )
        if self._game_journal is not None and self._boundary is not None:
            self._publish(self._snapshot(), self._boundary.phase)
        self._flush_statistics(partial=True)
        self._close_output()

    def step(
        self,
        env: managym.Env,
        action: int,
        *,
        annotation: DecisionAnnotation | dict[str, object] | None = None,
    ) -> Any:
        return self._decide(
            env, "step", int(action), lambda: env.step(action), annotation
        )

    def command(
        self,
        env: managym.Env,
        command: str,
        *,
        transition_only: bool = True,
        annotation: DecisionAnnotation | dict[str, object] | None = None,
    ) -> Any:
        """Retain and execute one action-aligned native Command.

        The native executor rejects a Command with answers; a rejected or stale
        Command leaves no committed decision or reducer callback. Pending input
        is durable even without an explicitly configured journal.
        """
        self._activate_journal(env)
        if self._game_journal is None:
            path = self.failure_directory / f"{uuid4().hex}.jsonl"
            self._game_journal = _GameJournal(path)
            self._automatic_journal = True
        execute = (
            env.execute_semantic_command_json
            if transition_only
            else env.step_semantic_command
        )
        return self._decide(
            env, "command", command, lambda: execute(command), annotation
        )

    def _decide(
        self,
        env: managym.Env,
        operation: Literal["step", "command"],
        chosen: int | str,
        call: Callable[[], Any],
        annotation: DecisionAnnotation | dict[str, object] | None,
    ) -> Any:
        """Retain one commitment, execute it, then acknowledge and notify."""
        from managym.replay import RecordedDecision

        self._activate_journal(env)
        if self._game_journal is not None:
            self._begin_boundary(
                self._snapshot(), operation, actions=[chosen], active=[True]
            )
            if self._automatic_journal and not self._journal_retained:
                if self.on_retained is not None:
                    self.on_retained(self._game_journal.writer.path)
                self._journal_retained = True
        inputs = self.inputs
        retained = (
            None
            if annotation is None
            else DecisionAnnotation.model_validate(annotation)
        )
        commitment = (
            chosen if isinstance(chosen, int) else Command(**json.loads(chosen))
        )
        inputs.actor = env.current_agent_index()
        inputs.frame = env.semantic_decision_frame_json()
        if self._selected(inputs):
            inputs.digests.append(env.state_digest())
            inputs.annotations.append(retained)
        # Frame hashes align with actions once any Command has been retained.
        if isinstance(chosen, str):
            inputs.command_frame_hashes.extend(
                [None] * (len(inputs.actions) - len(inputs.command_frame_hashes))
            )
            inputs.command_frame_hashes.append(semantic_frame_sha256(inputs.frame))
        elif inputs.command_frame_hashes:
            inputs.command_frame_hashes.append(None)
        inputs.actions.append(chosen)
        decision = (
            RecordedDecision(
                len(inputs.actions) - 1,
                env.observation_for_player(inputs.actor or 0),
                commitment,
                retained,
                None
                if isinstance(commitment, int)
                else DecisionFrame.from_json(inputs.frame),
            )
            if self.reducers
            else None
        )
        with self.guard():
            if self._game_journal is not None:
                self._publish(self._snapshot(), "executing")
            result = call()
            # Acknowledge the returned native commitment before invoking consumers.
            inputs.committed = len(inputs.actions)
            self._returned()
            if decision is not None:
                self._reduction.decision(decision)
            if env.is_game_over():
                end = GameEnd(
                    status="complete",
                    winner=env.winner_index(),
                    digest=env.state_digest() if self._selected(inputs) else None,
                )
                self._write_game(inputs, end)
                self._flush_statistics()
                self._finish_reducers(env, end)
            if self._game_journal is not None:
                self._publish(self._snapshot(), "ready")
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


class _ExposureSummary(BaseModel):
    """Narrow the native per-generation receipt once at the extension boundary."""

    seed: int
    terminal: bool
    cards: CapturedCards


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
        reducers: list[Reducer[Any]] | None = None,
        **recording: Unpack[RecordingOptions],
    ) -> None:
        super().__init__(**recording)
        # Each stream owns game-local state. Only returned rows cross streams;
        # the original reducer instances own the one final aggregation.
        self._reduction = Reduction(reducers or [])
        self._reductions = (
            [Reduction(deepcopy(reducers)) for _ in range(num_envs)] if reducers else []
        )
        self._analysis_active = [False] * num_envs if reducers else []
        self._analysis_indices = [0] * num_envs if reducers else []
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
        if self.statistics_path is not None:
            self.native.enable_statistics()
        if self._reductions:
            self.native.enable_exposure_capture()
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

    def _flush_statistics(self, *, partial: bool = False) -> None:
        if self.statistics_path is not None:
            rows = self.take_statistics()
            if partial:
                rows.extend(row for row in self.statistics() if row is not None)
            self._write_statistics(rows)

    @property
    def analysis_error(self) -> str | None:
        return self._reduction.error

    def rows(self) -> list[list[Any]]:
        """Tables across completed/closed games; unfinished games add no rows yet."""
        return self._reduction.tables()

    def _end_analysis(
        self,
        index: int,
        raw: str,
        *,
        winner: int | None = None,
        reason: str | None = None,
    ) -> None:
        from managym.replay import ReplayStop

        if not self._analysis_active[index]:
            return
        reduction = self._reductions[index]
        if self._reduction.error is not None:
            self._analysis_active[index] = False
            return
        try:
            receipt = _ExposureSummary.model_validate_json(raw)
            reduction.end(
                ReplayStop(
                    "complete" if receipt.terminal else "interrupted",
                    self._analysis_indices[index],
                    reason,
                    winner,
                ),
                receipt.cards,
            )
        except Exception as exc:
            reduction.fail(f"card facts: {type(exc).__name__}: {exc}")
        self._analysis_active[index] = False
        if reduction.error is not None:
            self._reduction.fail(reduction.error)
        if self._reduction.error is None:
            for total, rows in zip(self._reduction.rows, reduction.rows, strict=True):
                total.extend(rows)
        for rows in reduction.rows:
            rows.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for game in self.games:
            self._write_game(
                game,
                GameEnd(
                    status="failed" if self._failed else "interrupted",
                    reason="capture closed",
                ),
            )
        self._flush_statistics(partial=True)
        if any(self._analysis_active):
            for index, raw in enumerate(self.native.current_exposure()):
                self._end_analysis(index, raw, reason="capture closed")
        self._close_output()
        self.flush_policy_actions()

    def capture_failure(self, error: BaseException) -> None:
        self._failure(error, lambda: self._evidence_for(self.games, "step"))

    def recording_mask(self) -> list[bool]:
        return [self._selected(game) for game in self.games]

    def _snapshot(self) -> GameRecord:
        return self._evidence_for(self.games, "step")

    def _evidence_for(
        self, games: list[GameInputs], operation: Literal["reset", "step"]
    ) -> GameRecord:
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

    def reset_all_into_buffers(self, players: managym.GameSetup) -> None:
        self.reset_each_into_buffers([players for _ in self.games])

    def reset_each_into_buffers(self, players: list[managym.GameSetup]) -> None:
        self._reset_each(players, lambda: self.native.reset_each_into_buffers(players))

    def reset_all(self, players: managym.GameSetup) -> Any:
        return self._reset_each(
            [players for _ in self.games], lambda: self.native.reset_all(players)
        )

    def _reset_each(
        self, players: list[managym.GameSetup], call: Callable[[], Any]
    ) -> Any:
        self._activate_journal(self.native)
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reset_each",
                setup_known=not self._sampled_roots,
                seeds=self.next_seeds,
                players=[
                    _retain_players(setup, seed)
                    for setup, seed in zip(players, self.next_seeds, strict=True)
                ],
            )
        for game, setup, seed in zip(self.games, players, self.next_seeds, strict=True):
            self._write_game(game, GameEnd(status="interrupted", reason="reset"))
            game.players = _retain_players(setup, seed)
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self._executing()
            result = call()
        self._returned()
        self._synchronize()
        return result

    def reassign_roots_into_buffers(
        self, players: list[managym.GameSetup | None]
    ) -> None:
        """Retain new setups only after native atomic fresh-root validation."""
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reassign_roots",
                players=[
                    None if setup is None else _retain_players(setup, seed)
                    for setup, seed in zip(
                        players, (game.seed for game in self.games), strict=True
                    )
                ],
            )
        # Journal the intent from synchronized inputs before any native read.
        proposed = [game.model_copy(deep=True) for game in self.games]
        for game, setup, seed in zip(
            proposed, players, self.native.current_game_seeds(), strict=True
        ):
            if setup is not None:
                self._write_game(
                    self.games[game.stream - self.games[0].stream],
                    GameEnd(status="interrupted", reason="root reassigned"),
                )
                game.players = _retain_players(setup, seed)
                game.restart(seed)
        with self._guard(lambda: self._evidence_for(proposed, "reset")):
            self._executing()
            self.native.reassign_roots_into_buffers(players)
        self._returned()
        self.games = proposed
        self._synchronize()

    def reset_seeds_into_buffers(self, seeds: list[int]) -> None:
        self._activate_journal(self.native)
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "reset"),
                "reset_seeds",
                setup_known=not self._sampled_roots,
                seeds=seeds,
            )
        self.next_seeds = list(seeds)
        for game, seed in zip(self.games, seeds, strict=True):
            self._write_game(game, GameEnd(status="interrupted", reason="reset"))
            game.restart(seed)
        with self._guard(lambda: self._evidence_for(self.games, "reset")):
            self._executing()
            self.native.reset_seeds_into_buffers(seeds)
        self._returned()
        self._synchronize()

    def _synchronize(self, *, completed: bool = False) -> None:
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
        if restarted:
            self._flush_statistics()
        players = self.native.current_player_configs() if restarted else []
        infos = (
            self.native.get_last_info()
            if any(self.recording_mask()) or self._reductions
            else []
        )
        if restarted and self._reductions:
            for index, receipts in enumerate(self.native.take_exposure()):
                for raw in receipts:
                    info = infos[index] if completed else {}
                    self._end_analysis(
                        index,
                        raw,
                        winner=info.get("winner_index"),
                        reason=info.get("game_error") if completed else "root replaced",
                    )
        for index in restarted:
            game = self.games[index]
            if game.actions and self._selected(game):
                info = infos[index]
                self._write_game(
                    game,
                    GameEnd(
                        status="complete" if completed else "interrupted",
                        reason=None if completed else "root replaced",
                        winner=info.get("winner_index") if completed else None,
                        digest=info.get("record_terminal_digest")
                        if completed
                        else None,
                    ),
                )
            game.restart(seeds[index])
            # The ending row owns the old setup; retain the new native root only
            # after writing it, including disclosure resolved for the new seed.
            game.players = _retain_players(players[index], seeds[index])
        if self._reductions and self._reduction.error is None:
            for index in restarted:
                self._analysis_indices[index] = 0
                self._analysis_active[index] = True
                self._reductions[index].start(self.games[index].players)
                if self._reductions[index].error is not None:
                    self._reduction.fail(self._reductions[index].error)
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

    def step(
        self,
        actions: list[int],
        *,
        annotations: list[DecisionAnnotation | dict[str, object] | None] | None = None,
    ) -> Any:
        return self._step(
            actions, None, isolated=False, buffered=False, annotations=annotations
        )

    def step_into_buffers(
        self,
        actions: list[int],
        active: list[bool] | None = None,
        *,
        annotations: list[DecisionAnnotation | dict[str, object] | None] | None = None,
    ) -> None:
        self._step(actions, active, isolated=False, annotations=annotations)

    def step_isolated_into_buffers(
        self,
        actions: list[int],
        active: list[bool] | None = None,
        recover_capacity: bool = False,
        *,
        annotations: list[DecisionAnnotation | dict[str, object] | None] | None = None,
    ) -> None:
        self._step(
            actions,
            active,
            isolated=True,
            recover_capacity=recover_capacity,
            annotations=annotations,
        )

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
        annotations: list[DecisionAnnotation | dict[str, object] | None] | None = None,
    ) -> Any:
        self._activate_journal(self.native)
        enabled = active if active is not None else [True] * len(self.games)
        if self._game_journal is not None:
            self._begin_boundary(
                self._evidence_for(self.games, "step"),
                "step_isolated" if isolated else "step",
                actions=[int(action) for action in actions],
                active=enabled,
                setup_known=not self._sampled_roots,
            )
        if annotations is not None and len(annotations) != len(self.games):
            raise ValueError("annotations must have one entry per stream")
        selected_games = [
            on and self._selected(game)
            for on, game in zip(enabled, self.games, strict=True)
        ]
        if self._recording:
            self.native.record_games(selected_games)
        digests = (
            self.native.state_digests(selected_games)
            if any(selected_games)
            else [None] * len(self.games)
        )
        actors = self.native.current_agent_indices()
        frames = self.native.semantic_decision_frames_json(active)
        summaries = (
            self.native.current_action_summaries() if self._action_cutoff else []
        )
        hook_annotations: list[DecisionAnnotation | None] = []
        if self._reductions and self._reduction.error is None:
            hook_annotations = [
                None
                if annotation is None
                else DecisionAnnotation.model_validate(annotation)
                for annotation in (annotations or [None] * len(self.games))
            ]
        hook_observations = (
            self.native.decision_observations(enabled) if hook_annotations else []
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
                if selected_games[index]:
                    digest = digests[index]
                    assert digest is not None
                    game.digests.append(digest)
                    annotation = annotations[index] if annotations is not None else None
                    retained = (
                        None
                        if annotation is None
                        else DecisionAnnotation.model_validate(annotation)
                    )
                    if retained is not None:
                        if retained.probabilities is not None and frame is not None:
                            offers = json.loads(frame)["offers"]
                            if len(retained.probabilities) != len(offers):
                                retained = retained.model_copy(
                                    update={
                                        "probabilities": None,
                                        "extra": {
                                            **retained.extra,
                                            "probabilities_unavailable": "policy action capacity differs from native offers",
                                        },
                                    }
                                )
                        coordinates = self._coordinates()["context"].get(
                            "coordinates", {}
                        )
                        if (
                            retained.update is None
                            and isinstance(coordinates, dict)
                            and isinstance(coordinates.get("updates"), int)
                        ):
                            retained = retained.model_copy(
                                update={"update": coordinates["updates"]}
                            )
                    game.annotations.append(retained)
                game.actions.append(int(action))
                attempted.append(game)
        try:
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
        except BaseException:
            if self._reductions:
                self._reduction.fail(
                    "vector batch failed: committed analysis coverage is unknown"
                )
            raise
        infos = (
            self.native.get_last_info()
            if isolated and (any(selected_games) or self._reductions)
            else []
        )
        for index, (game, on) in enumerate(zip(self.games, enabled, strict=True)):
            if on and not (infos and infos[index].get("game_error")):
                game.committed = len(game.actions)
        self._returned()
        if hook_observations:
            from managym.replay import RecordedDecision

            for index, observation in enumerate(hook_observations):
                if observation is not None and not (
                    infos and infos[index].get("game_error")
                ):
                    reduction = self._reductions[index]
                    reduction.decision(
                        RecordedDecision(
                            self._analysis_indices[index],
                            observation,
                            int(actions[index]),
                            hook_annotations[index],
                        )
                    )
                    self._analysis_indices[index] += 1
                    if reduction.error is not None:
                        self._reduction.fail(reduction.error)
        self._synchronize(completed=True)
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
        with self._guard(self._snapshot):
            self.native = getattr(env, method)(*arguments)

    def _snapshot(self) -> GameRecord:
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
        with self._guard(self._snapshot):
            return self.native.step_active(actions)

    def finish_random(self) -> None:
        self.search.operations.append("finish_random")
        with self._guard(self._snapshot):
            self.native.finish_random()


def _read_journal(path: Path) -> GameRecord:
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
        evidence = GameRecord.model_validate(row)
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
                evidence.games[0].committed = len(evidence.games[0].actions)
            elif "error_class" in row:
                evidence.execution_outcome = "failed"
                evidence.error_class = row["error_class"]
                evidence.error = row["error"]
                evidence.traceback = row["traceback"]
    return evidence


def record_from_row(row: dict[str, Any]) -> GameRecord:
    """Project retained exposure receipts by field without loading a bot package."""
    if not isinstance(row, dict):
        raise ValueError("expected a game record object")
    if row.get("schema") != "training-exposure-v1":
        return GameRecord.model_validate(row)
    game = row["game"]
    trace = game["trace"]
    return GameRecord(
        mode="scalar",
        operation="step",
        skip_trivial=row["skip_trivial"],
        identities=row["identities"],
        context=row["context"],
        seconds=row["seconds"],
        games=[
            GameInputs(
                stream=row["stream"],
                seed=game["seed"],
                players=trace["player_configs"],
                actions=trace["actions"],
                end=GameEnd(
                    status="complete" if game["terminal"] else "interrupted",
                    winner_retained=False,
                ),
            )
        ],
    )


def records_from_source(
    source: object, *, journal_path: Path | None = None
) -> list[GameRecord] | None:
    """Admit record rows or files; None identifies another format, never bad evidence.

    A journal checkpoint needs its complete file to project subsequent changes.
    Recognized but malformed records raise rather than falling back to a trace.
    """
    if isinstance(source, Path):
        return list(read_records(source))
    if not isinstance(source, dict):
        return None
    schema = source.get("schema")
    if schema == "training-game-checkpoint-v1":
        if journal_path is None:
            raise ValueError("a journal requires its complete file")
        return list(read_records(journal_path))
    if schema in {"managym-game-v1", "native-failure-v1", "training-exposure-v1"}:
        return list(_games_from_snapshot(record_from_row(source)))
    return None


def _games_from_snapshot(record: GameRecord) -> Iterator[GameRecord]:
    """Project stream-local boundaries together with each game's inputs."""
    for index, game in enumerate(record.games):
        boundary = record.boundary
        if boundary is not None:
            fields = {
                name: [values[index]] if index < len(values) else []
                for name in (
                    "active",
                    "pending_actions",
                    "committed_counts",
                    "proposed_seeds",
                    "proposed_players",
                    "last_known_frames",
                    "frame_prefix_lengths",
                )
                if (values := getattr(boundary, name))
            }
            boundary = boundary.model_copy(update=fields)
        yield record.model_copy(update={"games": [game], "boundary": boundary})


def read_records(path: Path) -> Iterator[GameRecord]:
    """Yield one game at a time, including each stream of a projected journal."""
    for snapshot in _read_snapshots(path):
        yield from _games_from_snapshot(snapshot)


def _read_snapshots(path: Path) -> Iterator[GameRecord]:
    """Read retained rows without splitting failure-reproduction batches."""
    if path.is_dir():
        for child in sorted(path.glob("*.jsonl")):
            yield from _read_snapshots(child)
        return
    with path.open("rb") as source:
        first = source.readline()
        if not first:
            return
        if not first.endswith(b"\n"):
            try:
                incomplete = json.loads(first)
            except json.JSONDecodeError:
                incomplete = None
            if (
                isinstance(incomplete, dict)
                and incomplete.get("schema") == "training-game-checkpoint-v1"
            ):
                raise ValueError("incomplete initial training checkpoint")
            warnings.warn(
                "Incomplete final game record; ignored", RuntimeWarning, stacklevel=2
            )
            return
        row = json.loads(first)
        if not isinstance(row, dict):
            raise ValueError("expected a game record object")
        second = source.readline()
        if row.get("schema") == "training-game-checkpoint-v1" or (
            second and second.endswith(b"\n") and "schema" not in json.loads(second)
        ):
            yield _read_journal(path)
            return
        yield record_from_row(row)
        for line in chain([second] if second else [], source):
            if not line.endswith(b"\n"):
                warnings.warn(
                    "Incomplete final game record; ignored",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            yield record_from_row(json.loads(line))


def read_record(path: Path) -> GameRecord:
    """Read exactly one record or one projected journal, retaining all streams."""
    records = iter(_read_snapshots(path))
    record = next(records, None)
    if record is None or next(records, None) is not None:
        raise ValueError("expected exactly one game record")
    return record


class FailureEvidence(GameRecord):
    """Deprecated name of GameRecord, removed in 0.4.0."""

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "FailureEvidence is deprecated; use GameRecord; removal in 0.4.0",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


def read_failure(path: Path) -> GameRecord:
    warnings.warn(
        "read_failure is deprecated; use read_record; removal in 0.4.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_record(path)
