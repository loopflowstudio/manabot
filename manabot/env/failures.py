"""Training adapters for managym recording.

CaptureScope owns live run coordinates, producer destinations and outer phase
markers. Adapters retain exposure receipts and policy cutoff accounting; the
engine owns replay inputs, native boundaries and durable journal projection.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Literal
from uuid import uuid4
import warnings
from weakref import WeakValueDictionary

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.infra.log import getLogger
import managym
from managym import recording
from managym.card_capture import CapturedCards
from managym.recording import (
    FailureEvidence as _FailureEvidence,
    GameInputs as _GameInputs,
    PlayerSetup as _PlayerSetup,
    _append,
    _jsonl,
    atomic_private_json as _atomic_private_json,
)


class ActionCutoffCounts(BaseModel):
    decisions: int = 0
    overflows: int = 0
    dropped_actions: int = 0


class ActionCutoffSample(BaseModel):
    original_count: int
    retained_count: int
    dropped_count: int
    decision_kind: str
    game: _GameInputs


class NativeTrainingTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_configs: list[_PlayerSetup]
    decision_memory_include_own: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    actions: list[int]


class NativeTrainingGame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    terminal: bool
    trace: NativeTrainingTrace
    cards: CapturedCards

    @model_validator(mode="after")
    def bound(self) -> "NativeTrainingGame":
        if len(self.trace.player_configs) != 2:
            raise ValueError("training capture seed or seating differs")
        if self.cards.capture.exposure is None:
            raise ValueError("training capture has no exposure instrument")
        for seat, player in enumerate(self.trace.player_configs):
            for origin, expected in (
                ("deck", player.decklist),
                ("outside", player.sideboard),
            ):
                observed = Counter(
                    card.name
                    for card in self.cards.capture.initial
                    if card.owner == seat and card.origin == origin
                )
                if dict(observed) != expected:
                    raise ValueError("training capture cards differ from replay decks")
        choices = self.cards.capture.exposure.choices
        if len(choices) > len(self.trace.actions) or any(
            later.revision <= earlier.revision
            for earlier, later in zip(choices, choices[1:])
        ):
            raise ValueError("training choices do not match a committed replay prefix")
        return self


class TrainingExposureReceipt(BaseModel):
    """Private observed game, not a claim that its transitions reached an update.

    Native replay inputs own sampled decks, seed and committed positional actions.
    Context binds the learner/run and update coordinates at retention. Terminal
    and interrupted prefixes remain separate; replay has not been executed.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["training-exposure-v1"] = Field(
        default="training-exposure-v1", alias="schema"
    )
    phase: Literal["training"] = "training"
    stream: int
    game: NativeTrainingGame
    skip_trivial: bool
    identities: dict[str, object]
    context: dict[str, object]
    seconds: float

    @model_validator(mode="after")
    def committed_choices(self) -> "TrainingExposureReceipt":
        exposure = self.game.cards.capture.exposure
        assert exposure is not None
        if (
            not self.skip_trivial
            or [choice.selected for choice in exposure.choices]
            != self.game.trace.actions
        ):
            raise ValueError(
                "training exposure must retain every surfaced replay choice"
            )
        return self

    @property
    def sha256(self) -> str:
        """Digest of the exact durable JSONL bytes written by the capture owner."""
        body = _jsonl(self.model_dump(mode="json", by_alias=True))
        return hashlib.sha256(body.encode()).hexdigest()

    def replay_inputs(self) -> _FailureEvidence:
        return _FailureEvidence(
            mode="scalar",
            operation="step",
            skip_trivial=self.skip_trivial,
            games=[
                _GameInputs(
                    stream=self.stream,
                    seed=self.game.seed,
                    players=self.game.trace.player_configs,
                    decision_memory_include_own=self.game.trace.decision_memory_include_own,
                    actions=self.game.trace.actions,
                )
            ],
            identities=self.identities,
            context=self.context,
            seconds=self.seconds,
            execution_outcome="last_command_succeeded",
        )


@dataclass
class CaptureScope:
    directory: Path
    context: Callable[[], dict[str, object]] = dict
    paths: list[Path] = field(default_factory=list)
    training_exposure: bool = False
    exposure_directory: Path | None = None
    training_game_logs: bool = False
    _operation_sequence: int = 0
    _operations: list[str] = field(default_factory=list)
    _writers: WeakValueDictionary[str, "_TrainingCapture"] = field(
        default_factory=WeakValueDictionary
    )

    def retire_cursors(self) -> None:
        """Retire settled producers when a stage replaces its collector."""
        if self.training_game_logs:
            for path in self.journal_directory.glob("*.jsonl"):
                path.unlink()

    @property
    def journal_directory(self) -> Path:
        return self.directory.parent / "game-journals"

    @contextmanager
    def operation(self, name: str) -> Iterator[None]:
        """Owner-thread operation stack, restoring collection after inference."""
        if not self.training_game_logs:
            yield
            return
        self._operations.append(name)
        self._mark_operation(name, "pending")
        try:
            yield
        except BaseException:
            # Preserve the interrupted boundary if an exception unwinds it.
            self._operations.pop()
            raise
        else:
            self._operations.pop()
            self._mark_operation(name, "returned")
            if self._operations:
                self._mark_operation(self._operations[-1], "pending")

    def _mark_operation(self, name: str, state: str) -> None:
        self._operation_sequence += 1
        _atomic_private_json(
            self.journal_directory / "operation.json",
            {
                "operation": name,
                "state": state,
                "sequence": self._operation_sequence,
                "timestamp": time.time(),
                "context": {
                    key: value
                    for key, value in self.context().items()
                    if key
                    in {"run", "stage", "seed", "coordinates", "update", "updates"}
                },
            },
        )


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


class _TrainingCapture:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.scope = _scope.get()
        kwargs.setdefault("context_provider", self._live_context)
        if self.scope is not None:
            kwargs.setdefault("failure_directory", self.scope.directory)
            kwargs.setdefault("on_retained", self.scope.paths.append)
        self._policy_counts: dict[str, ActionCutoffCounts] = {}
        self._policy_samples: list[ActionCutoffSample] = []
        self._policy_path: Path | None = None
        super().__init__(*args, **kwargs)

    def _live_context(self) -> dict[str, object]:
        scope = _scope.get() or self.scope
        return scope.context() if scope else {}

    def _retain(self, evidence: _FailureEvidence) -> Path:
        scope = _scope.get() or self.scope
        if scope is not None:
            self.failure_directory = scope.directory
            self.on_retained = scope.paths.append
        return super()._retain(evidence)

    def _start_cursor(self, name: str) -> None:
        if self.scope is not None and self.scope.training_game_logs:
            base = name
            index = 0
            while name in self.scope._writers:
                index += 1
                name = f"{base}-{index}"
            self.scope._writers[name] = self
            self._game_journal = recording._GameJournal(
                self.scope.journal_directory / f"{name}.jsonl"
            )
        super()._start_cursor(name)

    def record_policy_actions(
        self, count: int, kind: str, capacity: int, game: _GameInputs
    ) -> None:
        """Count decisions consumed under the explicit first-legal policy.

        Keep at most four exact replay prefixes and log at powers of two, then
        flush exact aggregate frequencies at collector/game close. The complete
        native action set is never changed here.
        """
        if kind not in self._policy_counts:
            self._policy_counts[kind] = ActionCutoffCounts()
        counts = self._policy_counts[kind]
        counts.decisions += 1
        retained = managym.policy_action_count(count, capacity, "first-legal")
        if retained == count:
            return
        counts.overflows += 1
        counts.dropped_actions += count - retained
        if len(self._policy_samples) < 4:
            self._policy_samples.append(
                ActionCutoffSample(
                    original_count=count,
                    retained_count=retained,
                    dropped_count=count - retained,
                    decision_kind=kind,
                    game=game.model_copy(deep=True),
                )
            )
        overflows = sum(counts.overflows for counts in self._policy_counts.values())
        if overflows & (overflows - 1) == 0:
            self.flush_policy_actions()
            getLogger(__name__).info(
                "policy_action_cutoff %s",
                json.dumps(
                    {
                        "overflows": overflows,
                        "original_count": count,
                        "retained_count": retained,
                        "dropped_count": count - retained,
                        "decision_kind": kind,
                        "seed": game.seed,
                        "stream": game.stream,
                        "replay_path": str(self._policy_path),
                    }
                ),
            )

    def policy_action_report(self) -> dict[str, object]:
        return {
            "schema": "policy-action-cutoff-v1",
            "policy": "first-legal",
            "counts_by_kind": {
                kind: counts.model_dump()
                for kind, counts in self._policy_counts.items()
            },
            "samples": [
                sample.model_dump(mode="json") for sample in self._policy_samples
            ],
            "identities": self.identities,
            **self._coordinates(),
        }

    def flush_policy_actions(self) -> None:
        if not self._policy_counts:
            return
        if self._policy_path is None:
            scope = _scope.get() or self.scope
            directory = scope.directory if scope else Path(".runs/action-cutoffs")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._policy_path = directory / f"action-cutoff-{uuid4().hex}.json"
        temporary = self._policy_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.policy_action_report(), sort_keys=True) + "\n"
        )
        temporary.replace(self._policy_path)


class ScalarCapture(_TrainingCapture, recording.ScalarCapture):
    """Engine recording with run scope and policy accounting."""


class CapturedVector(_TrainingCapture, recording.CapturedVector):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._exposure = bool(self.scope and self.scope.training_exposure)
        if self._exposure:
            if self.player_policies != ("agent", "agent") or not self.skip_trivial:
                raise ValueError(
                    "training exposure requires surfaced seats and trivial skipping"
                )
            self.native.enable_exposure_capture()

    def _retain_exposure(self, *, current: bool = False) -> None:
        if not self._exposure or not any(game.players for game in self.games):
            return
        assert self.scope is not None
        retained = self.native.take_exposure()
        if current:
            for rows, raw in zip(retained, self.native.current_exposure(), strict=True):
                rows.append(raw)
        directory = (
            self.scope.exposure_directory or self.scope.directory.parent / "exposure"
        )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for inputs, rows in zip(self.games, retained, strict=True):
            for raw in rows:
                game = NativeTrainingGame.model_validate_json(raw)
                # Auto-reset prepares a fresh unused root; it is not an attempted game.
                if not game.trace.actions and not game.terminal:
                    continue
                receipt = TrainingExposureReceipt(
                    stream=inputs.stream,
                    game=game,
                    skip_trivial=self.skip_trivial,
                    identities=self.identities,
                    **self._coordinates(),
                )
                body = receipt.model_dump(mode="json", by_alias=True)
                path = directory / f"{receipt.sha256}.jsonl"
                if not path.exists():
                    _append(path, body, create=True)
                    self.scope.paths.append(path)


class CapturedSearch(_TrainingCapture, recording.CapturedSearch):
    """Engine search evidence with training policy accounting."""


# Published evidence names retain a warning-only alias until 0.4.0.
def __getattr__(name: str) -> Any:
    if name.startswith("_") or not hasattr(recording, name):
        raise AttributeError(name)
    warnings.warn(
        f"manabot.env.failures.{name} moved to managym.recording; removal in 0.4.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(recording, name)
