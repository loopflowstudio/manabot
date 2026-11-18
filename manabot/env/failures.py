"""Training adapters for managym recording.

CaptureScope owns live run coordinates, producer destinations and outer phase
markers. Adapters retain policy cutoff accounting; the
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
from typing import Any, Literal
from uuid import uuid4
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.infra.log import getLogger
from manabot.infra.run_settings import RunSettings
import managym
from managym import recording
from managym.card_capture import CapturedCards
from managym.diagnostics import Diagnostics, Operation
from managym.recording import (
    GameInputs as _GameInputs,
    GameRecord as _GameRecord,
    PlayerSetup as _PlayerSetup,
    _game_disclosure,
    _jsonl,
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
    decision_memory: bool = Field(default=True, exclude_if=lambda value: value)
    decision_memory_include_own: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    actions: list[int]

    @model_validator(mode="after")
    def uniform_disclosure(self) -> "NativeTrainingTrace":
        _game_disclosure(self.player_configs)
        return self


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


@dataclass
class CaptureScope:
    directory: Path
    context: Callable[[], dict[str, object]] = dict
    paths: list[Path] = field(default_factory=list)
    settings: RunSettings = RunSettings()
    _diagnostics: Diagnostics | None = field(default=None, init=False, repr=False)

    @property
    def detailed(self) -> bool:
        return (
            self._diagnostics.level
            if self._diagnostics is not None
            else self.settings.diagnostics
        ) == "detailed"

    def retire_cursors(self) -> None:
        """Retire settled producers when a stage replaces its collector."""
        if self.detailed:
            for path in self.journal_directory.glob("*.jsonl"):
                if path.name != "escalations.jsonl":
                    path.unlink()

    @property
    def journal_directory(self) -> Path:
        return self.directory.parent / "game-journals"

    @contextmanager
    def operation(self, name: str, *, per_step: bool = False) -> Iterator[Operation]:
        diagnostics = self._diagnostics
        if diagnostics is None:
            with capture_scope(self):
                with self.operation(name, per_step=per_step) as operation:
                    yield operation
            return
        with diagnostics.operation(name, per_step=per_step) as operation:
            yield operation


_scope: ContextVar[CaptureScope | None] = ContextVar("game_failure_scope", default=None)


@contextmanager
def capture_scope(scope: CaptureScope) -> Iterator[None]:
    token = _scope.set(scope)
    try:
        scope._diagnostics = Diagnostics(
            scope.journal_directory,
            level=scope.settings.diagnostics,
            stall_seconds=scope.settings.stall_seconds,
            context_provider=lambda: {
                key: value
                for key, value in scope.context().items()
                if key in {"run", "stage", "seed", "coordinates", "update", "updates"}
            },
        )
        yield
    finally:
        if scope._diagnostics is not None:
            scope._diagnostics.close()
            scope._diagnostics = None
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
            kwargs.setdefault("record_fraction", self.scope.settings.record_games)
            kwargs.setdefault("failure_directory", self.scope.directory)
            kwargs.setdefault("on_retained", self.scope.paths.append)
        if self.scope is not None and self.scope.settings.record_games:
            kwargs["context"] = {**kwargs.get("context", {}), "source_kind": "training"}
            kwargs.setdefault(
                "games_path",
                self.scope.directory.parent / "games" / f"{uuid4().hex}.jsonl",
            )
        if self.scope is not None and self.scope.settings.statistics:
            kwargs.setdefault(
                "statistics_path",
                self.scope.directory.parent / "statistics" / f"{uuid4().hex}.jsonl",
            )
        self._policy_counts: dict[str, ActionCutoffCounts] = {}
        self._policy_samples: list[ActionCutoffSample] = []
        self._policy_path: Path | None = None
        super().__init__(*args, **kwargs)

    def _live_context(self) -> dict[str, object]:
        scope = _scope.get() or self.scope
        return scope.context() if scope else {}

    def _retain(self, evidence: _GameRecord) -> Path:
        scope = _scope.get() or self.scope
        if scope is not None:
            self.failure_directory = scope.directory
            self.on_retained = scope.paths.append
        return super()._retain(evidence)

    def _start_cursor(self, name: str) -> None:
        super()._start_cursor(name)
        if self.scope is not None and self.scope._diagnostics is not None:
            self.scope._diagnostics.add_capture(self)

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
    """Engine recording with run scope and policy accounting."""


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
