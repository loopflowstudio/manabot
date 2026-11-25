"""Training adapters for managym recording.

CaptureScope owns live run coordinates, producer destinations and outer phase
markers. Adapters retain policy cutoff accounting; the
engine owns replay inputs, native boundaries and durable journal projection.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from manabot.infra.log import getLogger
from manabot.infra.run_settings import RunSettings
import managym
from managym import recording
from managym.diagnostics import Diagnostics, Operation


class ActionCutoffCounts(BaseModel):
    decisions: int = 0
    overflows: int = 0
    dropped_actions: int = 0


class ActionCutoffSample(BaseModel):
    original_count: int
    retained_count: int
    dropped_count: int
    decision_kind: str
    game: recording.GameInputs


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

    def _retain(self, evidence: recording.GameRecord) -> Path:
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
        self, count: int, kind: str, capacity: int, game: recording.GameInputs
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
                    game=(
                        self.game_inputs(game.stream - self.games[0].stream)
                        if isinstance(self, recording.RecordedVectorEnv)
                        else game
                    ).model_copy(deep=True),
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


class RecordedVectorEnv(_TrainingCapture, recording.RecordedVectorEnv):
    """Engine recording with run scope and policy accounting."""


class CapturedSearch(_TrainingCapture, recording.CapturedSearch):
    """Engine search evidence with training policy accounting."""
