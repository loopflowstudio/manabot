"""Backend capabilities for collection, incident capture and verification.

Incident payloads use Any: their nested tensors belong to the framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, runtime_checkable

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.policy.protocol import Backend, BackendPolicy, PolicyLoader
from manabot.training.updates.backend import Learner, LearnerFile, Policy, UpdateBackend

if TYPE_CHECKING:
    from etude.train.tracking import TrainingRun

    from manabot.infra.config import ModelConfig
    from manabot.policy.readings import Readings
    from manabot.training.learning import Precision
    from manabot.training.loops.self_play import Counts
    from manabot.training.rollout.collection import Collector, RolloutBatch
    from manabot.training.rollout.numerics import CollectionSession
    from manabot.training.rollout.selection_data import SelectionPredictor
    from managym import PlayerConfig


class NumericalIncident(Protocol):
    invariant: str
    tensors: dict[str, Any]
    health: dict[str, int | float]


@dataclass(frozen=True)
class IncidentCapture:
    evidence: object
    state: object
    shapes: dict[str, object]
    state_tensor_bytes: int
    full_batch_captured: bool
    suffix: str = ".pt"


class TrainingState(Protocol):
    def save(self, value: object, stream: BinaryIO) -> None: ...
    def capture_incident(
        self,
        error: NumericalIncident,
        learner: object,
        collector: Collector,
        batch: RolloutBatch | None,
        ema: BackendPolicy | None,
        rng: np.random.Generator,
        *,
        evidence_bytes: int,
        max_rows: int,
    ) -> IncidentCapture: ...


class PolicyFactory(Protocol):
    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> BackendPolicy: ...


class FreshPolicies(Protocol):
    @property
    def runtime(self) -> PolicyFactory: ...
    def seed(self, seed: int, *, deterministic: bool | None = None) -> None: ...


class SessionFactory(Protocol):
    def __call__(
        self, *, device: str, seed: int, precision: Precision = "float32"
    ) -> CollectionSession: ...


class CollectionSessions(Protocol):
    @property
    def CollectionSession(self) -> SessionFactory: ...


class SelectionPredictors(Protocol):
    @property
    def SelectionPredictor(
        self,
    ) -> Callable[[BackendPolicy, int], SelectionPredictor]: ...


class SelectionBackend(PolicyLoader, Protocol):
    @property
    def collection(self) -> SelectionPredictors: ...
    def metadata(self, path: str) -> dict[str, Any]: ...


class SelfPlayPolicy(BackendPolicy, Policy, Protocol):
    def memory_gates(self) -> dict[str, float]:
        """Effective decision-memory gates at an update boundary."""
        ...

    @property
    def readings(self) -> Readings: ...


class SelfPlayLearner(Learner, Protocol):
    @property
    def agent(self) -> SelfPlayPolicy: ...


class ExecutionScope(Protocol):
    def scoped(
        self, threads: int, precision: Precision, device: str
    ) -> AbstractContextManager[object]: ...
    def preserve_rng(self, device: str) -> AbstractContextManager[object]: ...


class VerificationBackend(Backend, Protocol):
    @property
    def runtime(self) -> ExecutionScope: ...


class SelfPlayRuntime(ExecutionScope, Protocol):
    def synchronize(self, device: str) -> None: ...
    def memory_metrics(self, device: str) -> dict[str, int]: ...
    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
        pretrained: BackendPolicy | None = None,
        precision: Precision | None = None,
    ) -> SelfPlayPolicy: ...
    def create_learner(
        self,
        agent: BackendPolicy,
        device: str,
        *,
        minibatch_rows: int,
        precision: Precision,
        execution_rows: int | None,
        bootstrap_rows: int | None,
    ) -> SelfPlayLearner: ...


class SelfPlayLearning(LearnerFile, Protocol):
    def update_ema(
        self, ema: BackendPolicy, agent: BackendPolicy, rate: float, /
    ) -> None: ...


class SelfPlayBackend(UpdateBackend, Protocol):
    """Numerics for one fresh self-play iteration and its portable policies."""

    name: str

    @property
    def runtime(self) -> SelfPlayRuntime: ...
    @property
    def learner(self) -> SelfPlayLearning: ...
    @property
    def collection(self) -> CollectionSessions: ...
    def metadata(self, path: str | Path) -> dict[str, Any]: ...
    def seed(self, seed: int, *, deterministic: bool | None = None) -> None: ...
    def load_policy(
        self,
        path: str | Path,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> tuple[BackendPolicy, ObservationSpace]: ...
    def save_policy(
        self,
        agent: BackendPolicy,
        obs_space: ObservationSpace,
        path: str | Path,
        *,
        player_configs: Sequence[PlayerConfig] | None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...


@runtime_checkable
class Continuation(TrainingState, Protocol):
    """Objects Etude can restore together, plus this framework's incident evidence."""

    def restore(
        self,
        run: TrainingRun,
        learner: SelfPlayLearner,
        ema: BackendPolicy | None,
        counts: Counts,
    ) -> Callable[[Path], None] | None:
        """Restore the live objects and supply a staged export for callback saves."""
        ...

    def framework(self) -> dict[str, str | None]: ...


def require_continuation(backend: SelfPlayBackend) -> Continuation:
    state = getattr(backend, "state", None)
    if not isinstance(state, Continuation):
        raise ValueError(f"{backend.name} does not support resumable self-play")
    return state
