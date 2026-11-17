"""Backend capabilities for collection, incident capture and verification.

Incident payloads use Any: their nested tensors belong to the framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

import numpy as np

from manabot.belief.protocol import SamplerBackend
from manabot.env.observation import ObservationSpace
from manabot.policy.protocol import BackendPolicy

if TYPE_CHECKING:
    from manabot.belief.encoding import BeliefEncodingSchema
    from manabot.belief.sampling_fit import BeliefTrainer
    from manabot.infra.config import ModelConfig
    from manabot.training.learning import Precision
    from manabot.training.rollout.collection import Collector, RolloutBatch
    from manabot.training.rollout.numerics import CollectionSession
    from manabot.training.rollout.selection_data import SelectionPredictor
    from manabot.verify.choice_corpus import Case
    from manabot.verify.learning_regression import PolicySnapshot
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


class TrainingState(Protocol):
    def preserve_rng(self) -> AbstractContextManager[object]: ...
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


class TrainingRuntime(Protocol):
    def configure(self, threads: int, precision: Precision, device: str) -> None: ...
    def threads(self) -> int: ...
    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> BackendPolicy: ...


class SessionFactory(Protocol):
    def __call__(
        self, *, device: str, seed: int, precision: Precision = "float32"
    ) -> CollectionSession: ...


class CollectionFactory(Protocol):
    @property
    def CollectionSession(self) -> SessionFactory: ...
    @property
    def SelectionPredictor(
        self,
    ) -> Callable[[BackendPolicy, int], SelectionPredictor]: ...


class Verification(Protocol):
    def inspect_policy(
        self, path: Path, cases: list[Case], updates: int
    ) -> PolicySnapshot: ...


class TrainingBackend(SamplerBackend, Protocol):
    """Execution roles beyond ordinary inference and portable updates.

    An unsupported runtime refuses before collection or recovery starts.
    """

    @property
    def verification(self) -> Verification: ...

    @property
    def runtime(self) -> TrainingRuntime: ...

    def save_policy(
        self,
        agent: BackendPolicy,
        obs_space: ObservationSpace,
        path: str | Path,
        *,
        player_configs: Sequence[PlayerConfig] | None,
        extra: dict[str, Any] | None = None,
        belief_schema: "BeliefEncodingSchema | None" = None,
    ) -> None: ...

    @property
    def state(self) -> "TrainingState": ...

    @property
    def belief_fit(self) -> BeliefTrainer: ...

    @property
    def collection(self) -> CollectionFactory: ...
