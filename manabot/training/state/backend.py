"""State capabilities received by host admission and recovery.

Only serialized framework payloads use Any: their nested format belongs to
its codec and is validated there. Saved dataclass ownership remains on the host.
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
    from manabot.training.regime import Precision
    from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
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
    def policy_state(self, policy: BackendPolicy) -> dict[str, Any]: ...
    def restore_policy(
        self,
        policy: BackendPolicy,
        state: dict[str, Any],
        *,
        source_config: ModelConfig | None = None,
    ) -> None: ...
    def optimizer_state(self, learner: object) -> dict[str, Any]: ...
    def restore_optimizer(self, learner: object, state: dict[str, Any]) -> None: ...
    def validate_optimizer(self, learner: object) -> None: ...
    def framework_rng(self) -> object: ...
    def restore_rng(self, state: object) -> None: ...
    def save(self, value: object, stream: BinaryIO) -> None: ...
    def load(self, stream: BinaryIO) -> object: ...
    def parameter_names(self, policy: BackendPolicy) -> list[str]: ...
    def validate_policy(
        self, policy: BackendPolicy, *, label: str = "learning-state"
    ) -> None: ...
    def clone_policy(self, policy: BackendPolicy) -> BackendPolicy: ...
    def admit_adam(self, path: str, policy: BackendPolicy) -> dict[str, Any]: ...
    def capture_incident(
        self,
        error: NumericalIncident,
        learner: object,
        collector: SeatRoutedCollector,
        batch: RolloutBatch | None,
        ema: BackendPolicy | None,
        rng: np.random.Generator,
        *,
        evidence_bytes: int,
        max_rows: int,
    ) -> IncidentCapture: ...


class TrainingRuntime(Protocol):
    def require_execution(self) -> None: ...
    def identities(self, devices: Sequence[str]) -> dict[str, object]: ...
    def configure(self, threads: int, precision: Precision, device: str) -> None: ...
    def threads(self) -> int: ...
    def synchronize(self, device: str) -> None: ...
    def create_policy(
        self,
        space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> BackendPolicy: ...
    def freeze(self, policy: BackendPolicy) -> None: ...
    def policy_device(self, policy: BackendPolicy) -> str: ...
    def preserve_policy_rng(
        self, policy: BackendPolicy
    ) -> AbstractContextManager[object]: ...
    def cuda_peak_metrics(self, device: str) -> dict[str, int]: ...
    def mps_memory_metrics(self, device: str) -> dict[str, int]: ...


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
