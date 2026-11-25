"""Framework-independent reads for policy inference and checkpoint admission.

Simulation loads a host-facing policy over Prepared batches. Inspection can
read checkpoint metadata and tensors without constructing a policy. The
parity learner exposes host copies of its parameters and gradients for
one captured minibatch's Targets and Settings.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

import numpy as np

from manabot.belief.encoding import BeliefRows
from manabot.env.observation import ObservationSpace
from manabot.policy.checkpoint import ParameterArrays, PolicyCheckpoint
from manabot.policy.prepare import Prepared
from manabot.policy.readings import Readings
from manabot.policy.spans import PolicyObservation

if TYPE_CHECKING:
    from manabot.infra.config import ModelConfig

BackendName = Literal["torch", "mlx"]
Precision = Literal["float32", "bfloat16"]


@dataclass(frozen=True)
class Targets:
    """Constants of one minibatch: sampled actions [B], advantages [B], value
    targets [B] or loss/draw/win mixtures [B,3], actor-selected rows [B],
    behavior log-likelihoods [B,A] (-inf off support) and reference probabilities [B,A]."""

    actions: np.ndarray
    advantages: np.ndarray
    value_targets: np.ndarray
    actor_mask: np.ndarray
    behavior_logs: np.ndarray
    reference: np.ndarray


@dataclass(frozen=True)
class Settings:
    """The damped update's coefficients for one minibatch."""

    learning_rate: float
    clip: float
    collection_kl: float
    tau: float
    max_grad_norm: float
    reference_forward_kl: float = 0.0


@dataclass(frozen=True)
class InferenceResult:
    """A completed host inference and the input transfer it required."""

    logits: np.ndarray
    transfer_bytes: int = 0
    transfers: int = 0


class InferenceScope(Protocol):
    def infer(
        self,
        obs: PolicyObservation,
        *,
        span: Callable[[str], AbstractContextManager[object]] | None = None,
    ) -> InferenceResult: ...
    def cache_programs(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class BackendPolicy(Protocol):
    observation_space: ObservationSpace
    belief_count_buckets: int
    value_semantic: Literal["signed_outcome", "win_logit"] | None

    @property
    def readings(self) -> Readings: ...

    @property
    def config(self) -> "ModelConfig": ...

    @property
    def device_type(self) -> str: ...

    @property
    def parameter_count(self) -> int: ...

    @property
    def resident_bytes(self) -> int:
        """Weights, input lookup buffers and the full cached catalog reservation."""
        ...

    @property
    def has_programs(self) -> bool: ...

    def inference_scope(self, *, cache_catalog: bool = True) -> InferenceScope: ...

    def sample_rows(
        self,
        logits: np.ndarray,
        masks: np.ndarray,
        seeds: list[int],
        deterministic: bool,
    ) -> list[int]:
        """Independent per-decision sampling, preserving the backend's global RNG."""
        ...

    def normalize_logits(
        self, logits: np.ndarray, *, double: bool = False
    ) -> np.ndarray:
        """Normalize host logits with the backend's established arithmetic."""
        ...

    def to(self, device: str) -> Self:
        """Place the policy for inference; refuse unsupported devices."""
        ...

    def eval(self) -> Self: ...

    def sample(
        self,
        batch: Prepared[np.ndarray],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray:
        """Action rows using the backend's RNG and established sampling arithmetic."""
        ...

    def sample_logits(
        self,
        logits: np.ndarray,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray: ...

    def win_probabilities(self, batch: Prepared[np.ndarray]) -> np.ndarray:
        """Sigmoid of scalar win-logit values, evaluated in the backend."""
        ...

    def prepare(
        self,
        obs: PolicyObservation,
        *,
        inference: bool = False,
        belief: BeliefRows[np.ndarray] | None = None,
    ) -> Prepared[np.ndarray]:
        """Index one admitted observation batch (`inference`: compact as
        PyTorch CPU inference does)."""
        ...

    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Logits [B,A] (-1e8 off the legal support) and values [B]."""
        ...

    def evaluate_probabilities(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities and values normalized in the policy's framework/device."""
        ...

    def cache_programs(self) -> None:
        """Encode weight-only inputs once, for inference with fixed weights."""
        ...

    def clear_cache(self) -> None:
        """Drop what `cache_programs` encoded; required before weights change."""
        ...


class BackendLearner(Protocol):
    """One optimizer over a policy's parameters. Losses and norms are the
    backend's arrays; `float()` reads them."""

    def set_learning_rate(self, rate: float) -> None:
        """The rate every following optimizer step uses."""
        ...

    def loss(self, objective: Callable[[], tuple[Any, ...]]) -> tuple[Any, ...]:
        """Evaluate `objective`, a closure over the policy forward whose first
        result is the loss; return its results. `step` and `gradients`
        differentiate the loss most recently returned here."""
        ...

    def step(self, loss: Any, max_grad_norm: float) -> Any:
        """One clipped optimizer step on `loss`: the pre-clip gradient norm."""
        ...

    def gradients(self, loss: Any) -> ParameterArrays:
        """Unclipped gradients of `loss` in the backend's layout, without
        stepping."""
        ...

    def parameters(self) -> ParameterArrays:
        """A host copy of the parameters in the backend's layout."""
        ...


class PolicyLoader(Protocol):
    """Load an admitted inference policy in the selected framework."""

    def load_policy(
        self,
        path: str | Path | PolicyCheckpoint,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> tuple[BackendPolicy, ObservationSpace]:
        """Read a portable policy and validate its world and input contract."""
        ...


class CheckpointReader(Protocol):
    """Inspect the saved metadata and tensors without creating a policy."""

    def read_policy(self, path: str) -> PolicyCheckpoint:
        """Read and bind the tensors without constructing a policy."""
        ...

    def metadata(self, path: str) -> dict[str, Any]:
        """Decode checkpoint metadata without constructing a policy."""
        ...


class Backend(PolicyLoader, CheckpointReader, Protocol):
    """Checkpoint admission needs the saved input contract and the loaded policy."""


class SimulationBackend(PolicyLoader, Protocol):
    def seed(self, seed: int, *, deterministic: bool | None = None) -> None:
        """Seed the backend's global action RNG; optionally set determinism."""
        ...


class SeededPolicyLoader(PolicyLoader, Protocol):
    def seeded_inference(
        self, seed: int, *, cpu_only: bool = False
    ) -> AbstractContextManager[object]: ...
