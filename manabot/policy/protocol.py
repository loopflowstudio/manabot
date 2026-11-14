"""What a framework backend provides: a policy, a learner and a parameter layout.

A backend (`manabot.backends.<name>`) runs the policy forward and an update's
optimizer step in one array framework, over the `Prepared` batches of
`manabot.policy.prepare`. `BackendPolicy.prepare` and `evaluate` and every
`Backend` method take and return NumPy, so callers such as
`manabot.backends.parity` compare backends without touching their arrays.
`BackendLearner` is what an update steps: it differentiates a loss written
once over the backend's own arrays (`manabot.training.updates`). `Targets`
and `Settings` are one captured minibatch's constants and coefficients.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self

import numpy as np

from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.policy.checkpoint import ParameterArrays, PolicyCheckpoint
from manabot.policy.prepare import Prepared

Precision = Literal["float32", "bfloat16"]


@dataclass(frozen=True)
class Targets:
    """Constants of one minibatch: sampled actions [B], advantages [B], value
    targets [B], actor-selected rows [B], behavior log-likelihoods [B,A]
    (-inf off support) and reference probabilities [B,A]."""

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


@dataclass(frozen=True)
class InferenceResult:
    """A completed host inference and the input transfer it required."""

    logits: np.ndarray
    transfer_bytes: int = 0
    transfers: int = 0


class InferenceScope(Protocol):
    def infer(
        self,
        obs: ObservationBuffers,
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

    def probability_from_logit(self, value: float) -> float: ...

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

    def win_probabilities(self, batch: Prepared[np.ndarray]) -> np.ndarray:
        """Sigmoid of scalar win-logit values, evaluated in the backend."""
        ...

    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]:
        """Index one admitted observation batch (`inference`: compact as
        PyTorch CPU inference does)."""
        ...

    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Logits [B,A] (-1e8 off the legal support) and values [B]."""
        ...

    def forward_distribution(self, batch: Prepared) -> tuple[Any, Any]:
        """The training forward as the backend's own arrays, differentiable:
        logits [B,A] and the critic's output [B,1]."""
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


class Backend(Protocol):
    name: str

    def load_policy(
        self, path: str, *, bucketed_shapes: bool = False, live_tokens: bool = False
    ) -> tuple[BackendPolicy, ObservationSpace]:
        """Read a portable policy and validate its world and input contract."""
        ...

    def metadata(self, path: str) -> dict[str, Any]:
        """Decode checkpoint metadata without constructing a policy."""
        ...

    def seed(self, seed: int) -> None:
        """Seed the backend's global action RNG at an explicit decision boundary."""
        ...

    def load(
        self, checkpoint: PolicyCheckpoint
    ) -> tuple[BackendPolicy, BackendLearner]:
        """A policy holding the checkpoint's weights, and a fresh learner on it."""
        ...

    def parameter_layout(self, canonical: ParameterArrays) -> ParameterArrays:
        """Canonical parameters (or deltas: the map is linear) in the
        backend's names and layout, as `BackendLearner.parameters` returns them."""
        ...

    def gradient_layout(self, canonical: ParameterArrays) -> ParameterArrays:
        """Gradients of canonical parameters as gradients of the backend's,
        as `BackendLearner.gradients` returns them."""
        ...
