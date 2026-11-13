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
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from manabot.env.observation import ObservationBuffers
from manabot.policy.checkpoint import ParameterArrays, PolicyCheckpoint
from manabot.policy.prepare import Prepared


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


class BackendPolicy(Protocol):
    def prepare(self, obs: ObservationBuffers, *, inference: bool = False) -> Prepared:
        """Index one admitted observation batch (`inference`: compact as
        PyTorch CPU inference does)."""
        ...

    def evaluate(self, batch: Prepared) -> tuple[np.ndarray, np.ndarray]:
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

    def gradients(self, loss: Any) -> dict[str, np.ndarray]:
        """Unclipped gradients of `loss` in the backend's layout, without
        stepping."""
        ...

    def parameters(self) -> dict[str, np.ndarray]:
        """A host copy of the parameters in the backend's layout."""
        ...


class Backend(Protocol):
    name: str

    def load(
        self, checkpoint: PolicyCheckpoint
    ) -> tuple[BackendPolicy, BackendLearner]:
        """A policy holding the checkpoint's weights, and a fresh learner on it."""
        ...

    def parameter_layout(self, canonical: ParameterArrays) -> dict[str, np.ndarray]:
        """Canonical parameters (or deltas: the map is linear) in the
        backend's names and layout, as `BackendLearner.parameters` returns them."""
        ...

    def gradient_layout(self, canonical: ParameterArrays) -> dict[str, np.ndarray]:
        """Gradients of canonical parameters as gradients of the backend's,
        as `BackendLearner.gradients` returns them."""
        ...
