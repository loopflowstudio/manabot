"""What a framework backend provides: a policy, a learner and a parameter layout.

A backend (`manabot.backends.<name>`) runs the policy forward and the damped
PPO learner step in one array framework, over the `Prepared` batches of
`manabot.policy.prepare`. Every method here takes and returns NumPy, so
callers such as `manabot.backends.parity` never touch the framework's arrays.
`Targets` and `Settings` are the learner's per-minibatch inputs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from manabot.policy.checkpoint import PolicyCheckpoint
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
    def prepare(
        self, obs: Mapping[str, np.ndarray], *, inference: bool = False
    ) -> Prepared:
        """Index one admitted observation batch (`inference`: compact as
        PyTorch CPU inference does)."""
        ...

    def evaluate(self, batch: Prepared) -> tuple[np.ndarray, np.ndarray]:
        """Logits [B,A] (-1e8 off the legal support) and values [B]."""
        ...

    def cache_programs(self) -> None:
        """Encode weight-only inputs once, for inference with fixed weights."""
        ...

    def clear_cache(self) -> None:
        """Drop what `cache_programs` encoded; required before weights change."""
        ...


class BackendLearner(Protocol):
    """One optimizer over a policy's parameters, in the backend's layout."""

    def gradients(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Loss and unclipped gradients, without stepping."""
        ...

    def step(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[float, float]:
        """One clipped optimizer step: loss and pre-clip gradient norm."""
        ...

    def parameters(self) -> dict[str, np.ndarray]: ...


class Backend(Protocol):
    name: str

    def load(
        self, checkpoint: PolicyCheckpoint
    ) -> tuple[BackendPolicy, BackendLearner]:
        """A policy holding the checkpoint's weights, and a fresh learner on it."""
        ...

    def parameter_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Canonical parameters (or deltas: the map is linear) in the
        backend's names and layout, as `BackendLearner.parameters` returns them."""
        ...

    def gradient_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Gradients of canonical parameters as gradients of the backend's,
        as `BackendLearner.gradients` returns them."""
        ...
