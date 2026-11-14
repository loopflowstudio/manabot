"""What an update calls on its backend, as Protocols.

The updates are written once over a backend module (`manabot.backends.torch`,
`manabot.backends.mlx`) passed as an argument. `Backend` is that module's
surface: its `rl`, `arrays`, `learner` and `rollout` files, each a Protocol a
module satisfies with plain functions. `Learner` and `Policy` are the objects
an update steps and runs forward. `manabot.policy.protocol` is the narrower
NumPy surface parity compares; `Learner` extends its `BackendLearner`.

Both backends satisfy `Backend`. The PPO update needs `PPOBackend`, which adds
standardization, top-count selection and a policy `__call__`/`get_value`; only
torch provides it. A categorical critic needs `CategoricalRL` (torch only);
`categorical_rl` narrows to it and refuses a backend without one.
`tests/training/test_update_backend_protocols.py` checks each backend's files
against these member lists and signatures.

Nothing here imports a framework. `Array` is a backend's own array
(`torch.Tensor`, `mlx.core.array`), opaque to the update: it does arithmetic,
indexing and reductions on it and reads it through `arrays.to_numpy` or
`float()`. `Device` and `Observations` are likewise the backend's own and are
only handed back to it.
"""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from manabot.infra.profiler import Profiler
from manabot.policy.protocol import BackendLearner
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import RolloutBatch

# A backend's array, device and admitted observation minibatch. `Any` because
# the update applies the framework's own operators to arrays, and only the
# backend that produced a device or observation batch reads it.
type Array = Any
type Device = Any
type Observations = Any

type FilterKind = Literal["top_count", "quantile"]


# -- rl ------------------------------------------------------------------------


@runtime_checkable
class RL(Protocol):
    """`backend.rl`: masks, likelihoods, returns, losses and selection over
    the backend's arrays. Shapes are in each backend's docstrings."""

    @property
    def NumericalError(self) -> type[Exception]:
        """What every check here raises on a broken invariant."""
        ...

    def require_finite(self, invariant: str, **arrays: Array) -> None: ...

    def float64_operand(self, tensor: Array) -> Array: ...

    def constant(self, tensor: Array) -> Array: ...

    def legal_mask(self, mask: Array) -> Array: ...

    def policy_logs(self, logits: Array, valid: Array) -> Array: ...

    def action_logs(self, logs: Array, actions: Array) -> Array: ...

    def reference_logs(self, reference: Array, valid: Array) -> Array: ...

    def transition_gae(
        self,
        rewards: Array,
        values: Array,
        ends: Array,
        next_value: Array,
        gamma: float,
        lam: float,
    ) -> tuple[Array, Array]: ...

    def clipped_surrogate(
        self, log_ratio: Array, advantages: Array, clip: float
    ) -> Array: ...

    def reverse_kl(self, logs: Array, target_logs: Array, valid: Array) -> Array: ...

    def squared_value_loss(self, values: Array, targets: Array) -> Array: ...

    def entropy(self, logs: Array, valid: Array) -> Array: ...

    def damped_policy_loss(
        self,
        logits: Array,
        actions: Array,
        advantages: Array,
        behavior_logs: Array,
        reference: Array,
        valid: Array,
        *,
        clip: float,
        collection_kl: float,
        tau: float,
    ) -> Array: ...

    def clip_fraction(self, log_ratio: Array, clip: float) -> float: ...

    def selection_mask(
        self, advantages: Array, kind: FilterKind, fraction: float, minimum: float
    ) -> Array: ...

    def selected_mean(self, values: Array, selected: Array) -> Array: ...


@runtime_checkable
class CategoricalRL(RL, Protocol):
    """`rl` of a backend with a win/draw/loss critic."""

    def expected_outcome(self, probabilities: Array) -> Array: ...

    def outcome_probabilities(self, value_logits: Array) -> Array: ...

    def categorical_lambda_returns(
        self,
        probabilities: Array,
        rewards: Array,
        ends: Array,
        bootstrap: Array,
        trace_lambda: float,
    ) -> Array: ...

    def categorical_value_loss(self, value_logits: Array, targets: Array) -> Array: ...


def categorical_rl(rl: RL) -> CategoricalRL:
    """`rl` as a categorical critic's functions; refuses a backend without them."""
    if not isinstance(rl, CategoricalRL):
        name = getattr(rl, "__name__", type(rl).__name__)
        raise TypeError(f"{name} has no categorical critic")
    return rl


class PPORL(CategoricalRL, Protocol):
    """`rl` of a backend the PPO update runs on."""

    def standardized_advantages(
        self, advantages: Array, over: Array | None = None
    ) -> Array: ...

    def selected_rows(
        self,
        advantages: Array,
        fraction: float,
        minimum: float,
        *,
        kind: FilterKind = "top_count",
    ) -> Array: ...


# -- arrays, learner and rollout files ------------------------------------------


class Arrays(Protocol):
    """`backend.arrays`: moving arrays between the host and the backend."""

    def from_numpy(self, array: np.ndarray, device: Device = ..., /) -> Array: ...

    def to_numpy(self, array: Array, /) -> np.ndarray: ...

    def float32(self, array: Array, /) -> Array: ...


class LearnerFile(Protocol):
    """`backend.learner`: the objective check and constant computation."""

    def check_objective(self, loss: Array) -> None:
        """Raise `rl.NumericalError` for a detached or nonfinite loss."""
        ...

    def no_grad(self) -> AbstractContextManager[object]:
        """Nothing computed inside records a gradient."""
        ...


class AdmittedRollout(Protocol):
    """A rollout batch admitted once for one policy (`backend.rollout`)."""

    @property
    def batch(self) -> RolloutBatch: ...

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None: ...

    def next_observation(self, device: Device) -> Observations: ...

    def rows(
        self, indexes: NDArray[np.int64], device: Device
    ) -> tuple[Observations, ActionRows]: ...

    def legal(self, obs: Observations) -> Array: ...

    def evidence(self, obs: Observations) -> Mapping[str, Array]: ...


class RolloutFile(Protocol):
    """`backend.rollout`. The first argument is the learner's `agent`."""

    @property
    def AdmittedRollout(self) -> Callable[[Any, RolloutBatch], AdmittedRollout]: ...


# -- policy and learner --------------------------------------------------------


class PolicyConfig(Protocol):
    @property
    def value_kind(self) -> str: ...


class Policy(Protocol):
    """The trainable policy: the training forward as the backend's arrays,
    logits [B,A] and the critic's raw output ([B,1], or [B,3] outcome logits)."""

    @property
    def config(self) -> PolicyConfig: ...

    def forward_distribution(self, obs: Observations, /) -> tuple[Array, Array]: ...


class PPOPolicy(Policy, Protocol):
    def __call__(self, obs: Observations, /) -> tuple[Array, Array]:
        """Logits [B,A] and scalar values [B]."""
        ...

    def get_value(self, obs: Observations, /) -> Array: ...


class Learner(BackendLearner, Protocol):
    """The policy, its optimizer and profiler, as an update steps them."""

    @property
    def agent(self) -> Policy: ...

    @property
    def device(self) -> Device: ...

    @property
    def minibatch_rows(self) -> int: ...

    @property
    def profiler(self) -> Profiler: ...

    def forward_precision(self) -> AbstractContextManager[object]:
        """Wrap each policy forward; callers cast its outputs to float32."""
        ...

    def step(self, loss: Array, max_grad_norm: float, *, recorder: Any = None) -> Array:
        """One clipped optimizer step on `loss`: the pre-clip gradient norm.
        `recorder` is the update's `OptimizerHealth`, which each backend
        reads through its own `StepRecorder`."""
        ...


class PPOLearner(Learner, Protocol):
    @property
    def agent(self) -> PPOPolicy: ...


# -- the backend module --------------------------------------------------------


class Backend(Protocol):
    """A backend module as the damped update takes it."""

    @property
    def rl(self) -> RL: ...

    @property
    def arrays(self) -> Arrays: ...

    @property
    def learner(self) -> LearnerFile: ...

    @property
    def rollout(self) -> RolloutFile: ...


class PPOBackend(Backend, Protocol):
    @property
    def rl(self) -> PPORL: ...
