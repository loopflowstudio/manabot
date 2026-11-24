"""What an update calls on its backend, as Protocols.

The updates are written once over a configured backend or a directly imported
backend module passed as an argument. `UpdateBackend` is that module's
surface: its `rl`, `arrays`, `learner` and `rollout` files, each a Protocol a
module satisfies with plain functions. `Learner` and `Policy` are the objects
an update steps and runs forward. `manabot.policy.protocol.Backend` also
provides checkpoint admission. Parity supplies its own learner and layout contract.

Both backends satisfy `UpdateBackend`. The PPO update needs `PPOBackend`, which adds
standardization and a policy `__call__`/`get_value`; only
torch provides it. A categorical critic needs `CategoricalRL` (torch only);
`categorical_rl` narrows to it and refuses a backend without one. `ppo_rl`
likewise refuses an unsupported PPO update before reading rollout or learner state.
`tests/training/test_update_backend_protocols.py` checks each backend's files
against these member lists and signatures.

Nothing here imports a framework. `Array` is a backend's own array
(`torch.Tensor`, `mlx.core.array`), opaque to the update: it does arithmetic,
indexing and reductions on it and reads it through `arrays.to_numpy` or
`float()`. `Device` and `Observations` are likewise the backend's own and are
only handed back to it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.training.rollout.actions import ActionRows, ActionValidator
from manabot.training.updates.objective import Array, CategoricalValueLoss, LossTerms

if TYPE_CHECKING:
    from manabot.training.rollout.collection import RolloutBatch
    from manabot.training.updates.health import StepRecorder

# A backend's array, device and admitted observation minibatch. `Any` because
# the update applies the framework's own operators to arrays, and only the
# backend that produced a device or observation batch reads it.
type Device = Any
type Observations = Prepared[Array]

type FilterKind = Literal["top_count", "quantile"]


# -- rl ------------------------------------------------------------------------


@runtime_checkable
class RL(LossTerms, Protocol):
    """`backend.rl`: masks, likelihoods, returns, losses and selection over
    the backend's arrays. Shapes are in each backend's docstrings."""

    @property
    def NumericalError(self) -> type[Exception]:
        """What every check here raises on a broken invariant."""
        ...

    def require_finite(self, invariant: str, **arrays: Array) -> None: ...

    def transition_gae(
        self,
        rewards: Array,
        values: Array,
        ends: Array,
        next_value: Array,
        gamma: float,
        lam: float,
        bootstrap_indexes: np.ndarray | None = None,
    ) -> tuple[Array, Array]: ...

    def entropy(self, logs: Array, valid: Array) -> Array: ...

    def clip_fraction(self, log_ratio: Array, clip: float) -> float: ...

    def selected_magnitudes(
        self, advantages: Array, kind: FilterKind, fraction: float, minimum: float
    ) -> tuple[Array, float | None]: ...


@runtime_checkable
class CategoricalRL(RL, CategoricalValueLoss, Protocol):
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
        bootstrap_indexes: np.ndarray | None = None,
    ) -> Array: ...


def categorical_rl(rl: RL) -> CategoricalRL:
    """`rl` as a categorical critic's functions; refuses a backend without them."""
    if not isinstance(rl, CategoricalRL):
        name = getattr(rl, "__name__", type(rl).__name__)
        raise TypeError(f"{name} has no categorical critic")
    return rl


@runtime_checkable
class PPORL(CategoricalRL, Protocol):
    """`rl` of a backend the PPO update runs on."""

    def standardized_advantages(
        self, advantages: Array, over: Array | None = None
    ) -> Array: ...


def ppo_rl(rl: RL) -> PPORL:
    """Refuse an unsupported update before reading its rollout or learner."""
    if not isinstance(rl, PPORL):
        name = getattr(rl, "__name__", type(rl).__name__)
        raise TypeError(f"{name} has no PPO update")
    return rl


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

    def next_observation(self, device: Device) -> Observations | None: ...
    def next_observations(
        self, device: Device, size: int
    ) -> Iterator[Observations]: ...

    def rows(
        self, indexes: NDArray[np.int64], device: Device
    ) -> tuple[Observations, ActionRows]: ...

    def legal(self, obs: Observations) -> Array: ...

    def evidence(self, obs: Observations) -> Mapping[str, Array]: ...


class RolloutFile(ActionValidator, Protocol):
    """`backend.rollout`. The first argument is the learner's `agent`."""

    @property
    def AdmittedRollout(self) -> Callable[[Any, RolloutBatch], AdmittedRollout]: ...


class PPOAdmittedRollout(AdmittedRollout, Protocol):
    def minibatches(
        self, order: NDArray[np.int64], size: int, device: Device
    ) -> AbstractContextManager[
        Iterator[tuple[NDArray[np.int64], Observations, ActionRows]]
    ]:
        """Ordered minibatches; exit releases any preparation worker."""
        ...


class PPORolloutFile(RolloutFile, Protocol):
    @property
    def AdmittedRollout(self) -> Callable[[Any, RolloutBatch], PPOAdmittedRollout]: ...


# -- policy and learner --------------------------------------------------------


class PolicyConfig(Protocol):
    @property
    def value_kind(self) -> str: ...


class Policy(Protocol):
    """The trainable policy: the training forward as the backend's arrays,
    logits [B,A] and the critic's raw output ([B,1], or [B,3] outcome logits)."""

    @property
    def config(self) -> PolicyConfig: ...

    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]: ...

    def forward_distribution(self, obs: Observations, /) -> tuple[Array, Array]: ...


class PPOPolicy(Policy, Protocol):
    def __call__(self, obs: Observations, /) -> tuple[Array, Array]:
        """Logits [B,A] and scalar values [B]."""
        ...

    def get_value(self, obs: Observations, /) -> Array: ...


class Learner(Protocol):
    """The policy, its optimizer and profiler, as an update steps them."""

    execution_rows: int | None
    bootstrap_rows: int | None

    def set_learning_rate(self, rate: float) -> None: ...
    def loss(self, objective: Callable[[], tuple[Array, ...]]) -> tuple[Array, ...]: ...

    def accumulate(self, loss: Array, *, first: bool) -> None: ...

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

    def step(
        self,
        loss: Array,
        max_grad_norm: float,
        *,
        recorder: StepRecorder | None = None,
        accumulated: bool = False,
    ) -> Array:
        """One clipped optimizer step on `loss`: the pre-clip gradient norm.
        `recorder` is the update's `OptimizerHealth`, which each backend
        reads through the shared `StepRecorder`."""
        ...


class PPOLearner(Learner, Protocol):
    @property
    def agent(self) -> PPOPolicy: ...


# -- the backend module --------------------------------------------------------


class UpdateBackend(Protocol):
    """A backend module as the damped update takes it."""

    @property
    def rl(self) -> RL: ...

    @property
    def arrays(self) -> Arrays: ...

    @property
    def learner(self) -> LearnerFile: ...

    @property
    def rollout(self) -> RolloutFile: ...


class PPOBackend(UpdateBackend, Protocol):
    @property
    def rl(self) -> PPORL: ...

    @property
    def rollout(self) -> PPORolloutFile: ...
