"""Local numerical health, independent of tracker availability and RNG.

Batch measurements cover every collected row. A learner reports each optimizer
step through `StepRecorder`. The player's `OptimizerHealth` samples gradient
norms and parameter displacement on the first step of iteration 1 and every
25th iteration. Finiteness checks run on every step.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from manabot.infra.profiler import Profiler

if TYPE_CHECKING:
    from manabot.training.rollout.actions import PackedActions
    from manabot.training.updates.backend import UpdateBackend

HEALTH_INTERVAL = 25
# Only these policy parameters are exclusive to the critic in every aggregation.
CRITIC_READOUT_PREFIXES = ("value_head.", "value_readout.")


class StepSample(TypedDict, total=False):
    """Measurements supplied by a learner as they become available during a step."""

    gradient_norm_before: float
    gradient_norm_after: float
    # Pre-clip norm over critic-only parameters (value_head, value_readout).
    # The rest of gradient_norm_before is shared encoder and actor; encoder
    # gradients from the value loss are not separated.
    gradient_norm_critic_readout: float
    # Pre-clip norm of the value token, present only in the value_token
    # aggregation. Actor and critic share it: policy and value loss both reach it.
    gradient_norm_value_token: float
    gradient_missing_tensors: int
    gradient_zero_tensors: int
    gradient_nonzero_tensors: int
    frozen_parameter_tensors: int
    parameter_delta_l2: float
    parameter_relative_delta: float


class StepRecorder(Protocol):
    """Observe optimizer steps without changing the loss, step or RNG.

    `track(section)` times "Backward and gradient clipping" and "Optimizer
    step". Each ends at a finite check that already waits for the device.
    `rejecting()` wraps the whole step, so it sees every `NumericalError`.
    On a sampled step the learner reports clipped gradient measurements before
    Adam, then parameter displacement after its finite check. Every completed
    step calls `record_step`, including those without a sample.
    """

    def track(self, section: str) -> AbstractContextManager[object]: ...

    def rejecting(self) -> AbstractContextManager[object]: ...

    def sampling(self) -> bool: ...

    def record_sample(self, sample: StepSample) -> None: ...

    def record_step(self) -> None: ...


class UnrecordedStep:
    """A step without an observer performs no timing or sampling."""

    def track(self, section: str) -> AbstractContextManager[object]:
        return nullcontext()

    def rejecting(self) -> AbstractContextManager[object]:
        return nullcontext()

    def sampling(self) -> bool:
        return False

    def record_sample(self, sample: StepSample) -> None:
        pass

    def record_step(self) -> None:
        pass


UNRECORDED_STEP: StepRecorder = UnrecordedStep()


def parameter_movement(delta: float, scale: float) -> StepSample:
    """A step's L2 displacement, relative to the pre-step weights when they are nonzero."""
    return {
        "parameter_delta_l2": delta,
        "parameter_relative_delta": delta / scale if scale > 0 else delta,
    }


class NumericalHealth(StepSample, total=False):
    legal_logit_gap_max: float
    legal_probability_min: float
    legal_underflow_count: int
    nonfinite_count: int
    forced_rows: int
    zero_advantage_rows: int
    actor_rows: int
    critic_rows: int
    empty_actor_batch: int
    optimizer_steps: int
    skipped_steps: int
    rejected_steps: int
    sampled_steps: int
    clip_fraction: float


def batch_health(
    action_data: "PackedActions",
    advantages: NDArray[np.floating],
    selected: NDArray[np.bool_],
    *,
    actor_only: bool,
) -> NumericalHealth:
    """Measurements over every collected row; `advantages` and `selected` are
    host rows of any one shape."""
    logs = action_data.log_probabilities
    probabilities = np.exp(logs)
    starts = action_data.offsets[:-1]
    gaps = np.maximum.reduceat(logs, starts) - np.minimum.reduceat(logs, starts)
    return NumericalHealth(
        legal_logit_gap_max=float(gaps.max()),
        legal_probability_min=float(probabilities.min()),
        legal_underflow_count=int((probabilities == 0).sum()),
        nonfinite_count=int((~np.isfinite(logs)).sum()),
        forced_rows=int((np.diff(action_data.offsets) == 1).sum()),
        zero_advantage_rows=int((advantages == 0).sum()),
        actor_rows=int(selected.sum()),
        critic_rows=selected.size if actor_only else int(selected.sum()),
        empty_actor_batch=int(not selected.any()),
        optimizer_steps=0,
        skipped_steps=0,
        rejected_steps=0,
        sampled_steps=0,
    )


class OptimizerHealth:
    """Record what `gradient_step` does to one iteration's `NumericalHealth`.

    Counts steps and rejections on every step. Detailed gradient and parameter
    measurements sample the first step of iteration 1 and every 25th
    iteration. Recording never changes the loss, the step or any RNG.

    `check` rejects a detached or nonfinite objective; it waits for the loss on
    the device, so a caller timing its forward pass ends that timer there.
    `backend` supplies the objective check and numerical refusal type. Its
    learner measures gradients and parameter changes.
    With a profiler, `gradient_step` times backward and the optimizer step
    separately; each ends at a finite check that already waits for the device,
    so timing adds no synchronization.
    """

    def __init__(
        self,
        health: NumericalHealth,
        iteration: int,
        profiler: Profiler | None = None,
        *,
        backend: "UpdateBackend",
    ) -> None:
        self.health = health
        self._check_objective: Callable[[Any], None] = backend.learner.check_objective
        self._refusal: type[Exception] = backend.rl.NumericalError
        self.measure = iteration == 1 or iteration % HEALTH_INTERVAL == 0
        self.profiler = profiler or Profiler(enabled=False)

    def track(self, section: str) -> AbstractContextManager[object]:
        return self.profiler.track(section)

    def check(self, loss: Any) -> None:
        with self.rejecting():
            self._check_objective(loss)

    @contextmanager
    def rejecting(self) -> Iterator[None]:
        """Count a NumericalError as a rejected step and attach the record."""
        try:
            yield
        except self._refusal as error:
            self.health["rejected_steps"] += 1
            error.health = dict(self.health)  # type: ignore[attr-defined]
            raise

    def sampling(self) -> bool:
        """Whether the coming step is this iteration's detailed sample."""
        return self.measure and not self.health["sampled_steps"]

    def record_sample(self, sample: StepSample) -> None:
        self.health.update(sample)
        self.health["sampled_steps"] = 1

    def record_step(self) -> None:
        self.health["optimizer_steps"] += 1


@contextmanager
def failure_tensors(
    tensors: Callable[[], Mapping[str, Any]],
    refusal: type[Exception],
) -> Iterator[None]:
    """Attach `tensors()` to a `refusal` (the backend's `rl.NumericalError`)
    raised inside, then re-raise it.

    The callable runs only on failure, so a healthy step builds no evidence.
    """
    try:
        yield
    except refusal as error:
        error.tensors.update(tensors())  # type: ignore[attr-defined]
        raise


def minibatch_tensors(
    *,
    logits: Any,
    value_logits: Any,
    valid: Any,
    actions: Any,
    behavior_logs: Any,
    reference: Any,
    advantages: Any,
    value_targets: Any,
    actor_selected: Any,
    indexes: Any,
    obs: Mapping[str, Any],
) -> dict[str, Any]:
    """The evidence a rejected update step retains: its minibatch and outputs,
    as the backend's arrays. `logits` and `value_logits` arrive as constants."""
    return {
        "logits": logits,
        "value_logits": value_logits,
        "valid": valid,
        "actions": actions,
        "behavior_logs": behavior_logs,
        "reference": reference,
        "advantages": advantages,
        "value_targets": value_targets,
        "actor_selected": actor_selected,
        "minibatch_indexes": indexes,
        **{f"minibatch_observation/{key}": value for key, value in obs.items()},
    }
