"""The MLX learner: optimizer state and the guarded update step.

`Learner` bundles an `MLXActorCritic` with its Adam optimizer (eps 1e-5, bias
correction, the rate set per iteration) and a profiler; the updates in
`manabot.training.updates` step it through `loss` and `step`, as they step the
torch backend's. It carries no loss of its own: an update hands `loss` its
minibatch objective as a closure over the policy forward, which MLX
differentiates as a function (`mlx.nn.value_and_grad`).

`loss` evaluates the objective and its gradients together, so the forward runs
once; `step` clips the global gradient norm, applies Adam and refuses
nonfinite results (`NumericalError`, `manabot.backends.mlx.rl`), and
`gradients` returns them without stepping. The learner runs in its policy's
precision (bfloat16 for wide models, float32 otherwise, or an explicit choice);
bfloat16 forwards keep float32 master weights, loss, clipping and Adam.
An optional recorder (`manabot.training.updates.health.OptimizerHealth`)
counts completed and rejected steps. On sampled steps it receives clipped
gradient measurements before Adam and parameter displacement after Adam.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
from numpy.typing import NDArray

from manabot.backends.mlx.arrays import all_finite, flat_numpy
from manabot.backends.mlx.rl import NumericalError, require
from manabot.infra.profiler import Profiler
from manabot.training.updates.health import (
    CRITIC_READOUT_PREFIXES,
    UNRECORDED_STEP,
    StepRecorder,
    StepSample,
    parameter_movement,
)

if TYPE_CHECKING:
    from manabot.backends.mlx.policy import MLXActorCritic
    from manabot.policy.checkpoint import ParameterArrays
    from manabot.policy.protocol import Precision


def check_objective(loss: mx.array) -> None:
    """Raise NumericalError for a nonfinite loss."""
    require(mx.isfinite(loss), "nonfinite_objective", loss=loss)


def no_grad() -> AbstractContextManager[object]:
    """Compute constants. MLX records no graph outside a differentiated
    function, so there is nothing to switch off."""
    return nullcontext()


def update_ema(ema: "MLXActorCritic", agent: "MLXActorCritic", decay: float) -> None:
    """Update `ema` weights in place toward `agent`."""
    if not 0 <= decay < 1:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
    updated = tree_map(
        lambda target, source: target * decay + source * (1.0 - decay),
        ema.parameters(),
        agent.parameters(),
    )
    ema.update(updated)
    ema.clear_cache()
    mx.eval(ema.parameters())


class Learner:
    """Trainable policy and Adam state shared by every update."""

    def __init__(
        self,
        agent: "MLXActorCritic",
        device: object = "gpu",
        *,
        minibatch_rows: int,
        precision: Precision | None = None,
        execution_rows: int | None = None,
        bootstrap_rows: int | None = None,
    ) -> None:
        if any(v is not None and v < 1 for v in (execution_rows, bootstrap_rows)):
            raise ValueError("execution row limits must be positive")
        self.execution_rows = execution_rows
        self.bootstrap_rows = bootstrap_rows
        if minibatch_rows < 1:
            raise ValueError("minibatch_rows must be positive")
        if precision is not None:
            agent.set_precision(precision)
        self.agent = agent
        self._require_masters("the MLX learner requires float32 master parameters")
        self.device = device
        # Cached programs are constants of the weights they were encoded from.
        agent.clear_cache()
        self.minibatch_rows = minibatch_rows
        self.optimizer = optim.Adam(
            learning_rate=2.5e-4, betas=[0.9, 0.999], eps=1e-5, bias_correction=True
        )
        self.profiler = Profiler()
        # The loss `loss` last returned and its parameter gradients.
        self._pending: tuple[mx.array, dict[str, object]] | None = None
        # Summed chunk gradients across execution chunks of one logical minibatch.
        self._accumulated: dict[str, object] | None = None

    def _require_masters(self, message: str) -> None:
        """Gradients and Adam apply to float32 masters; inside
        `forward_precision` the policy holds compute-dtype casts instead."""
        if self.agent.compute_dtype != mx.float32:
            raise ValueError(message)

    @property
    def precision(self) -> Precision:
        """The policy owns the precision used by both inference and learning."""
        return self.agent.precision

    def forward_precision(self) -> AbstractContextManager[object]:
        return self.agent.forward_precision()

    def set_learning_rate(self, rate: float) -> None:
        """The rate every following optimizer step uses."""
        self.optimizer.learning_rate = rate

    def loss(
        self, objective: Callable[[], tuple[mx.array, ...]]
    ) -> tuple[mx.array, ...]:
        """Evaluate `objective`, a closure over this policy's forward whose
        first result is the loss, with the loss's parameter gradients; return
        its results. `step` and `gradients` take that loss."""
        self._require_masters("close the inference precision scope before learning")
        # Cached inference programs have no gradient path to the current weights.
        self.agent.clear_cache()

        def float32_loss() -> tuple[mx.array, ...]:
            result = objective()
            return (result[0].astype(mx.float32), *result[1:])

        outputs, gradients = nn.value_and_grad(self.agent, float32_loss)()
        gradients = tree_map(lambda g: g.astype(mx.float32), gradients)
        mx.eval(outputs, gradients)
        self._pending = (outputs[0], gradients)
        return outputs

    def accumulate(self, loss: mx.array, *, first: bool) -> None:
        """Accumulate chunk gradients without stepping; clip and Adam stay at the logical boundary."""
        check_objective(loss)
        gradients = self._gradients(loss)
        self._pending = None
        if first:
            self._accumulated = gradients
        else:
            if self._accumulated is None:
                raise ValueError("accumulate(first=False) requires an active batch")
            self._accumulated = tree_map(mx.add, self._accumulated, gradients)
            mx.eval(self._accumulated)

    def _gradients(self, loss: mx.array) -> dict[str, object]:
        if self._pending is None or self._pending[0] is not loss:
            raise ValueError("step and gradients take the loss `loss` last returned")
        return self._pending[1]

    def step(
        self,
        loss: mx.array,
        max_grad_norm: float,
        *,
        recorder: StepRecorder | None = None,
        accumulated: bool = False,
    ) -> mx.array:
        """Clip the global gradient norm to `max_grad_norm` and apply Adam;
        return the pre-clip norm.

        Raises before stepping on a nonfinite loss or gradients; the policy
        and optimizer are then unchanged. Raises after stepping if a parameter
        became nonfinite.
        """
        self._require_masters("close the inference precision scope before stepping")
        if accumulated:
            if self._accumulated is None:
                raise ValueError(
                    "step(accumulated=True) requires accumulated gradients"
                )
            gradients = self._accumulated
            self._accumulated = None
        else:
            gradients = self._gradients(loss)
            self._pending = None
        recorder = recorder if recorder is not None else UNRECORDED_STEP
        with recorder.rejecting():
            check_objective(loss)
            with recorder.track("Backward and gradient clipping"):
                clipped, norm = optim.clip_grad_norm(gradients, max_grad_norm)
                finite = mx.isfinite(norm)
                sampling = recorder.sampling()
                # Adam replaces immutable arrays, so these references preserve
                # the pre-step weights without copying a second model.
                before = dict(tree_flatten(self.agent.parameters())) if sampling else {}
                sample = (
                    gradient_health(gradients, clipped, self.agent, before, finite)
                    if sampling
                    else None
                )
                if not bool(finite):
                    raise NumericalError(
                        "nonfinite_gradients",
                        {f"gradient/{name}": g for name, g in tree_flatten(gradients)},
                    )
                if sample is not None:
                    recorder.record_sample(sample)
            with recorder.track("Optimizer step"):
                self.optimizer.update(self.agent, clipped)
                self.agent.clear_cache()
                updated = self.agent.parameters()
                parameters = tree_flatten(updated)
                finite = all_finite(p for _, p in parameters)
                movement = (
                    (
                        squared_sums(p - before[name] for name, p in parameters),
                        squared_sums(before.values()),
                    )
                    if sampling
                    else ()
                )
                mx.eval(finite, updated, self.optimizer.state, movement)
                if not bool(finite):
                    raise NumericalError(
                        "nonfinite_parameters_after_step",
                        {f"parameter/{name}": p for name, p in parameters},
                    )
                if sampling:
                    delta, scale = (l2_norm(np.asarray(v)) for v in movement)
                    recorder.record_sample(parameter_movement(delta, scale))
                recorder.record_step()
        return norm

    def gradients(self, loss: mx.array) -> ParameterArrays:
        """Unclipped gradients of `loss` in MLX's layout, without stepping."""
        return flat_numpy(self._gradients(loss))

    def parameters(self) -> ParameterArrays:
        """A host copy of the parameters in MLX's layout."""
        return flat_numpy(self.agent.parameters())


def squared_sums(values: Iterable[mx.array]) -> mx.array:
    """One float32 sum of squares per leaf on the device."""
    return mx.stack([value.astype(mx.float32).square().sum() for value in values])


def l2_norm(squares: NDArray[np.float32]) -> float:
    return float(np.sqrt(squares.sum(dtype=np.float64)))


def gradient_health(
    gradients: dict[str, object],
    clipped: dict[str, object],
    agent: nn.Module,
    parameters: dict[str, mx.array],
    finite: mx.array,
) -> StepSample:
    """Sample at the clipping check. Only per-leaf scalars reach the host."""
    grads = dict(tree_flatten(gradients))
    clipped_grads = dict(tree_flatten(clipped))
    trainable = dict(tree_flatten(agent.trainable_parameters()))
    pre = squared_sums(grads.values())
    post = squared_sums(clipped_grads.values())
    # A tiny nonzero gradient can have a float32 square of zero.
    nonzero = mx.stack([mx.any(g != 0) for g in clipped_grads.values()])
    mx.eval(finite, pre, post, nonzero)
    squares = np.asarray(pre)
    critic = [
        i for i, name in enumerate(grads) if name.startswith(CRITIC_READOUT_PREFIXES)
    ]
    active = int(np.count_nonzero(np.asarray(nonzero)))
    sample: StepSample = {
        "gradient_norm_before": l2_norm(squares),
        "gradient_norm_after": l2_norm(np.asarray(post)),
        "gradient_norm_critic_readout": l2_norm(squares[critic]),
        "gradient_missing_tensors": len(trainable.keys() - grads.keys()),
        "gradient_zero_tensors": len(clipped_grads) - active,
        "gradient_nonzero_tensors": active,
        "frozen_parameter_tensors": len(parameters) - len(trainable),
    }
    if "value_token" in parameters:
        token = [i for i, name in enumerate(grads) if name == "value_token"]
        sample["gradient_norm_value_token"] = l2_norm(squares[token])
    return sample
