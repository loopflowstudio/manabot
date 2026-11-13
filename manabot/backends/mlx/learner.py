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
`gradients` returns them without stepping. Float32 is the default; explicit
bfloat16/float16 forwards keep float32 master weights, loss, clipping and Adam.
Float16 uses fixed loss scaling (128); overflow is refused, never skipped.
An optional recorder (`manabot.training.updates.health.OptimizerHealth`)
counts steps and rejections; the sampled gradient and parameter detail it
reads from torch modules stays absent here.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from manabot.backends.mlx.arrays import flat_numpy
from manabot.backends.mlx.rl import NumericalError, require
from manabot.infra.profiler import Profiler
from manabot.policy.checkpoint import ParameterArrays
from manabot.policy.protocol import Precision

if TYPE_CHECKING:
    from manabot.backends.mlx.policy import MLXActorCritic


class StepRecorder(Protocol):
    """What `Learner.step` reports: its timed sections, every refusal and each
    completed step. Recording never changes the loss or the step."""

    def track(self, section: str) -> AbstractContextManager[object]: ...

    def rejecting(self) -> AbstractContextManager[object]: ...

    def record_step(self, parameters: list[object], before: list[object]) -> None: ...


def check_objective(loss: mx.array) -> None:
    """Raise NumericalError for a nonfinite loss."""
    require(mx.isfinite(loss), "nonfinite_objective", loss=loss)


def no_grad() -> AbstractContextManager[object]:
    """Compute constants. MLX records no graph outside a differentiated
    function, so there is nothing to switch off."""
    return nullcontext()


class Learner:
    """Trainable policy and Adam state shared by every update."""

    def __init__(
        self,
        agent: "MLXActorCritic",
        device: object = "gpu",
        *,
        minibatch_rows: int,
        precision: Precision | None = None,
    ) -> None:
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

    def _require_masters(self, message: str) -> None:
        """Gradients and Adam apply to float32 masters; inside
        `forward_precision` the policy holds compute-dtype casts instead."""
        if self.agent.compute_dtype != mx.float32:
            raise ValueError(message)

    @property
    def precision(self) -> Precision:
        """The policy owns the precision used by both inference and learning."""
        return self.agent.precision

    @property
    def loss_scale(self) -> float:
        return 128.0 if self.precision == "float16" else 1.0

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
        scale = self.loss_scale

        def scaled() -> tuple[mx.array, ...]:
            result = objective()
            return (result[0].astype(mx.float32) * scale, *result[1:])

        outputs, gradients = nn.value_and_grad(self.agent, scaled)()
        outputs = (outputs[0] / scale, *outputs[1:])
        gradients = tree_map(lambda g: g.astype(mx.float32) / scale, gradients)
        mx.eval(outputs, gradients)
        self._pending = (outputs[0], gradients)
        return outputs

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
    ) -> mx.array:
        """Clip the global gradient norm to `max_grad_norm` and apply Adam;
        return the pre-clip norm.

        Raises before stepping on a nonfinite loss or gradients; the policy
        and optimizer are then unchanged. Raises after stepping if a parameter
        became nonfinite.
        """
        self._require_masters("close the inference precision scope before stepping")
        gradients = self._gradients(loss)
        self._pending = None
        track = recorder.track if recorder is not None else _untracked
        with recorder.rejecting() if recorder is not None else nullcontext():
            check_objective(loss)
            with track("Backward and gradient clipping"):
                clipped, norm = optim.clip_grad_norm(gradients, max_grad_norm)
                if not bool(mx.isfinite(norm)):
                    raise NumericalError(
                        "nonfinite_gradients",
                        {f"gradient/{name}": g for name, g in tree_flatten(gradients)},
                    )
            with track("Optimizer step"):
                self.optimizer.update(self.agent, clipped)
                self.agent.clear_cache()
                parameters = tree_flatten(self.agent.parameters())
                finite = mx.all(
                    mx.isfinite(mx.concatenate([p.flatten() for _, p in parameters]))
                )
                mx.eval(finite, self.agent.parameters(), self.optimizer.state)
                if not bool(finite):
                    raise NumericalError(
                        "nonfinite_parameters_after_step",
                        {f"parameter/{name}": p for name, p in parameters},
                    )
                if recorder is not None:
                    recorder.record_step([], [])
        return norm

    def gradients(self, loss: mx.array) -> ParameterArrays:
        """Unclipped gradients of `loss` in MLX's layout, without stepping."""
        return flat_numpy(self._gradients(loss))

    def parameters(self) -> ParameterArrays:
        """A host copy of the parameters in MLX's layout."""
        return flat_numpy(self.agent.parameters())


def _untracked(section: str) -> AbstractContextManager[object]:
    return nullcontext()
