"""The torch learner: optimizer state, precision and the guarded update step.

`Learner` bundles a trainable `ActorCritic` with its Adam optimizer, its
forward precision and a profiler; the updates in `manabot.training.updates`
step it through `Learner.loss` and `Learner.step`, the two calls every
backend's learner has. `gradient_step` runs backward, global-norm clipping and the
optimizer for any `nn.Module`, refusing to step on a loss or gradients it
cannot trust (`NumericalError`, `manabot.backends.torch.rl`); an optional
`StepRecorder` (the player's `OptimizerHealth` in
`manabot.training.updates.health` is one) observes what it did. `update_ema`
keeps the averaged model that evaluation and exports read. `forward_precision` and
`apply_matmul_precision` apply a stage's declared numeric precision;
`no_grad` wraps work whose outputs are constants (targets, diagnostics).
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from manabot.backends.torch.rl import NumericalError, float64_operand, require
from manabot.infra.profiler import Profiler
from manabot.training.updates.health import (
    CRITIC_READOUT_PREFIXES,
    UNRECORDED_STEP,
    parameter_movement,
)

if TYPE_CHECKING:
    # Annotations only: the update step and EMA serve any model, so importing
    # them loads neither the player's network nor the training configuration.
    from manabot.backends.torch.policy import ActorCritic
    from manabot.policy.checkpoint import ParameterArrays
    from manabot.training.learning import Precision
    from manabot.training.updates.health import StepRecorder, StepSample

# -- Update step ---------------------------------------------------------------


def check_objective(loss: Tensor) -> None:
    """Raise NumericalError for a loss that is detached or nonfinite."""
    if not loss.requires_grad:
        raise NumericalError("detached_objective", {"loss": loss})
    require(torch.isfinite(loss), "nonfinite_objective", loss=loss)


def gradient_step(
    loss: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    *,
    recorder: "StepRecorder | None" = None,
    accumulated: bool = False,
) -> Tensor:
    """Zero, backward, clip the global norm to `max_grad_norm`, step; return the pre-clip norm.

    Raises before stepping on a detached or nonfinite loss, a loss that reaches
    no parameter, or nonfinite gradients; the model and optimizer are then
    unchanged. Raises after stepping if a parameter became nonfinite.
    `accumulated` preserves gradients already backpropagated by execution chunks;
    clipping and Adam still occur exactly once at the logical boundary.
    """
    recorder = recorder if recorder is not None else UNRECORDED_STEP
    with recorder.rejecting():
        check_objective(loss)
        parameters = list(model.parameters())
        if not accumulated:
            optimizer.zero_grad(set_to_none=True)
        with recorder.track("Backward and gradient clipping"):
            if not accumulated:
                loss.backward()
            if not any(p.grad is not None for p in parameters):
                raise NumericalError("missing_all_gradients", {"loss": loss})
            critic_norms = critic_gradient_norms(model) if recorder.sampling() else None
            try:
                # error_if_nonfinite reads the norm on the host: the device wait
                # that ends the backward timer.
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise NumericalError(
                    "nonfinite_gradients",
                    {
                        f"gradient/{name}": p.grad
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    },
                ) from error
            if critic_norms is not None:
                recorder.record_sample(gradient_health(model, norm, critic_norms))
        with recorder.track("Optimizer step"):
            before = (
                [p.detach().clone() for p in parameters]
                if critic_norms is not None
                else []
            )
            optimizer.step()
            # One flat read: per-tensor checks cost two launches per parameter tensor.
            if not torch.isfinite(
                torch.cat([p.detach().flatten() for p in parameters])
            ).all():
                raise NumericalError(
                    "nonfinite_parameters_after_step",
                    {f"parameter/{name}": p for name, p in model.named_parameters()},
                )
            if before:
                recorder.record_sample(parameter_health(parameters, before))
            recorder.record_step()
    return norm


@torch.no_grad()
def update_ema(averaged: nn.Module, model: nn.Module, rate: float) -> None:
    """Move `averaged`'s parameters toward `model`'s by `1 - rate`; copy buffers.

    Initialize with `deepcopy(model)`. Regimes advance it once per
    collection/update iteration, including filtered skips. Buffers describe the
    current model and are copied exactly. Which of the two acts is the
    caller's choice; this changes neither.
    """
    if not 0 <= rate < 1:
        raise ValueError("EMA rate must be in [0, 1)")
    for name, dest in averaged.named_parameters():
        dest.lerp_(model.get_parameter(name), 1 - rate)
    for name, dest in averaged.named_buffers():
        dest.copy_(model.get_buffer(name))


# -- Precision -----------------------------------------------------------------
# Numeric precision the settings declare (`SelfPlay.precision`).
#
# - `float32`: nothing changes; every helper here is a no-op.
# - `tf32`: for the stage, CUDA matmul and cuDNN may use TF32 tensor cores
#   (`allow_tf32`). Weights, activations and losses stay float32 tensors.
# - `bf16`: `torch.autocast(device, bfloat16)` on CUDA or MPS around the policy
#   forwards only, in collection (behavior sampling and stored values) and in the
#   learner (bootstrap targets and the loss forward). Callers cast forward outputs
#   back to float32, so behavior log-probabilities, targets, losses and their
#   reductions are float32; weights and Adam state stay float32 and no gradient
#   scaler is used, since bf16 keeps float32's exponent range.
#
# Both reduced precisions change numerics, so nothing chooses one from the
# device or the model's width. `SelfPlay` rejects `tf32` off CUDA and `bf16`
# on CPU.


# The process's switches before any stage set them. float32 keeps them, so float32
# runs train exactly as before these options existed (cuDNN's default allows TF32).
_PROCESS_TF32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def apply_matmul_precision(precision: "Precision") -> None:
    """Set the TF32 switches a stage runs under: on for tf32, else the process's own."""
    matmul, cudnn = (True, True) if precision == "tf32" else _PROCESS_TF32
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def validate_precision(precision: str, device: torch.device) -> None:
    if precision not in ("float32", "tf32", "bf16"):
        raise ValueError(f"unsupported Torch precision: {precision!r}")
    if precision == "bf16" and device.type not in {"cuda", "mps"}:
        raise ValueError("precision bf16 requires CUDA or MPS")


def forward_precision(
    precision: "Precision", device: torch.device
) -> AbstractContextManager[object]:
    """The context one policy forward runs in: bf16 autocast on `device`
    (CUDA or MPS), else nothing."""
    validate_precision(precision, device)
    if precision != "bf16":
        return nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


def no_grad() -> AbstractContextManager[object]:
    """Compute constants: nothing inside records a gradient graph."""
    return torch.no_grad()


# -- Learner -------------------------------------------------------------------


class Learner:
    """Trainable policy and Adam state shared by every update.

    The profiler times learning phases between existing device waits; it adds
    no synchronization and draws no training randomness.
    """

    def __init__(
        self,
        agent: "ActorCritic",
        device: torch.device | str,
        *,
        minibatch_rows: int,
        precision: "Precision" = "float32",
        execution_rows: int | None = None,
        bootstrap_rows: int | None = None,
    ) -> None:
        if minibatch_rows < 1:
            raise ValueError("minibatch_rows must be positive")
        if any(v is not None and v < 1 for v in (execution_rows, bootstrap_rows)):
            raise ValueError("execution row limits must be positive")
        self.execution_rows, self.bootstrap_rows = execution_rows, bootstrap_rows
        self.device = torch.device(device)
        validate_precision(precision, self.device)
        # Checkpoint admission returns eval mode. Recurrent CUDA kernels need
        # a training-mode forward before backward, including on warm starts.
        self.agent = agent.to(self.device).train()
        self.minibatch_rows = minibatch_rows
        self.precision: Precision = precision
        # The learning rule sets the rate before stepping; retain the initial
        # Adam configuration in snapshots even when selection skips all rows.
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=2.5e-4, eps=1e-5)
        self.profiler = Profiler()

    def forward_precision(self) -> AbstractContextManager[object]:
        """Wrap each policy forward; callers cast its outputs to float32."""
        return forward_precision(self.precision, self.device)

    def set_learning_rate(self, rate: float) -> None:
        """The rate every following optimizer step uses."""
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def loss(self, objective: Callable[[], tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        """Run `objective`, whose first result is the loss `step` or
        `gradients` then differentiates; the rest are returned with it.

        An update writes its minibatch loss as this closure over the policy
        forward, because a functional framework differentiates a function
        rather than a value; torch runs it once and keeps the graph.
        """
        return objective()

    def step(
        self,
        loss: Tensor,
        max_grad_norm: float,
        *,
        recorder: "StepRecorder | None" = None,
        accumulated: bool = False,
    ) -> Tensor:
        """`gradient_step` on this learner's policy and optimizer: the
        pre-clip gradient norm."""
        return gradient_step(
            loss,
            self.agent,
            self.optimizer,
            max_grad_norm,
            recorder=recorder,
            accumulated=accumulated,
        )

    def accumulate(self, loss: Tensor, *, first: bool) -> None:
        """Backward a correctly weighted contribution, freeing its graph now."""
        check_objective(loss)
        if first:
            self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

    def gradients(self, loss: Tensor) -> "ParameterArrays":
        """Unclipped gradients of `loss` by parameter name, without stepping
        (zero where none flowed)."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {
            name: (
                np.zeros(tuple(parameter.shape), np.float32)
                if parameter.grad is None
                else parameter.grad.cpu().numpy().copy()
            )
            for name, parameter in self.agent.named_parameters()
        }
        self.optimizer.zero_grad(set_to_none=True)
        return gradients

    def parameters(self) -> "ParameterArrays":
        """A host copy of every parameter by name."""
        return {
            name: parameter.detach().cpu().numpy().copy()
            for name, parameter in self.agent.named_parameters()
        }


def _norm(tensors: list[Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(
        torch.stack(
            [float64_operand(t.detach()).double().square().sum() for t in tensors]
        )
        .sum()
        .sqrt()
    )


def critic_gradient_norms(model: nn.Module) -> tuple[float, float | None]:
    """Pre-clip norms of the critic-only readout and, if present, the value token."""
    named = dict(model.named_parameters())
    critic = _norm(
        [
            p.grad
            for name, p in named.items()
            if p.grad is not None and name.startswith(CRITIC_READOUT_PREFIXES)
        ]
    )
    token = named.get("value_token")
    if token is None:
        return critic, None
    return critic, _norm([] if token.grad is None else [token.grad])


def gradient_health(
    model: nn.Module, norm_before: Tensor, critic_norms: tuple[float, float | None]
) -> "StepSample":
    """A sampled clipped step's gradients, without changing them."""
    critic_norm, token_norm = critic_norms
    parameters = list(model.parameters())
    grads = [p.grad for p in parameters if p.grad is not None]
    health: StepSample = dict(
        gradient_norm_before=float(norm_before),
        gradient_norm_after=_norm(grads),
        gradient_norm_critic_readout=critic_norm,
        gradient_missing_tensors=sum(
            p.requires_grad and p.grad is None for p in parameters
        ),
        gradient_zero_tensors=sum(not bool(g.any()) for g in grads),
        gradient_nonzero_tensors=sum(bool(g.any()) for g in grads),
        frozen_parameter_tensors=sum(not p.requires_grad for p in parameters),
    )
    if token_norm is not None:
        health["gradient_norm_value_token"] = token_norm

    return health


def parameter_health(
    parameters: list[nn.Parameter], before: list[Tensor]
) -> "StepSample":
    """Parameter displacement of a sampled step, with established float64 arithmetic."""
    delta = (
        torch.stack(
            [
                (float64_operand(p.detach()).double() - float64_operand(b).double())
                .square()
                .sum()
                for p, b in zip(parameters, before, strict=True)
            ]
        )
        .sum()
        .sqrt()
    )
    scale = (
        torch.stack([float64_operand(b).double().square().sum() for b in before])
        .sum()
        .sqrt()
    )
    return parameter_movement(float(delta), float(scale))
