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
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from manabot.backends.torch.rl import NumericalError, float64_operand, require
from manabot.infra.profiler import Profiler

if TYPE_CHECKING:
    # Annotations only: the update step and EMA serve any model, so importing
    # them loads neither the player's network nor the training configuration.
    from manabot.backends.torch.policy import ActorCritic
    from manabot.policy.checkpoint import ParameterArrays
    from manabot.training.learning import Precision

# -- Update step ---------------------------------------------------------------


class StepRecorder(Protocol):
    """What `gradient_step` reports; recording never changes the loss, step or RNG.

    `track(section)` times "Backward and gradient clipping" and "Optimizer
    step"; each ends at a finite check that already waits for the device.
    `rejecting()` wraps the whole step, so it sees every `NumericalError`.
    When `sampling()` is true for a step, `before_clip` reads the gradients
    before clipping rescales them in place and `record_gradients` reads them
    after; `record_step` follows every completed step, with the pre-step
    parameters only on a sampled step (empty otherwise).
    """

    def track(self, section: str) -> AbstractContextManager[object]: ...

    def rejecting(self) -> AbstractContextManager[object]: ...

    def sampling(self) -> bool: ...

    def before_clip(self, model: nn.Module) -> None: ...

    def record_gradients(self, model: nn.Module, norm_before: Tensor) -> None: ...

    def record_step(
        self, parameters: list[nn.Parameter], before: list[Tensor]
    ) -> None: ...


def check_objective(loss: Tensor) -> None:
    """Raise NumericalError for a loss that is detached or nonfinite."""
    if not loss.requires_grad:
        raise NumericalError("detached_objective", {"loss": loss})
    require(torch.isfinite(loss), "nonfinite_objective", loss=loss)


class _Unrecorded:
    """The `StepRecorder` of an unobserved step: no timing, no samples."""

    def track(self, section: str) -> AbstractContextManager[object]:
        return nullcontext()

    def rejecting(self) -> AbstractContextManager[object]:
        return nullcontext()

    def sampling(self) -> bool:
        return False

    def before_clip(self, model: nn.Module) -> None:
        pass

    def record_gradients(self, model: nn.Module, norm_before: Tensor) -> None:
        pass

    def record_step(self, parameters: list[nn.Parameter], before: list[Tensor]) -> None:
        pass


def gradient_step(
    loss: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    *,
    recorder: StepRecorder | None = None,
) -> Tensor:
    """Zero, backward, clip the global norm to `max_grad_norm`, step; return the pre-clip norm.

    Raises before stepping on a detached or nonfinite loss, a loss that reaches
    no parameter, or nonfinite gradients; the model and optimizer are then
    unchanged. Raises after stepping if a parameter became nonfinite.
    """
    recorder = recorder or _Unrecorded()
    with recorder.rejecting():
        check_objective(loss)
        parameters = list(model.parameters())
        optimizer.zero_grad(set_to_none=True)
        with recorder.track("Backward and gradient clipping"):
            loss.backward()
            sampled = recorder.sampling()
            if not any(p.grad is not None for p in parameters):
                raise NumericalError("missing_all_gradients", {"loss": loss})
            if sampled:
                recorder.before_clip(model)
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
            if sampled:
                recorder.record_gradients(model, norm)
        with recorder.track("Optimizer step"):
            before = [p.detach().clone() for p in parameters] if sampled else []
            optimizer.step()
            # One flat read: per-tensor checks cost two launches per parameter tensor.
            if not torch.isfinite(
                torch.cat([p.detach().flatten() for p in parameters])
            ).all():
                raise NumericalError(
                    "nonfinite_parameters_after_step",
                    {f"parameter/{name}": p for name, p in model.named_parameters()},
                )
            recorder.record_step(parameters, before)
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
# Numeric precision a training stage declares in `Execution.precision`.
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
# A regime's `auto` resolves to `bf16` for wide models on CUDA and MPS and to
# `float32` otherwise (`manabot.training.regime.ladder_precision`). Both reduced
# precisions change numerics, so a saved regime keeps the precision it recorded.
# `Execution` rejects `tf32` off CUDA and `bf16` on CPU.


# The process's switches before any stage set them. float32 keeps them, so float32
# runs train exactly as before these options existed (cuDNN's default allows TF32).
_PROCESS_TF32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def apply_matmul_precision(precision: "Precision") -> None:
    """Set the TF32 switches a stage runs under: on for tf32, else the process's own."""
    matmul, cudnn = (True, True) if precision == "tf32" else _PROCESS_TF32
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def forward_precision(
    precision: "Precision", device: torch.device
) -> AbstractContextManager[object]:
    """The context one policy forward runs in: bf16 autocast on `device`
    (CUDA or MPS), else nothing."""
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
    ) -> None:
        if minibatch_rows < 1:
            raise ValueError("minibatch_rows must be positive")
        self.device = torch.device(device)
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
        recorder: StepRecorder | None = None,
    ) -> Tensor:
        """`gradient_step` on this learner's policy and optimizer: the
        pre-clip gradient norm."""
        return gradient_step(
            loss, self.agent, self.optimizer, max_grad_norm, recorder=recorder
        )

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


# ActorCritic parameter prefixes that only the critic output reads, in every
# value aggregation.
_CRITIC_READOUT = ("value_head.", "value_readout.")
# The value token is an unmasked key in object attention, so actor logits read
# it too: its gradient mixes policy and value loss and is reported on its own.
_VALUE_TOKEN = "value_token"


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
            if p.grad is not None and name.startswith(_CRITIC_READOUT)
        ]
    )
    token = named.get(_VALUE_TOKEN)
    if token is None:
        return critic, None
    return critic, _norm([] if token.grad is None else [token.grad])


def gradient_health(
    model: nn.Module, norm_before: Tensor, critic_norms: tuple[float, float | None]
) -> dict[str, float | int]:
    """A sampled clipped step's gradients, without changing them."""
    health: dict[str, float | int] = {}
    if critic_norms is None:
        raise RuntimeError("record_gradients requires before_clip first")
    critic_norm, token_norm = critic_norms
    parameters = list(model.parameters())
    grads = [p.grad for p in parameters if p.grad is not None]
    health.update(
        sampled_steps=1,
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
) -> dict[str, float]:
    """Parameter displacement of a sampled step, with established float64 arithmetic."""
    health: dict[str, float] = {}
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
    health["parameter_delta_l2"] = float(delta)
    health["parameter_relative_delta"] = (
        float(delta / scale) if scale > 0 else float(delta)
    )
    return health
