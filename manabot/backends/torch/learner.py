"""The torch learner: optimizer state, precision and the guarded update step.

`Learner` bundles a trainable `ActorCritic` with its Adam optimizer, its
forward precision and a profiler; the updates in `manabot.training.updates`
step it. `gradient_step` runs backward, global-norm clipping and the
optimizer for any `nn.Module`, refusing to step on a loss or gradients it
cannot trust (`NumericalError`, `manabot.backends.torch.rl`); an optional
`StepRecorder` (the player's `OptimizerHealth` in
`manabot.training.updates.health` is one) observes what it did. `update_ema`
keeps the averaged model that evaluation and exports read. `forward_precision` and
`apply_matmul_precision` apply a stage's declared numeric precision;
`no_grad` wraps work whose outputs are constants (targets, diagnostics).
"""

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn

from manabot.backends.torch.rl import NumericalError, require
from manabot.infra.profiler import Profiler

if TYPE_CHECKING:
    # Annotations only: the update step and EMA serve any model, so importing
    # them loads neither the player's network nor the training configuration.
    from manabot.backends.torch.policy import ActorCritic
    from manabot.training.regime import Precision

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
# - `bf16`: `torch.autocast(cuda, bfloat16)` around the policy forwards only, in
#   collection (behavior sampling and stored values) and in the learner (bootstrap
#   targets and the loss forward). Callers cast forward outputs back to float32,
#   so behavior log-probabilities, targets, losses and their reductions are float32;
#   weights and Adam state stay float32 and no gradient scaler is used, since bf16
#   keeps float32's exponent range.
#
# Both reduced precisions change numerics, so they are declared regime variants,
# never edits of a float32 regime; `Execution` rejects them off CUDA.


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
    """The context one policy forward runs in: bf16 autocast, else nothing."""
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
