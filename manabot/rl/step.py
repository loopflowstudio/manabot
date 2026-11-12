"""The update step: one guarded gradient step, and the parameter average.

`gradient_step` runs backward, global-norm clipping and the optimizer for any
`nn.Module`. It refuses to step on a loss or gradients it cannot trust and
raises `NumericalError` (`manabot.rl.numerics`) instead. An optional
`StepRecorder` observes what it did: timing, rejections and sampled gradient
detail; the player's `OptimizerHealth`
(`manabot.training.learning.health`) is one. `update_ema` keeps the averaged
model that evaluation and exports read.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import torch
from torch import Tensor, nn

from manabot.rl.numerics import NumericalError, require


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
