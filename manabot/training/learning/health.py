"""Local numerical health, independent of tracker availability and RNG.

Batch measurements cover every collected row. Detailed gradient/parameter
measurements sample the first optimizer step at iteration 1 and every 25th
iteration. Safety checks run on every step; absent samples remain absent.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor, nn

from manabot.infra.profiler import Profiler
from manabot.model.policy_distribution import NumericalError, require
from manabot.training.rollout.actions import PackedActions

HEALTH_INTERVAL = 25
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
        torch.stack([t.detach().double().square().sum() for t in tensors]).sum().sqrt()
    )


class NumericalHealth(TypedDict, total=False):
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
    clip_fraction: float


@torch.no_grad()
def batch_health(
    action_data: PackedActions,
    advantages: Tensor,
    selected: Tensor,
    *,
    actor_only: bool,
) -> NumericalHealth:
    logs, probabilities = action_data.log_probabilities, action_data.probabilities
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
        critic_rows=selected.numel() if actor_only else int(selected.sum()),
        empty_actor_batch=int(not selected.any()),
        optimizer_steps=0,
        skipped_steps=0,
        rejected_steps=0,
        sampled_steps=0,
    )


class OptimizerHealth:
    """Guard Adam and sample its actual effect without altering loss or RNG.

    `check` rejects a detached or nonfinite objective; it waits for the loss on
    the device, so a caller timing its forward pass ends that timer there.
    `step` then runs backward, clipping and Adam. With a profiler, backward and
    the optimizer step are timed separately; each ends at a finite check that
    already waits for the device, so timing adds no synchronization.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        health: NumericalHealth,
        iteration: int,
        profiler: Profiler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.health = health
        self.measure = iteration == 1 or iteration % HEALTH_INTERVAL == 0
        self.profiler = profiler or Profiler(enabled=False)

    def check(self, loss: Tensor) -> None:
        with self._rejecting():
            self._check(loss)

    def step(self, loss: Tensor, max_norm: float, *, checked: bool = False) -> Tensor:
        with self._rejecting():
            if not checked:
                self._check(loss)
            return self._step(loss, max_norm)

    @contextmanager
    def _rejecting(self) -> Iterator[None]:
        try:
            yield
        except NumericalError as error:
            self.health["rejected_steps"] += 1
            error.health = dict(self.health)
            raise

    def _check(self, loss: Tensor) -> None:
        if not loss.requires_grad:
            raise NumericalError("detached_objective", {"loss": loss})
        require(torch.isfinite(loss), "nonfinite_objective", loss=loss)

    def _step(self, loss: Tensor, max_norm: float) -> Tensor:
        parameters = list(self.model.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        with self.profiler.track("Backward and gradient clipping"):
            norm, measured = self._backward(loss, parameters, max_norm)
        with self.profiler.track("Optimizer step"):
            self._optimize(parameters, measured)
        return norm

    def _backward(
        self, loss: Tensor, parameters: list[nn.Parameter], max_norm: float
    ) -> tuple[Tensor, bool]:
        loss.backward()
        measured = self.measure and not self.health["sampled_steps"]
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            raise NumericalError("missing_all_gradients", {"loss": loss})
        # Read before clipping rescales the gradients in place.
        named = dict(self.model.named_parameters()) if measured else {}
        critic_norm = _norm(
            [
                p.grad
                for name, p in named.items()
                if p.grad is not None and name.startswith(_CRITIC_READOUT)
            ]
        )
        token = named.get(_VALUE_TOKEN)
        token_norm = (
            None if token is None else _norm([] if token.grad is None else [token.grad])
        )
        try:
            # error_if_nonfinite reads the norm on the host: the device wait
            # that ends the backward timer.
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=True
            )
        except RuntimeError as error:
            raise NumericalError(
                "nonfinite_gradients",
                {
                    f"gradient/{name}": p.grad
                    for name, p in self.model.named_parameters()
                    if p.grad is not None
                },
            ) from error
        if measured:
            self.health.update(
                sampled_steps=1,
                gradient_norm_before=float(norm),
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
                self.health["gradient_norm_value_token"] = token_norm
        return norm, measured

    def _optimize(self, parameters: list[nn.Parameter], measured: bool) -> None:
        before = [p.detach().clone() for p in parameters] if measured else []
        self.optimizer.step()
        # One flat read: per-tensor checks cost two launches per parameter tensor.
        if not torch.isfinite(
            torch.cat([p.detach().flatten() for p in parameters])
        ).all():
            raise NumericalError(
                "nonfinite_parameters_after_step",
                {f"parameter/{name}": p for name, p in self.model.named_parameters()},
            )
        self.health["optimizer_steps"] += 1
        if measured:
            delta = (
                torch.stack(
                    [
                        (p.detach().double() - b.double()).square().sum()
                        for p, b in zip(parameters, before, strict=True)
                    ]
                )
                .sum()
                .sqrt()
            )
            scale = (
                torch.stack([b.double().square().sum() for b in before]).sum().sqrt()
            )
            self.health["parameter_delta_l2"] = float(delta)
            self.health["parameter_relative_delta"] = (
                float(delta / scale) if scale > 0 else float(delta)
            )
