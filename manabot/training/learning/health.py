"""Local numerical health, independent of tracker availability and RNG.

Batch measurements cover every collected row. Detailed gradient/parameter
measurements sample the first optimizer step at iteration 1 and every 25th
iteration. Safety checks run on every step; absent samples remain absent.
"""

from typing import TypedDict

import torch
from torch import Tensor, nn

from manabot.model.policy_distribution import NumericalError, require

HEALTH_INTERVAL = 25


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
    gradient_missing_tensors: int
    gradient_zero_tensors: int
    gradient_nonzero_tensors: int
    frozen_parameter_tensors: int
    parameter_delta_l2: float
    parameter_relative_delta: float
    clip_fraction: float


@torch.no_grad()
def batch_health(
    logs: Tensor,
    probabilities: Tensor,
    valid: Tensor,
    advantages: Tensor,
    selected: Tensor,
    *,
    actor_only: bool,
) -> NumericalHealth:
    gaps = logs.masked_fill(~valid, -torch.inf).amax(-1) - logs.masked_fill(
        ~valid, torch.inf
    ).amin(-1)
    return NumericalHealth(
        legal_logit_gap_max=float(gaps.max()),
        legal_probability_min=float(probabilities.masked_fill(~valid, torch.inf).min()),
        legal_underflow_count=int(((probabilities == 0) & valid).sum()),
        nonfinite_count=int((~torch.isfinite(logs) & valid).sum()),
        forced_rows=int((valid.sum(-1) == 1).sum()),
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
    """Guard Adam and sample its actual effect without altering loss or RNG."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        health: NumericalHealth,
        iteration: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.health = health
        self.measure = iteration == 1 or iteration % HEALTH_INTERVAL == 0

    def step(self, loss: Tensor, max_norm: float) -> Tensor:
        try:
            return self._step(loss, max_norm)
        except NumericalError as error:
            self.health["rejected_steps"] += 1
            error.health = dict(self.health)
            raise

    def _step(self, loss: Tensor, max_norm: float) -> Tensor:
        parameters = list(self.model.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        if not loss.requires_grad:
            raise NumericalError("detached_objective", {"loss": loss})
        require(torch.isfinite(loss), "nonfinite_objective", loss=loss)
        loss.backward()
        measured = self.measure and not self.health["sampled_steps"]
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            raise NumericalError("missing_all_gradients", {"loss": loss})
        try:
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
        before = [p.detach().clone() for p in parameters] if measured else []
        if measured:
            self.health.update(
                sampled_steps=1,
                gradient_norm_before=float(norm),
                gradient_norm_after=float(
                    torch.stack([g.detach().double().square().sum() for g in grads])
                    .sum()
                    .sqrt()
                ),
                gradient_missing_tensors=sum(
                    p.requires_grad and p.grad is None for p in parameters
                ),
                gradient_zero_tensors=sum(not bool(g.any()) for g in grads),
                gradient_nonzero_tensors=sum(bool(g.any()) for g in grads),
                frozen_parameter_tensors=sum(not p.requires_grad for p in parameters),
            )
        self.optimizer.step()
        finite = torch.stack([torch.isfinite(p).all() for p in parameters]).all()
        if not finite:
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
        return norm
