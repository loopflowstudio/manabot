"""Local numerical health, independent of tracker availability and RNG.

Batch measurements cover every collected row (`batch_health`). An
`OptimizerHealth` is the player's `StepRecorder` (`manabot.rl.step`): passed
to `gradient_step`, it counts steps and samples gradient and parameter detail,
including the critic readout's share, at iteration 1 and every 25th iteration.
Safety checks run on every step, observed or not; absent samples remain absent.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor, nn

from manabot.infra.profiler import Profiler
from manabot.rl.numerics import NumericalError, float64_operand
from manabot.rl.step import check_objective
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
        torch.stack(
            [float64_operand(t.detach()).double().square().sum() for t in tensors]
        )
        .sum()
        .sqrt()
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


@torch.no_grad()
def clip_fraction(log_ratio: Tensor, clip: float) -> float:
    """Fraction of rows whose likelihood ratio lies outside `1 ± clip`.

    The ratio is exponentiated in float64 so large log ratios do not overflow.
    """
    return float(
        ((float64_operand(log_ratio).double().exp() - 1).abs() > clip).float().mean()
    )


class OptimizerHealth:
    """Record what `gradient_step` does to one iteration's `NumericalHealth`.

    Counts steps and rejections on every step. Detailed gradient and parameter
    measurements sample the first step of iteration 1 and every 25th
    iteration. Recording never changes the loss, the step or any RNG.

    `check` rejects a detached or nonfinite objective; it waits for the loss on
    the device, so a caller timing its forward pass ends that timer there.
    With a profiler, `gradient_step` times backward and the optimizer step
    separately; each ends at a finite check that already waits for the device,
    so timing adds no synchronization.
    """

    _critic_norms: tuple[float, float | None] | None

    def __init__(
        self,
        health: NumericalHealth,
        iteration: int,
        profiler: Profiler | None = None,
    ) -> None:
        self.health = health
        self.measure = iteration == 1 or iteration % HEALTH_INTERVAL == 0
        self.profiler = profiler or Profiler(enabled=False)
        self._critic_norms = None

    def track(self, section: str) -> AbstractContextManager[object]:
        return self.profiler.track(section)

    def check(self, loss: Tensor) -> None:
        with self.rejecting():
            check_objective(loss)

    @contextmanager
    def rejecting(self) -> Iterator[None]:
        """Count a NumericalError as a rejected step and attach the record."""
        try:
            yield
        except NumericalError as error:
            self.health["rejected_steps"] += 1
            error.health = dict(self.health)
            raise

    def sampling(self) -> bool:
        """Whether the coming step is this iteration's detailed sample."""
        return self.measure and not self.health["sampled_steps"]

    def before_clip(self, model: nn.Module) -> None:
        """Read the critic's pre-clip gradient norms (`critic_gradient_norms`)."""
        self._critic_norms = critic_gradient_norms(model)

    def record_gradients(self, model: nn.Module, norm_before: Tensor) -> None:
        """After clipping, on a sampled step that `before_clip` already read."""
        if self._critic_norms is None:
            raise RuntimeError("record_gradients requires before_clip first")
        critic_norm, token_norm = self._critic_norms
        self._critic_norms = None
        parameters = list(model.parameters())
        grads = [p.grad for p in parameters if p.grad is not None]
        self.health.update(
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
            self.health["gradient_norm_value_token"] = token_norm

    def record_step(self, parameters: list[nn.Parameter], before: list[Tensor]) -> None:
        """Count a completed step; `before` is empty unless this step is sampled."""
        self.health["optimizer_steps"] += 1
        if not before:
            return
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
        self.health["parameter_delta_l2"] = float(delta)
        self.health["parameter_relative_delta"] = (
            float(delta / scale) if scale > 0 else float(delta)
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


@contextmanager
def failure_tensors(tensors: Callable[[], dict[str, Tensor]]) -> Iterator[None]:
    """Attach `tensors()` to a NumericalError raised inside, then re-raise it.

    The callable runs only on failure, so a healthy step builds no evidence.
    """
    try:
        yield
    except NumericalError as error:
        error.tensors.update(tensors())
        raise


def minibatch_tensors(
    *,
    logits: Tensor,
    value_logits: Tensor,
    valid: Tensor,
    actions: Tensor,
    behavior_logs: Tensor,
    reference: Tensor,
    advantages: Tensor,
    value_targets: Tensor,
    actor_selected: Tensor,
    indexes: Tensor,
    obs: dict[str, Tensor],
) -> dict[str, Tensor]:
    """The evidence a rejected update step retains: its minibatch and outputs."""
    return {
        "logits": logits.detach(),
        "value_logits": value_logits.detach(),
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
