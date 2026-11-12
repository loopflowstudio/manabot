"""The damped-policy PPO learner step in MLX (`manabot.training.learning.damped_policy`).

`damped_loss` is equation (6) of the damped update over one minibatch plus the
scalar critic's squared error: a clipped likelihood-ratio surrogate, reverse KL
to the collection policy and reverse KL to the reference policy, averaged over
the actor-selected rows. Behavior log-likelihoods, the reference and targets
are constants; gradients reach the policy and critic logits only through the
current forward.

One departure: PyTorch evaluates the surrogate ratio in float64 so that
exp(log ratio) cannot overflow; MLX's GPU has no float64, so the ratio is
float32 here. Ratios that differ in float32 would need |log ratio| > 88,
which no sampled action reaches in practice; the parity check would fail first.

`MLXLearner.step` then clips the global gradient norm and applies Adam with the
learner's settings (lr per iteration, eps 1e-5, bias correction), matching
`OptimizerHealth`'s backward and optimizer step. It measures the step; it does
not write checkpoints.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from manabot.model.mlx.model import MLXActorCritic
from manabot.model.mlx.prepare import Prepared

Gradients = dict[str, object]


@dataclass(frozen=True)
class Targets:
    """Constants of one minibatch: sampled actions [B], advantages [B], value
    targets [B], actor-selected rows [B], behavior log-likelihoods [B,A]
    (-inf off support) and reference probabilities [B,A]."""

    actions: np.ndarray
    advantages: np.ndarray
    value_targets: np.ndarray
    actor_mask: np.ndarray
    behavior_logs: np.ndarray
    reference: np.ndarray


@dataclass(frozen=True)
class Settings:
    learning_rate: float
    clip: float
    collection_kl: float
    tau: float
    max_grad_norm: float


def _reverse_kl(logs: mx.array, target: mx.array, valid: mx.array) -> mx.array:
    """KL(current || target), zero on padding without 0 * -inf."""
    difference = mx.where(valid, logs, 0.0) - mx.where(valid, target, 0.0)
    return (mx.exp(logs) * difference).sum(-1)


def damped_loss(
    policy: MLXActorCritic, batch: Prepared, targets: Targets, settings: Settings
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Loss, and the logits and values it read."""
    logits, values = policy(batch)
    valid = mx.array(batch.legal)
    masked = mx.where(valid, logits, -mx.inf)
    logs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    actions = mx.array(targets.actions.astype(np.int64))[:, None]
    behavior = mx.array(targets.behavior_logs)
    ratio = mx.exp(
        mx.take_along_axis(logs, actions, axis=-1)[:, 0]
        - mx.take_along_axis(behavior, actions, axis=-1)[:, 0]
    )
    advantages = mx.array(targets.advantages)
    surrogate = -mx.minimum(
        ratio * advantages,
        mx.clip(ratio, 1 - settings.clip, 1 + settings.clip) * advantages,
    )
    reference = mx.log(mx.where(valid, mx.array(targets.reference), 1.0))
    policy_rows = (
        surrogate
        + settings.collection_kl * _reverse_kl(logs, behavior, valid)
        + settings.tau * _reverse_kl(logs, reference, valid)
    )
    selected = mx.array(targets.actor_mask.astype(np.float32))
    policy_loss = (policy_rows * selected).sum() / mx.maximum(selected.sum(), 1)
    value_loss = mx.square(values[:, 0] - mx.array(targets.value_targets)).mean()
    return policy_loss + value_loss, (logits, values)


class MLXLearner:
    """Adam over `policy`'s parameters, one minibatch per `step`."""

    def __init__(self, policy: MLXActorCritic) -> None:
        self.policy = policy
        policy.clear_cache()
        self.optimizer = optim.Adam(
            learning_rate=2.5e-4, betas=[0.9, 0.999], eps=1e-5, bias_correction=True
        )
        self._value_and_grad = nn.value_and_grad(policy, self._loss)

    def _loss(self, batch: Prepared, targets: Targets, settings: Settings) -> mx.array:
        loss, _ = damped_loss(self.policy, batch, targets, settings)
        return loss

    def gradients(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[mx.array, Gradients]:
        """Loss and unclipped parameter gradients, evaluated."""
        loss, grads = self._value_and_grad(batch, targets, settings)
        mx.eval(loss, grads)
        return loss, grads

    def step(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[float, float]:
        """One optimizer step; returns the loss and the pre-clip gradient norm."""
        loss, grads = self._value_and_grad(batch, targets, settings)
        grads, norm = optim.clip_grad_norm(grads, settings.max_grad_norm)
        self.optimizer.learning_rate = settings.learning_rate
        self.optimizer.update(self.policy, grads)
        mx.eval(loss, norm, self.policy.parameters(), self.optimizer.state)
        return float(loss), float(norm)
