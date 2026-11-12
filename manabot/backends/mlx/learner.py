"""The damped-policy PPO learner step in MLX (`manabot.training.updates.damped_policy`).

`damped_loss` is equation (6) of the damped update over one minibatch plus the
scalar critic's squared error (`manabot.backends.mlx.rl`): a clipped
likelihood-ratio surrogate, reverse KL to the collection policy and reverse KL
to the reference policy, averaged over the actor-selected rows. Behavior
log-likelihoods, the reference and targets are constants; gradients reach the
policy and critic logits only through the current forward.

Precision is float32 throughout, including the surrogate ratio PyTorch takes
in float64 (`rl.clipped_surrogate` states the bound).

`MLXLearner.step` then clips the global gradient norm and applies Adam with the
learner's settings (lr per iteration, eps 1e-5, bias correction), matching
`OptimizerHealth`'s backward and optimizer step. It keeps no EMA or reference
copy of its own: the reference distribution arrives in `Targets`. It measures
the step; it does not write checkpoints.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from manabot.backends.mlx.arrays import FLOAT, flat_numpy, to_mlx
from manabot.backends.mlx.policy import MLXActorCritic
from manabot.backends.mlx.rl import (
    clipped_surrogate,
    masked_log_softmax,
    reverse_kl,
    selected_mean,
    squared_error,
)
from manabot.policy.prepare import Prepared
from manabot.policy.protocol import Settings, Targets


def damped_loss(
    policy: MLXActorCritic, batch: Prepared, targets: Targets, settings: Settings
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Loss, and the logits and values it read."""
    logits, values = policy(batch)
    valid = to_mlx(batch.legal)
    logs = masked_log_softmax(logits, valid)
    behavior = to_mlx(targets.behavior_logs)
    surrogate = clipped_surrogate(
        logs,
        behavior,
        to_mlx(targets.actions.astype(np.int64))[:, None],
        to_mlx(targets.advantages),
        settings.clip,
    )
    reference = mx.log(mx.where(valid, to_mlx(targets.reference), 1.0))
    policy_rows = (
        surrogate
        + settings.collection_kl * reverse_kl(logs, behavior, valid)
        + settings.tau * reverse_kl(logs, reference, valid)
    )
    policy_loss = selected_mean(policy_rows, to_mlx(targets.actor_mask).astype(FLOAT))
    value_loss = squared_error(values[:, 0], to_mlx(targets.value_targets))
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
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Loss and unclipped parameter gradients."""
        loss, grads = self._value_and_grad(batch, targets, settings)
        mx.eval(loss, grads)
        return float(loss), flat_numpy(grads)

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

    def parameters(self) -> dict[str, np.ndarray]:
        return flat_numpy(self.policy.parameters())
