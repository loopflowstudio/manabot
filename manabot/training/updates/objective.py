"""Differentiable damped policy and critic losses over a framework's RL primitives.

`damped_policy_loss` returns one actor term per row. `minibatch_objective`
combines the selected actor rows and all critic rows with the weights of their
logical minibatch. Both operate directly on the learner's differentiable arrays.
This module can also be used independently of game, rollout and model code.
"""

from typing import Any, Protocol, runtime_checkable

from manabot.training.updates.decision_weights import actor_term, critic_term

# The framework owns array operations, indexing, reductions and differentiation.
type Array = Any


class LossTerms(Protocol):
    """Framework operations composing the actor and scalar critic objective."""

    def float64_operand(self, tensor: Array) -> Array: ...

    def constant(self, tensor: Array) -> Array: ...

    def legal_mask(self, mask: Array) -> Array: ...

    def policy_logs(self, logits: Array, valid: Array) -> Array: ...

    def action_logs(self, logs: Array, actions: Array) -> Array: ...

    def reference_logs(self, reference: Array, valid: Array) -> Array: ...

    def clipped_surrogate(
        self, log_ratio: Array, advantages: Array, clip: float
    ) -> Array: ...

    def reverse_kl(self, logs: Array, target_logs: Array, valid: Array) -> Array: ...

    def forward_kl(self, logs: Array, target_logs: Array, valid: Array) -> Array: ...

    def squared_value_loss(self, values: Array, targets: Array) -> Array: ...

    def check_logs(self, logs: Array, valid: Array) -> None: ...

    def check_probabilities(
        self, probabilities: Array, valid: Array, *, positive: bool
    ) -> None: ...

    def require(self, condition: Array, invariant: str, **tensors: Array) -> None: ...

    def weighted_sum(self, values: Array, weights: Array) -> Array: ...

    def selected_mean(self, values: Array, selected: Array) -> Array: ...


@runtime_checkable
class CategoricalValueLoss(Protocol):
    def categorical_value_loss(self, value_logits: Array, targets: Array) -> Array: ...


def damped_policy_loss(
    rl: LossTerms,
    logits: Array,
    actions: Array,
    advantages: Array,
    behavior_logs: Array,
    reference: Array,
    valid: Array,
    *,
    clip: float,
    collection_kl: float,
    tau: float,
    reference_forward_kl: float = 0.0,
) -> Array:
    """The damped policy objective, equation (6), per row [B] over offers [B,A].

    An optional forward KL keeps a restoring gradient on rare legal offers.
    Sources and departures: docs/learning.md#fidelity-to-the-sources.

    Ratios use the actual action likelihood at collection, even after earlier
    minibatches change the learner. The reverse KLs sum all legal actions.
    This is per-action reuse correction, not trajectory importance weighting.
    MPS combines these terms on the host to retain float64 surrogate range.
    """
    valid = rl.legal_mask(valid)
    if logits.ndim != 2 or logits.shape != valid.shape:
        raise ValueError("policy logits require [B,A] legal support")
    if actions.shape != logits.shape[:1] or advantages.shape != actions.shape:
        raise ValueError("actions and advantages require [B]")
    if not 0 < clip < 1 or collection_kl < 0 or tau < 0 or reference_forward_kl < 0:
        raise ValueError("invalid damping coefficients")
    rl.check_logs(behavior_logs, valid)
    rl.check_probabilities(reference, valid, positive=True)
    rl.require(
        advantages.abs() < float("inf"),
        "nonfinite_advantages",
        advantages=advantages,
        valid=valid,
    )
    if ((actions < 0) | (actions >= logits.shape[-1])).any():
        raise ValueError("action index outside legal support")
    # Admitted behavior logs are finite on legal offers and -inf on padding.
    # Gather before subtracting so padding never contributes -inf - -inf.
    chosen_behavior = rl.action_logs(rl.constant(behavior_logs), actions)
    if not (chosen_behavior > -float("inf")).all():
        raise ValueError("sampled action is not legal")
    logs = rl.policy_logs(logits, valid)
    ratio_logs = rl.action_logs(logs, actions) - chosen_behavior
    surrogate = rl.clipped_surrogate(ratio_logs, advantages, clip)
    magnet_logs = rl.reference_logs(reference, valid)
    loss = (
        surrogate
        + collection_kl * rl.float64_operand(rl.reverse_kl(logs, behavior_logs, valid))
        + tau * rl.float64_operand(rl.reverse_kl(logs, magnet_logs, valid))
    )
    if reference_forward_kl:
        loss = loss + reference_forward_kl * rl.float64_operand(
            rl.forward_kl(logs, magnet_logs, valid)
        )
    return loss


def minibatch_objective(
    rl: LossTerms,
    logits: Array,
    value_logits: Array,
    valid: Array,
    *,
    actions: Array,
    advantages: Array,
    value_targets: Array,
    behavior_logs: Array,
    reference: Array,
    actor_mask: Array,
    categorical: bool,
    clip: float,
    collection_kl: float,
    tau: float,
    reference_forward_kl: float = 0.0,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    weights: Array | None = None,
) -> tuple[Array, Array, Array]:
    """The damped rule's loss on one minibatch: (loss, policy loss, value loss).

    `rl` is a backend's `rl` module and every array is that backend's:
    float32 `logits` [B,A] and critic output ([B,1], or [B,3] outcome logits
    when `categorical`) from the current forward, the legal mask [B,A], and
    the constants `actions` [B], `advantages` [B], `value_targets` ([B] or
    [B,3]), `behavior_logs` [B,A], `reference` probabilities [B,A] and the
    actor-selected rows [B]. Optional [B,2] coefficients include the actor
    and critic weights and their logical minibatch denominators. Gradients reach only
    `logits` and `value_logits`.
    """
    policy_loss = actor_term(
        rl,
        damped_policy_loss(
            rl,
            logits,
            actions,
            advantages,
            behavior_logs,
            reference,
            valid,
            clip=clip,
            collection_kl=collection_kl,
            tau=tau,
            reference_forward_kl=reference_forward_kl,
        ),
        actor_mask,
        policy_weight,
        weights,
    )
    if categorical:
        if not isinstance(rl, CategoricalValueLoss):
            raise TypeError(
                f"{getattr(rl, '__name__', type(rl).__name__)} has no categorical critic"
            )
        value_rows = rl.categorical_value_loss(value_logits, value_targets)
    else:
        # Representation ablation, not the paper's outcome objective.
        value_rows = rl.squared_value_loss(value_logits, value_targets)
    value_loss = critic_term(rl, value_rows, value_weight, weights)
    return policy_loss + rl.float64_operand(value_loss), policy_loss, value_loss
