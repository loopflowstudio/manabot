"""The standard reinforcement-learning functions over MLX arrays.

The same functions, with the same names, arguments, shapes and per-row
results, as `manabot.backends.torch.rl`; the updates in
`manabot.training.updates` reach them as `backend.rl`. Rows are minibatch
samples [B]; distributions are over action slots [B,A] with `valid` marking
the legal support. Targets, behavior likelihoods and references are
constants: gradients reach only the current policy and value outputs.

Three differences from torch are deliberate. MLX's GPU has no float64, so the
surrogate ratio is float32 (`clipped_surrogate` states the limit) and
`float64_operand` is the identity. Returns and advantage selection run on the
host through `manabot.policy.returns`, whose NumPy functions match torch's
bit for bit. The categorical critic's functions are absent: the backend
refuses a categorical checkpoint (`manabot.backends.mlx.checkpoint.require_supported`).
"""

from collections.abc import Mapping
from typing import Literal

import mlx.core as mx
import numpy as np

from manabot.policy import returns

# -- Numerical refusals --------------------------------------------------------


class NumericalError(ValueError):
    """An identified invariant plus arrays for bounded private failure capture."""

    def __init__(self, invariant: str, tensors: Mapping[str, mx.array]) -> None:
        super().__init__(invariant)
        self.invariant = invariant
        self.health: dict[str, int | float] = {}
        self.tensors = {key: mx.stop_gradient(value) for key, value in tensors.items()}


def require(condition: mx.array, invariant: str, **tensors: mx.array) -> None:
    """Raise `NumericalError(invariant)` unless every element of `condition` holds."""
    if not bool(condition.all()):
        raise NumericalError(
            invariant, {**tensors, "failed": mx.logical_not(condition)}
        )


def require_finite(invariant: str, **tensors: mx.array) -> None:
    """Raise `NumericalError(invariant)` unless every element of each array is finite."""
    for tensor in tensors.values():
        require(mx.isfinite(tensor), invariant, **tensors)


def float64_operand(tensor: mx.array) -> mx.array:
    """The identity: MLX computes every term in float32 on one device."""
    return tensor


def constant(tensor: mx.array) -> mx.array:
    """The same values with no gradient path: targets, evidence, diagnostics."""
    return mx.stop_gradient(tensor)


# -- Policy distribution -------------------------------------------------------


def legal_mask(mask: mx.array) -> mx.array:
    """The boolean legal support of a 0/1 mask [*,A]."""
    require(
        mx.isfinite(mask) & ((mask == 0) | (mask == 1)),
        "invalid_action_mask",
        mask=mask,
    )
    return mask.astype(mx.bool_)


def policy_logs(logits: mx.array, valid: mx.array) -> mx.array:
    """Normalize in float32 for both behavior and learning, including half forwards.

    Padding is -inf and legal logs must be finite. Promote before normalization,
    so recorded behavior logs do not retain half-rounded log-softmax values.
    """
    logits = logits.astype(mx.float32)
    if logits.shape != valid.shape or logits.ndim < 2:
        raise NumericalError("policy_shape", {"logits": logits, "valid": valid})
    require(valid.any(-1), "empty_legal_support", logits=logits, valid=valid)
    illegal = mx.logical_not(valid)
    require(
        mx.isfinite(logits) | illegal,
        "nonfinite_legal_logits",
        logits=logits,
        valid=valid,
    )
    masked = mx.where(valid, logits, -mx.inf)
    logs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    require(
        mx.isfinite(logs) | illegal,
        "nonfinite_legal_logs",
        logs=logs,
        logits=logits,
        valid=valid,
    )
    return logs


def action_logs(logs: mx.array, actions: mx.array) -> mx.array:
    """Float32 log probability of each sampled action [B]."""
    return mx.take_along_axis(logs.astype(mx.float32), actions[:, None], axis=-1)[:, 0]


def reference_logs(reference: mx.array, valid: mx.array) -> mx.array:
    """Log probabilities [B,A] of a constant reference, zero off `valid`."""
    return mx.log(mx.where(valid, mx.stop_gradient(reference), 1.0))


def check_probabilities(
    probabilities: mx.array, valid: mx.array, *, positive: bool
) -> None:
    if probabilities.shape != valid.shape or probabilities.ndim < 2:
        raise NumericalError(
            "distribution_shape", {"probabilities": probabilities, "valid": valid}
        )
    tensors = {"probabilities": probabilities, "valid": valid}
    illegal = mx.logical_not(valid)
    require(valid.any(-1), "empty_legal_support", **tensors)
    require(mx.isfinite(probabilities), "nonfinite_probabilities", **tensors)
    require(probabilities >= 0, "negative_probabilities", **tensors)
    if positive:
        require((probabilities > 0) | illegal, "nonpositive_legal_support", **tensors)
    require((probabilities == 0) | valid, "illegal_probability_mass", **tensors)
    require(
        mx.abs(probabilities.sum(-1) - 1) <= 2e-5,
        "probability_normalization",
        **tensors,
    )


def check_logs(logs: mx.array, valid: mx.array) -> None:
    if logs.shape != valid.shape:
        raise NumericalError("log_shape", {"logs": logs, "valid": valid})
    require(valid.any(-1), "empty_legal_support", logs=logs, valid=valid)
    require(
        mx.isfinite(logs) | mx.logical_not(valid),
        "nonfinite_behavior_logs",
        logs=logs,
        valid=valid,
    )
    require(mx.isneginf(logs) | valid, "illegal_log_mass", logs=logs, valid=valid)
    require(
        mx.abs(mx.logsumexp(logs, axis=-1)) <= 1e-5,
        "log_normalization",
        logs=logs,
        valid=valid,
    )


# -- Returns -------------------------------------------------------------------


def transition_gae(
    rewards: mx.array,
    values: mx.array,
    ends: mx.array,
    next_value: mx.array,
    gamma: float,
    lam: float,
) -> tuple[mx.array, mx.array]:
    """GAE for transition-end flags [T,E], on the host
    (`manabot.policy.returns.transition_gae`)."""
    advantages, value_targets = returns.transition_gae(
        np.array(rewards),
        np.array(values),
        np.array(ends),
        np.array(next_value),
        gamma,
        lam,
    )
    return mx.array(advantages), mx.array(value_targets)


# -- Losses --------------------------------------------------------------------
# Per-row loss terms over one minibatch; the caller reduces, masks and weights.


def clipped_surrogate(
    log_ratio: mx.array, advantages: mx.array, clip: float
) -> mx.array:
    """Negative clipped surrogate [B] (Schulman et al. 2017, eq. 7) from log ratios.

    PyTorch evaluates the ratio in float64 to cover sampled float32 actions;
    MLX's GPU has no float64, so the ratio is float32 here. Besides rounding
    differences, positive log ratios above about 88 can overflow. Nonfinite
    objectives or gradients are refused before the optimizer step; parity on
    captured inputs does not establish agreement outside them. Advantages are
    constants.
    """
    ratio = mx.exp(log_ratio)
    advantages = mx.stop_gradient(advantages)
    return -mx.minimum(
        ratio * advantages, mx.clip(ratio, 1 - clip, 1 + clip) * advantages
    )


def reverse_kl(logs: mx.array, target_logs: mx.array, valid: mx.array) -> mx.array:
    """KL(current || frozen target) [B], zero terms on padding without 0 * -inf."""
    difference = mx.where(valid, logs, 0.0) - mx.where(
        valid, mx.stop_gradient(target_logs), 0.0
    )
    return (mx.exp(logs) * difference).sum(-1)


def squared_value_loss(values: mx.array, targets: mx.array) -> mx.array:
    """Squared error [B] of a scalar critic's `[B]` or `[B,1]` predictions."""
    if values.shape != targets.shape:
        values = values.squeeze(-1)
    if values.shape != targets.shape or targets.ndim != 1:
        raise ValueError("scalar values and targets require [B]")
    return mx.square(values - targets)


def entropy(logs: mx.array, valid: mx.array) -> mx.array:
    """Policy entropy [B] of normalized log-probabilities `[B,A]` over `valid` entries."""
    return -(mx.exp(logs) * mx.where(valid, logs, 0.0)).sum(-1)


def damped_policy_loss(
    logits: mx.array,
    actions: mx.array,
    advantages: mx.array,
    behavior_logs: mx.array,
    reference: mx.array,
    valid: mx.array,
    *,
    clip: float,
    collection_kl: float,
    tau: float,
) -> mx.array:
    """The damped policy objective, equation (6), per row [B] over offers [B,A].

    Sources and departures: docs/learning.md#fidelity-to-the-sources. Ratios
    use the actual action likelihood at collection; the reverse KLs sum all
    legal actions.
    """
    if logits.ndim != 2 or logits.shape != valid.shape:
        raise ValueError("policy logits require [B,A] legal support")
    if actions.shape != logits.shape[:1] or advantages.shape != actions.shape:
        raise ValueError("actions and advantages require [B]")
    if not 0 < clip < 1 or collection_kl < 0 or tau < 0:
        raise ValueError("invalid damping coefficients")
    check_logs(behavior_logs, valid)
    check_probabilities(reference, valid, positive=True)
    require(
        mx.isfinite(advantages),
        "nonfinite_advantages",
        advantages=advantages,
        valid=valid,
    )
    if bool(((actions < 0) | (actions >= logits.shape[-1])).any()):
        raise ValueError("action index outside legal support")
    if not bool(mx.take_along_axis(valid, actions[:, None], axis=-1).all()):
        raise ValueError("sampled action is not legal")
    logs = policy_logs(logits, valid)
    behavior_logs = mx.stop_gradient(behavior_logs)
    # Each side is gathered before the difference: on padding both are -inf.
    ratio_logs = action_logs(logs, actions) - action_logs(behavior_logs, actions)
    return (
        clipped_surrogate(ratio_logs, advantages, clip)
        + collection_kl * reverse_kl(logs, behavior_logs, valid)
        + tau * reverse_kl(logs, reference_logs(reference, valid), valid)
    )


def clip_fraction(log_ratio: mx.array, clip: float) -> float:
    """Fraction of rows whose likelihood ratio lies outside `1 ± clip`."""
    outside = mx.abs(mx.exp(mx.stop_gradient(log_ratio)) - 1) > clip
    return float(outside.astype(mx.float32).mean())


# -- Selection -----------------------------------------------------------------


def selection_mask(
    advantages: mx.array,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> mx.array:
    """Boolean mask shaped like `advantages` keeping `fraction` of rows,
    |A| >= minimum, on the host (`manabot.policy.returns.selection_mask`)."""
    return mx.array(
        returns.selection_mask(np.array(advantages), kind, fraction, minimum)
    )


def selected_mean(values: mx.array, selected: mx.array) -> mx.array:
    """Mean of `values` [B] over the rows of bool `selected` [B]; an empty
    selection is zero with zero gradient."""
    count = selected.astype(mx.float32).sum()
    return mx.where(selected, values, 0.0).sum() / mx.maximum(count, 1)
