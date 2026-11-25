"""The standard reinforcement-learning functions over torch tensors.

The masked policy distribution (legal mask, normalized logs and the checks on
collected behavior), returns (GAE, standardized advantages, categorical lambda
returns), per-row loss terms for the shared damped policy objective, advantage
selection and the named numerical refusals they raise. The updates in
`manabot.training.updates` reach these as `backend.rl`.
They read tensors and small typed records and import nothing else from
manabot beyond framework-free numerical code
(`tests/test_package_boundaries.py` enforces it), so another role reuses them
unchanged. Other backends expose matching functions over their own arrays
for the operations they support; the gradient step and EMA live in
`manabot.backends.torch.learner`.
"""

from collections.abc import Mapping
import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from manabot.backends.torch.arrays import from_numpy, to_numpy
import manabot.policy.returns as returns

# -- Numerical refusals --------------------------------------------------------
# Named numerical refusals, and the host boundary for float64 arithmetic.
#
# A learning piece that cannot trust a tensor raises `NumericalError` naming
# the invariant it checked, carrying the tensors that broke it; callers attach
# more evidence and decide whether the run survives.


class NumericalError(ValueError):
    """An identified invariant plus tensors for bounded private failure capture."""

    def __init__(self, invariant: str, tensors: Mapping[str, Tensor]) -> None:
        super().__init__(invariant)
        self.invariant = invariant
        self.health: dict[str, int | float] = {}
        self.tensors = {key: value.detach() for key, value in tensors.items()}


def require(condition: Tensor, invariant: str, **tensors: Tensor) -> None:
    """Raise `NumericalError(invariant)` unless every element of `condition` holds."""
    if not bool(condition.all()):
        raise NumericalError(invariant, {**tensors, "failed": ~condition})


def require_finite(invariant: str, **tensors: Tensor) -> None:
    """Raise `NumericalError(invariant)` unless every element of each tensor is finite."""
    for tensor in tensors.values():
        require(torch.isfinite(tensor), invariant, **tensors)


def float64_operand(tensor: Tensor) -> Tensor:
    """Move MPS operands to the host, where float64 is supported.

    MPS has no float64 tensors. The CPU transfer preserves autograd; callers
    retain their explicit double casts and keep companion operands together.
    Elsewhere this is a no-op relabel: the identical tensor, dtype and device
    are returned, with no transfer or additional tensor operation.
    """
    return tensor.cpu() if tensor.device.type == "mps" else tensor


def constant(tensor: Tensor) -> Tensor:
    """The same values with no gradient path: targets, evidence, diagnostics."""
    return tensor.detach()


# -- Policy distribution -------------------------------------------------------
# Masked categorical policies with retained log-space support.
#
# Normalized legal logs describe the policy; exp(logs) are the finite precision
# sampling weights, used only on the collecting device. Health derives host
# float32 exponentials from saved logs, counting underflow without a floor.
# Collection and learning share this normalization and legal mask.


def legal_mask(mask: Tensor) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    require(
        torch.isfinite(mask) & ((mask == 0) | (mask == 1)),
        "invalid_action_mask",
        mask=mask,
    )
    return mask.bool()


def policy_logs(logits: Tensor, valid: Tensor) -> Tensor:
    """Normalize [*,A] logits; padding is -inf and legal logs must be finite.

    log_softmax subtracts the maximum before reduction, avoiding cancellation
    from subtracting a large absolute logsumexp. Unrepresentable log differences
    fail explicitly even when the original logits were finite.
    """
    if logits.shape != valid.shape or logits.ndim < 2:
        raise NumericalError("policy_shape", {"logits": logits, "valid": valid})
    require(valid.any(-1), "empty_legal_support", logits=logits, valid=valid)
    require(
        torch.isfinite(logits) | ~valid,
        "nonfinite_legal_logits",
        logits=logits,
        valid=valid,
    )
    logs = logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    require(
        torch.isfinite(logs) | ~valid,
        "nonfinite_legal_logs",
        logs=logs,
        logits=logits,
        valid=valid,
    )
    return logs


def action_logs(logs: Tensor, actions: Tensor) -> Tensor:
    """The entry of each row of `logs` [B,A] at its sampled action [B]: [B]."""
    return logs.gather(-1, actions[:, None]).squeeze(-1)


def reference_logs(reference: Tensor, valid: Tensor) -> Tensor:
    """Log probabilities [B,A] of a constant reference, zero off `valid`."""
    return reference.detach().masked_fill(~valid, 1).log()


def check_probabilities(
    probabilities: Tensor, valid: Tensor, *, positive: bool
) -> None:
    if probabilities.shape != valid.shape or probabilities.ndim < 2:
        raise NumericalError(
            "distribution_shape", {"probabilities": probabilities, "valid": valid}
        )
    tensors = {"probabilities": probabilities, "valid": valid}
    require(valid.any(-1), "empty_legal_support", **tensors)
    require(torch.isfinite(probabilities), "nonfinite_probabilities", **tensors)
    require(probabilities >= 0, "negative_probabilities", **tensors)
    if positive:
        require((probabilities > 0) | ~valid, "nonpositive_legal_support", **tensors)
    require((probabilities == 0) | valid, "illegal_probability_mass", **tensors)
    require(
        torch.isclose(
            probabilities.sum(-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ),
        "probability_normalization",
        **tensors,
    )


def check_logs(logs: Tensor, valid: Tensor) -> None:
    if logs.shape != valid.shape:
        raise NumericalError("log_shape", {"logs": logs, "valid": valid})
    require(valid.any(-1), "empty_legal_support", logs=logs, valid=valid)
    require(
        torch.isfinite(logs) | ~valid, "nonfinite_behavior_logs", logs=logs, valid=valid
    )
    require(torch.isneginf(logs) | valid, "illegal_log_mass", logs=logs, valid=valid)
    require(
        torch.isclose(
            logs.logsumexp(-1), torch.zeros_like(logs[..., 0]), atol=1e-5, rtol=0
        ),
        "log_normalization",
        logs=logs,
        valid=valid,
    )


def check_behavior(
    logs: Tensor,
    actions: Tensor,
    selected_logs: Tensor,
    valid: Tensor,
) -> None:
    """Admit collection evidence: normalized logs [*,A] and the action taken.

    The logs are the only stored behavior distribution. A legal log stays
    finite however unlikely the action; its exponential is never compared.
    The saved selected log must be the log of a legal recorded action.
    """
    check_logs(logs, valid)
    tensors = dict(logs=logs, actions=actions, selected_logs=selected_logs, valid=valid)
    if actions.shape != valid.shape[:-1] or selected_logs.shape != actions.shape:
        raise NumericalError("behavior_shape", tensors)
    require(
        (actions >= 0) & (actions < logs.shape[-1]), "action_out_of_range", **tensors
    )
    chosen = actions[..., None]
    require(valid.gather(-1, chosen), "illegal_sampled_action", **tensors)
    require(
        torch.isclose(
            logs.gather(-1, chosen).squeeze(-1), selected_logs, atol=1e-5, rtol=1e-5
        ),
        "selected_log_mismatch",
        **tensors,
    )


# -- Returns -------------------------------------------------------------------
# Same-player lambda returns for transition-end rollout flags.
#
# Each column must follow one player through an episode. A true end belongs to
# that row's reward and cuts both bootstrap and trace, including across packed
# seat trajectories. Gamma and lambda advance per retained player decision;
# opponent and skipped forced decisions do not add discount steps.
#
# Complete-game rollouts are one packed column of ~50,000 rows, so a recursion
# over time is ~50,000 dependent steps. An end flag cuts the recursion, which
# makes trajectories independent: `manabot.policy.returns.reverse_waves`
# advances every trajectory by one row per step, on the host, in as many steps
# as the longest trajectory has rows. Each row still takes the same float
# operations in the same order as a row-by-row recursion, and device kernels of
# that size are launch latency, not work. GAE is that module's NumPy function;
# the categorical targets use the same host recursion in both backends.


def transition_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
    bootstrap_indexes: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for transition-end flags [T,E], on the host
    (`manabot.policy.returns.transition_gae`), back on the rewards' device."""
    advantages, value_targets = returns.transition_gae(
        to_numpy(rewards),
        to_numpy(values),
        to_numpy(ends),
        to_numpy(next_value),
        gamma,
        lam,
        bootstrap_indexes,
    )
    return (
        from_numpy(advantages, rewards.device),
        from_numpy(value_targets, rewards.device),
    )


def standardized_advantages(advantages: Tensor, over: Tensor | None = None) -> Tensor:
    """Shift and scale every row by the mean and population std of `over`'s rows.

    `over` indexes the rows that set the statistics (all rows when omitted); the
    result keeps every row, so rows outside `over` are rescaled the same way.
    The std is clamped at 1e-8, so a constant batch centres to zeros.
    """
    reference = advantages if over is None else advantages[over]
    return (advantages - reference.mean()) / reference.std(unbiased=False).clamp_min(
        1e-8
    )


def expected_outcome(probabilities: Tensor) -> Tensor:
    """Expected signed outcome `[...]` of loss/draw/win mixtures `[...,3]`: P(win) - P(loss)."""
    return probabilities[..., 2] - probabilities[..., 0]


def outcome_probabilities(value_logits: Tensor) -> Tensor:
    """Loss/draw/win mixtures `[...,3]` of a categorical critic's logits."""
    return value_logits.softmax(-1)


def categorical_lambda_returns(
    probabilities: Tensor,
    rewards: Tensor,
    ends: Tensor,
    bootstrap: Tensor,
    trace_lambda: float,
    bootstrap_indexes: np.ndarray | None = None,
) -> Tensor:
    """Frozen loss/draw/win lambda targets on the predictions' device."""
    targets = returns.categorical_lambda_returns(
        to_numpy(probabilities),
        to_numpy(rewards),
        to_numpy(ends),
        to_numpy(bootstrap),
        trace_lambda,
        bootstrap_indexes,
    )
    return from_numpy(targets, probabilities.device).to(
        torch.promote_types(probabilities.dtype, bootstrap.dtype)
    )


# -- Losses --------------------------------------------------------------------
# Per-row loss terms over one minibatch; the caller reduces, masks and weights.
#
# Every function returns one value per row, `[B]`. Policy terms read padded
# categorical log-probabilities `[B,A]` that are `-inf` off a `valid` mask
# `[B,A]`, and never read the padding. Targets, behavior likelihoods and
# references are constants: gradients reach only the current policy and value
# outputs.


def clipped_surrogate(log_ratio: Tensor, advantages: Tensor, clip: float) -> Tensor:
    """Negative clipped surrogate [B] (Schulman et al. 2017, eq. 7) from log ratios.

    Double ratios avoid float32 exp overflow: a sampled float32 action has log
    likelihood >= about -104, which float64 ratios cover without changing
    clipping or dropping unbounded bad moves. Advantages are constants.
    Nonfinite objectives/gradients are still rejected before the step.
    MPS returns host float64 terms with gradients connected to the input.
    """
    ratio = float64_operand(log_ratio).double().exp()
    advantages = float64_operand(advantages)
    return -torch.minimum(
        ratio * advantages.detach(),
        ratio.clamp(1 - clip, 1 + clip) * advantages.detach(),
    )


def legal_kl(left: Tensor, right: Tensor, valid: Tensor) -> Tensor:
    """KL(left || right) per row, masking log padding before subtraction."""
    difference = left.masked_fill(~valid, 0) - right.masked_fill(~valid, 0)
    return (left.exp() * difference).sum(-1)


def reverse_kl(logs: Tensor, target_logs: Tensor, valid: Tensor) -> Tensor:
    """KL(current || frozen target) [B] over legal offers."""
    return legal_kl(logs, target_logs.detach(), valid)


def forward_kl(logs: Tensor, target_logs: Tensor, valid: Tensor) -> Tensor:
    """KL(frozen target || current) [B] over legal offers."""
    return legal_kl(target_logs.detach(), logs, valid)


def squared_value_loss(values: Tensor, targets: Tensor) -> Tensor:
    """Squared error [B] of a scalar critic's `[B]` or `[B,1]` predictions."""
    if values.shape != targets.shape:
        values = values.squeeze(-1)
    if values.shape != targets.shape or targets.ndim != 1:
        raise ValueError("scalar values and targets require [B]")
    return (values - targets).square()


def categorical_value_loss(value_logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy [B] of loss/draw/win logits `[B,3]` against target mixtures.

    Targets are `categorical_lambda_returns` rows (`returns`), so this is the
    outcome objective of the damped rule, minimized where the predicted
    distribution equals the target mixture.
    """
    if value_logits.ndim != 2 or value_logits.shape[-1] != 3:
        raise ValueError("categorical values require [B,3] logits")
    if targets.shape != value_logits.shape:
        raise ValueError("categorical targets require [B,3]")
    return -(targets * value_logits.log_softmax(-1)).sum(-1)


def entropy(logs: Tensor, valid: Tensor) -> Tensor:
    """Policy entropy [B] of normalized log-probabilities `[B,A]` over `valid` entries."""
    return -(logs.exp() * logs.masked_fill(~valid.bool(), 0)).sum(-1)


@torch.no_grad()
def clip_fraction(log_ratio: Tensor, clip: float) -> float:
    """Fraction of rows whose likelihood ratio lies outside `1 ± clip`.

    The ratio is exponentiated in float64 so large log ratios do not overflow.
    """
    return float(
        ((float64_operand(log_ratio).double().exp() - 1).abs() > clip).float().mean()
    )


# -- Selection -----------------------------------------------------------------
# Advantage selection: which rows a policy loss reads, and its masked mean.
#
# Both filters rank rows by advantage magnitude. A top-count filter keeps a
# fixed share (stable ties); a quantile filter keeps every row at or above an
# inclusive magnitude quantile, so ties are never split.


def selected_rows(
    advantages: Tensor,
    fraction: float,
    minimum: float,
    *,
    kind: Literal["top_count", "quantile"] = "top_count",
) -> Tensor:
    """Select [B] indexes, preserving each learner's pre-shuffle order.

    Top-count sorts by magnitude with stable ties; quantiles retain row order.
    """
    return selected_magnitudes(advantages, kind, fraction, minimum)[0]


def selected_magnitudes(
    advantages: Tensor,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> tuple[Tensor, float | None]:
    """Retained flat indexes and empirical cutoff before the absolute floor."""
    magnitude = advantages.detach().abs().flatten()
    if kind == "quantile":
        cutoff = float(torch.quantile(magnitude, 1 - fraction))
        return (magnitude >= max(cutoff, minimum)).nonzero().flatten(), cutoff
    count = max(1, math.ceil(len(magnitude) * fraction))
    indices = torch.argsort(magnitude, descending=True, stable=True)[:count]
    cutoff = float(magnitude[indices[-1]]) if len(indices) else None
    return indices[magnitude[indices] >= minimum], cutoff


def magnitude_quantile(advantages: Tensor, quantile: float) -> float:
    """The inclusive `quantile` of |advantages| over every row."""
    return float(torch.quantile(advantages.detach().abs(), quantile))


def selected_moves(advantages: Tensor, quantile: float, minimum: float) -> Tensor:
    """Inclusive magnitude quantile over all [T,E] rows; retain every tie."""
    threshold = max(magnitude_quantile(advantages, quantile), minimum)
    return advantages.detach().abs() >= threshold


def selection_mask(
    advantages: Tensor,
    kind: Literal["top_count", "quantile"],
    fraction: float,
    minimum: float,
) -> Tensor:
    """Boolean mask shaped like `advantages` keeping `fraction` of rows, |A| >= minimum."""
    mask = torch.zeros_like(advantages, dtype=torch.bool).flatten()
    mask[selected_rows(advantages.flatten(), fraction, minimum, kind=kind)] = True
    return mask.reshape_as(advantages)


def weighted_sum(values: Tensor, weights: Tensor) -> Tensor:
    """Sum per-row losses with constant logical-minibatch coefficients."""
    return (float64_operand(values) * float64_operand(weights)).sum()


def selected_mean(values: Tensor, selected: Tensor) -> Tensor:
    """Selected mean, on the host for MPS; empty support has zero gradient."""
    values, selected = float64_operand(values), float64_operand(selected)
    return values[selected].sum() / selected.sum().clamp_min(1)
