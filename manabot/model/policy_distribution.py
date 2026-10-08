"""Masked categorical policies with retained log-space support.

Normalized legal logs describe the policy; exp(logs) are the finite precision
sampling weights. Underflow in the latter is counted, never repaired with a
floor. Collection and learning share this normalization and legal mask.
"""

from collections.abc import Mapping

import torch
from torch import Tensor


class NumericalError(ValueError):
    """An identified invariant plus tensors for bounded private failure capture."""

    def __init__(self, invariant: str, tensors: Mapping[str, Tensor]) -> None:
        super().__init__(invariant)
        self.invariant = invariant
        self.health: dict[str, int | float] = {}
        self.tensors = {key: value.detach() for key, value in tensors.items()}


def require(condition: Tensor, invariant: str, **tensors: Tensor) -> None:
    if not bool(condition.all()):
        raise NumericalError(invariant, {**tensors, "failed": ~condition})


def legal_mask(mask: Tensor) -> Tensor:
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
    probabilities: Tensor,
    actions: Tensor,
    selected_logs: Tensor,
    valid: Tensor,
) -> None:
    """Admit copied collection evidence, including rounded legal zeros.

    Zero is accepted only when exp(the saved finite legal log) also rounds to
    zero. Positive sampled mass and its selected log identity remain mandatory.
    """
    check_probabilities(probabilities, valid, positive=False)
    tensors = dict(
        logs=logs,
        probabilities=probabilities,
        actions=actions,
        selected_logs=selected_logs,
        valid=valid,
    )
    if (
        logs.shape != valid.shape
        or actions.shape != valid.shape[:-1]
        or selected_logs.shape != actions.shape
    ):
        raise NumericalError("behavior_shape", tensors)
    require(torch.isfinite(logs) | ~valid, "nonfinite_behavior_logs", **tensors)
    require(torch.isneginf(logs) | valid, "illegal_log_mass", **tensors)
    require(
        torch.isclose(
            logs.logsumexp(-1), torch.zeros_like(selected_logs), atol=1e-5, rtol=0
        ),
        "log_normalization",
        **tensors,
    )
    # Zero absolute tolerance prevents accepting fabricated tiny probabilities.
    require(
        torch.isclose(logs.exp(), probabilities, atol=0, rtol=1e-5),
        "behavior_log_probability_mismatch",
        **tensors,
    )
    require(
        (actions >= 0) & (actions < logs.shape[-1]), "action_out_of_range", **tensors
    )
    chosen = actions[..., None]
    require(valid.gather(-1, chosen), "illegal_sampled_action", **tensors)
    require(probabilities.gather(-1, chosen) > 0, "zero_sampled_probability", **tensors)
    require(
        torch.isclose(
            logs.gather(-1, chosen).squeeze(-1), selected_logs, atol=1e-5, rtol=1e-5
        ),
        "selected_log_mismatch",
        **tensors,
    )


def reverse_kl(logs: Tensor, target_logs: Tensor, valid: Tensor) -> Tensor:
    """KL(current || frozen target), zero terms on padding without 0 * -inf."""
    difference = logs.masked_fill(~valid, 0) - target_logs.detach().masked_fill(
        ~valid, 0
    )
    return (logs.exp() * difference).sum(-1)


def clipped_surrogate(log_ratio: Tensor, advantages: Tensor, clip: float) -> Tensor:
    """Negative clipped surrogate; double ratios avoid float32 exp overflow.

    A sampled float32 action has log likelihood >= about -104. Float64 ratios
    cover that range without changing clipping or dropping unbounded bad moves.
    Nonfinite objectives/gradients are still rejected before Adam.
    """
    ratio = log_ratio.double().exp()
    return -torch.minimum(
        ratio * advantages.detach(),
        ratio.clamp(1 - clip, 1 + clip) * advantages.detach(),
    )
