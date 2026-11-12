"""Masked categorical policies with retained log-space support.

Normalized legal logs describe the policy; exp(logs) are the finite precision
sampling weights. Underflow in the latter is counted, never repaired with a
floor. Collection and learning share this normalization and legal mask.
"""

import torch
from torch import Tensor

from manabot.infra.deprecation import moved
from manabot.rl.numerics import NumericalError as _NumericalError, require as _require


def legal_mask(mask: Tensor) -> Tensor:
    _require(
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
        raise _NumericalError("policy_shape", {"logits": logits, "valid": valid})
    _require(valid.any(-1), "empty_legal_support", logits=logits, valid=valid)
    _require(
        torch.isfinite(logits) | ~valid,
        "nonfinite_legal_logits",
        logits=logits,
        valid=valid,
    )
    logs = logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    _require(
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
        raise _NumericalError(
            "distribution_shape", {"probabilities": probabilities, "valid": valid}
        )
    tensors = {"probabilities": probabilities, "valid": valid}
    _require(valid.any(-1), "empty_legal_support", **tensors)
    _require(torch.isfinite(probabilities), "nonfinite_probabilities", **tensors)
    _require(probabilities >= 0, "negative_probabilities", **tensors)
    if positive:
        _require((probabilities > 0) | ~valid, "nonpositive_legal_support", **tensors)
    _require((probabilities == 0) | valid, "illegal_probability_mass", **tensors)
    _require(
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
        raise _NumericalError("log_shape", {"logs": logs, "valid": valid})
    _require(valid.any(-1), "empty_legal_support", logs=logs, valid=valid)
    _require(
        torch.isfinite(logs) | ~valid, "nonfinite_behavior_logs", logs=logs, valid=valid
    )
    _require(torch.isneginf(logs) | valid, "illegal_log_mass", logs=logs, valid=valid)
    _require(
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
        raise _NumericalError("behavior_shape", tensors)
    _require(torch.isfinite(logs) | ~valid, "nonfinite_behavior_logs", **tensors)
    _require(torch.isneginf(logs) | valid, "illegal_log_mass", **tensors)
    _require(
        torch.isclose(
            logs.logsumexp(-1), torch.zeros_like(selected_logs), atol=1e-5, rtol=0
        ),
        "log_normalization",
        **tensors,
    )
    # Zero absolute tolerance prevents accepting fabricated tiny probabilities.
    _require(
        torch.isclose(logs.exp(), probabilities, atol=0, rtol=1e-5),
        "behavior_log_probability_mismatch",
        **tensors,
    )
    _require(
        (actions >= 0) & (actions < logs.shape[-1]), "action_out_of_range", **tensors
    )
    chosen = actions[..., None]
    _require(valid.gather(-1, chosen), "illegal_sampled_action", **tensors)
    _require(
        probabilities.gather(-1, chosen) > 0, "zero_sampled_probability", **tensors
    )
    _require(
        torch.isclose(
            logs.gather(-1, chosen).squeeze(-1), selected_logs, atol=1e-5, rtol=1e-5
        ),
        "selected_log_mismatch",
        **tensors,
    )


__getattr__ = moved(
    __name__,
    {
        "NumericalError": "manabot.rl.numerics",
        "require": "manabot.rl.numerics",
        "clipped_surrogate": "manabot.rl.losses",
        "reverse_kl": "manabot.rl.losses",
    },
)
