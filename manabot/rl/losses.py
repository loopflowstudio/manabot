"""Per-row loss terms over one minibatch; the caller reduces, masks and weights.

Every function returns one value per row, `[B]`. Policy terms read padded
categorical log-probabilities `[B,A]` that are `-inf` off a `valid` mask
`[B,A]`, and never read the padding. Targets, behavior likelihoods and
references are constants: gradients reach only the current policy and value
outputs.
"""

import torch
from torch import Tensor

from manabot.rl.numerics import float64_operand


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


def reverse_kl(logs: Tensor, target_logs: Tensor, valid: Tensor) -> Tensor:
    """KL(current || frozen target) [B], zero terms on padding without 0 * -inf."""
    difference = logs.masked_fill(~valid, 0) - target_logs.detach().masked_fill(
        ~valid, 0
    )
    return (logs.exp() * difference).sum(-1)


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
