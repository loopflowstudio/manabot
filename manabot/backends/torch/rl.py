"""The standard reinforcement-learning functions over torch tensors.

Returns (GAE, standardized advantages, categorical lambda returns), per-row
loss terms, advantage selection and the named numerical refusals they raise.
They read tensors and small typed records and import nothing else from
manabot (`tests/test_package_boundaries.py` enforces it), so another role
reuses them unchanged. Every backend under `manabot.backends` has the same
functions over its own arrays; the gradient step and EMA live in
`manabot.backends.torch.learner`.
"""

from collections.abc import Iterator, Mapping
import math
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

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


def float64_operand(tensor: Tensor) -> Tensor:
    """Move MPS operands to the host, where float64 is supported.

    MPS has no float64 tensors. The CPU transfer preserves autograd; callers
    retain their explicit double casts and keep companion operands together.
    Elsewhere this is a no-op relabel: the identical tensor, dtype and device
    are returned, with no transfer or additional tensor operation.
    """
    return tensor.cpu() if tensor.device.type == "mps" else tensor


# -- Returns -------------------------------------------------------------------
# Same-player lambda returns for transition-end rollout flags.
#
# Each column must follow one player through an episode. A true end belongs to
# that row's reward and cuts both bootstrap and trace, including across packed
# seat trajectories. Gamma and lambda advance per retained player decision;
# opponent and skipped forced decisions do not add discount steps.
#
# Complete-game rollouts are one packed column of ~50,000 rows, so a recursion
# over time is ~50,000 dependent steps. An end flag cuts the recursion, which makes
# trajectories independent: `reverse_waves` advances every trajectory by one row
# per step, on the host, in as many steps as the longest trajectory has rows. Each
# row still takes the same float operations in the same order as a row-by-row
# recursion, and device kernels of that size are launch latency, not work.


def terminal_distances(ends: torch.Tensor) -> torch.Tensor:
    """Return [T,E] distances (terminal=0, censored=-1), reset at each end."""
    steps = torch.arange(len(ends), device=ends.device).unsqueeze(1)
    # The nearest end at or after each row, by a running minimum toward the start.
    following = torch.where(ends, steps, len(ends)).flip(0).cummin(0).values.flip(0)
    return torch.where(following < len(ends), following - steps, -1)


def reverse_waves(ends: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (time, column) rows of [T,E] bool flags, each after the row it reads.

    A trajectory ends at an end flag or a column's last row. Wave k holds the row
    k from the end of every trajectory with more than k rows.
    """
    last = ends.clone()
    last[-1] = True
    waves = terminal_distances(last).flatten()
    steps, columns = torch.unravel_index(waves.argsort(stable=True), ends.shape)
    sizes = waves.bincount().tolist()
    return zip(steps.split(sizes), columns.split(sizes), strict=True)


def transition_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE (Schulman et al. 2016): advantages and lambda returns, both [T,E].

    `rewards`, `values` and `ends` are [T,E]; terminal rewards credit their own
    episode. `next_value` [E] is read only where a column's last row is not an
    end. Mismatched shapes raise `ValueError` rather than broadcast.
    """
    if (
        rewards.ndim != 2
        or values.shape != rewards.shape
        or ends.shape != rewards.shape
        or next_value.numel() != rewards.shape[1]
    ):
        raise ValueError(
            "transition_gae requires rewards, values and ends [T,E] and next_value [E]; "
            f"got {tuple(rewards.shape)}, {tuple(values.shape)}, "
            f"{tuple(ends.shape)} and {tuple(next_value.shape)}"
        )
    host_ends = ends.cpu().bool()
    host_values = values.cpu()
    live = (~host_ends).to(rewards.dtype)
    following = torch.cat([host_values[1:], next_value.cpu().reshape(1, -1)])
    deltas = rewards.cpu() + gamma * live * following - host_values
    carried = gamma * lam * live
    # The extra zero row is the empty tail past each column's last row.
    advantages = torch.zeros(len(rewards) + 1, rewards.shape[1], dtype=rewards.dtype)
    for steps, columns in reverse_waves(host_ends):
        advantages[steps, columns] = (
            deltas[steps, columns]
            + carried[steps, columns] * advantages[steps + 1, columns]
        )
    moved = advantages[:-1].to(rewards.device)
    return moved, moved + values


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


@torch.no_grad()
def categorical_lambda_returns(
    probabilities: Tensor,
    rewards: Tensor,
    ends: Tensor,
    bootstrap: Tensor,
    trace_lambda: float,
) -> Tensor:
    """Undiscounted outcome mixtures [T,E,3], ordered loss/draw/win.

    Each row ends at the next decision of the SAME player or at terminal.
    Terminal targets are one-hot signed outcomes. A nonterminal tail uses the
    exact paused next observation; reset observations never cross end markers.
    For earlier rows mix (1-lambda)*next_prediction + lambda*next_return.
    No policy receives true hidden-state input from these outcome labels.
    """
    if probabilities.ndim != 3 or probabilities.shape[-1] != 3:
        raise ValueError("outcome probabilities require [T,E,3]")
    if (
        rewards.shape != probabilities.shape[:2]
        or ends.shape != rewards.shape
        or bootstrap.shape != probabilities.shape[1:]
        or not len(rewards)
        or not 0 <= trace_lambda <= 1
    ):
        raise ValueError("invalid outcome target shapes or lambda")
    for name, distribution in (
        ("probabilities", probabilities),
        ("bootstrap", bootstrap),
    ):
        if (
            not torch.isfinite(distribution).all()
            or (distribution < 0).any()
            or not torch.allclose(
                distribution.sum(-1),
                torch.ones_like(distribution.sum(-1)),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise ValueError(f"outcome {name} must be distributions over 3 outcomes")
    if (
        not torch.isfinite(rewards).all()
        or not ((rewards == -1) | (rewards == 0) | (rewards == 1)).all()
    ):
        raise ValueError("outcome rewards must be signed win/loss/draw")
    ends = ends.bool()
    if (rewards[~ends] != 0).any():
        raise ValueError("nonterminal outcome reward must be zero")
    # Trajectories run in parallel on the host (`reverse_waves`), row order intact.
    host_probabilities = probabilities.cpu()
    host_ends = ends.cpu()
    last_row = bootstrap.cpu()[None]
    following = torch.cat([host_probabilities[1:], last_row])
    terminal = F.one_hot((rewards.cpu() + 1).long(), 3).to(host_probabilities.dtype)
    # The extra row is the trace past each column's last row.
    targets = torch.cat([torch.empty_like(host_probabilities), last_row])
    for steps, columns in reverse_waves(host_ends):
        continuation = (1 - trace_lambda) * following[
            steps, columns
        ] + trace_lambda * targets[steps + 1, columns]
        targets[steps, columns] = torch.where(
            host_ends[steps, columns, None], terminal[steps, columns], continuation
        )
    return targets[:-1].to(probabilities.device)


# -- Losses --------------------------------------------------------------------
# Per-row loss terms over one minibatch; the caller reduces, masks and weights.
#
# Every loss returns one value per row, `[B]`; `support_logs` builds a frozen
# target for `reverse_kl` and `clip_fraction` is the surrogate's diagnostic. Policy terms read padded
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


@torch.no_grad()
def clip_fraction(log_ratio: Tensor, clip: float) -> float:
    """Fraction of rows whose likelihood ratio lies outside `1 ± clip`.

    The ratio is exponentiated in float64 so large log ratios do not overflow.
    """
    return float(
        ((float64_operand(log_ratio).double().exp() - 1).abs() > clip).float().mean()
    )


def support_logs(probabilities: Tensor, valid: Tensor) -> Tensor:
    """Log-probabilities `[B,A]` of a distribution, zero off `valid` (log 1).

    `reverse_kl` never reads the padding; filling it with 1 keeps `log` finite.
    """
    return probabilities.masked_fill(~valid, 1).log()


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
    if kind == "quantile":
        return selected_moves(advantages, 1 - fraction, minimum).nonzero().flatten()
    magnitude = advantages.abs()
    count = max(1, math.ceil(len(magnitude) * fraction))
    indices = torch.argsort(magnitude, descending=True, stable=True)[:count]
    return indices[magnitude[indices] >= minimum]


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


def selected_mean(values: Tensor, selected: Tensor) -> Tensor:
    """Selected mean, on the host for MPS; empty support has zero gradient."""
    values, selected = float64_operand(values), float64_operand(selected)
    return values[selected].sum() / selected.sum().clamp_min(1)
