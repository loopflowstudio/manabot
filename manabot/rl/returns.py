"""Same-player lambda returns for transition-end rollout flags.

Each column must follow one player through an episode. A true end belongs to
that row's reward and cuts both bootstrap and trace, including across packed
seat trajectories. Gamma and lambda advance per retained player decision;
opponent and skipped forced decisions do not add discount steps.

Complete-game rollouts are one packed column of ~50,000 rows, so a recursion
over time is ~50,000 dependent steps. An end flag cuts the recursion, which makes
trajectories independent: `reverse_waves` advances every trajectory by one row
per step, on the host, in as many steps as the longest trajectory has rows. Each
row still takes the same float operations in the same order as a row-by-row
recursion, and device kernels of that size are launch latency, not work.
"""

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.nn import functional as F


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
    """GAE for transition-end flags; terminal rewards credit their own episode."""
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
    for distribution in (probabilities, bootstrap):
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
            raise ValueError("outcome probabilities must normalize")
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
