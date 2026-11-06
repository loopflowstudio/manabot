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
