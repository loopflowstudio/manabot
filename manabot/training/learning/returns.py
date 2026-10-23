"""Same-player lambda returns for transition-end rollout flags.

Each column must follow one player through an episode. A true end belongs to
that row's reward and cuts both bootstrap and trace, including across packed
seat trajectories. Gamma and lambda advance per retained player decision;
opponent and skipped forced decisions do not add discount steps.
"""

import torch


def transition_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for transition-end flags; terminal rewards credit their own episode."""
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    following = next_value.reshape(-1)
    for step in reversed(range(len(rewards))):
        live = (~ends[step].bool()).to(rewards.dtype)
        delta = rewards[step] + gamma * live * following - values[step]
        tail = delta + gamma * lam * live * tail
        advantages[step] = tail
        following = values[step]
    return advantages, advantages + values
