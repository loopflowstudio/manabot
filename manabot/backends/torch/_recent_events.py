"""Stateless identity-aware encoding of managym public history v1.

Rows [B,E,12] contain kind/amount and two typed references. Definition transport
IDs join the same semantic catalog as visible cards. Context indexes gather
current objects, never become numeric features; departed objects keep only their
public definition, historical zone and owner. The native ledger owns visibility,
incarnation boundaries, the chronological suffix and reset.
"""

import torch
from torch import nn
import torch.nn.functional as F

from manabot.policy.prepare import EventRows


class RecentEventEncoder(nn.Module):
    """Order-sensitive shared context; no memory survives a forward call."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # kind(6), amount/20, log ordinal age; two references each with semantic
        # definition + current context + owner(3), zone(8), availability(3).
        self.projection = nn.Sequential(
            nn.Linear(8 + 2 * (2 * width + 14), width),
            nn.Tanh(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        rows: EventRows[torch.Tensor],
        objects: torch.Tensor,
        definitions: torch.Tensor,
    ) -> torch.Tensor:
        """Pool history in chronological order; an empty history returns zero."""

        def categorical(values: torch.Tensor, size: int) -> torch.Tensor:
            return F.one_hot(values, size).to(rows.amounts.dtype)

        features = [
            categorical(rows.kinds, 6),
            rows.amounts,
            rows.ages.log1p().unsqueeze(-1),
        ]
        for index in range(2):
            context = objects.flatten(0, 1)[rows.focus[..., index]]
            context = context.masked_fill(~rows.focus_mask[..., index, None], 0)
            features.extend(
                (
                    definitions[rows.definitions[..., index]],
                    context,
                    categorical(rows.owners[..., index], 3),
                    categorical(rows.zones[..., index], 8),
                    categorical(rows.status[..., index], 3),
                )
            )
        encoded = self.projection(torch.cat(features, dim=-1))
        encoded = encoded.masked_fill(~rows.validity.unsqueeze(-1), 0)
        return encoded.sum(1) / rows.counts.clamp_min(1)
