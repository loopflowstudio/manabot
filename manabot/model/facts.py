"""
Policy-input v5 fact rows, joined to the player or permanent each describes.

A fact row states one open-vocabulary fact (a counter kind, an added subtype, a
set name or creature type, a spent ability, a continuous-effect record) about
one object; `manabot.env.observation` documents the columns. Each row is
embedded from its kind, its vocabulary id and effect detail (learning-token
values, rebuilt from their 16-bit halves and hashed into `buckets` rows) and its
amounts, duration, timestamp rank and source, then summed into its object's
pre-attention embedding. The output layer starts at zero, so a fresh model's
objects are unchanged until the facts earn weight; gradients reach every fact
row that names a valid object.
"""

import torch
import torch.nn as nn

from manabot.env.observation import (
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_DURATION_INDEX,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_ROWS,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
)


class FactEncoder(nn.Module):
    def __init__(self, width: int, buckets: int = 4096) -> None:
        super().__init__()
        self.buckets = buckets
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        self.ident = nn.Embedding(buckets, width)
        self.detail = nn.Embedding(buckets, width)
        self.source = nn.Embedding(buckets, width)
        # Amounts (2), duration one-hot (2) and timestamp rank.
        self.numeric = nn.Linear(5, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _bucket(self, rows: torch.Tensor, index: int) -> torch.Tensor:
        """The vocabulary id at `index` (two 16-bit halves) as a bucket."""
        high = rows[..., index].long()
        low = rows[..., index + 1].long()
        return ((high << 16) | low) % self.buckets

    def forward(self, facts: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        """Add each valid fact's embedding to its object; `objects` is [B,N,D]."""
        if facts.shape[1:] != (FACT_ROWS, FACT_DIM):
            raise ValueError(f"expected [batch, {FACT_ROWS}, {FACT_DIM}] facts")
        valid = facts[..., FACT_VALID_INDEX]
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError("fact validity must be binary")
        facts = torch.where(valid.bool().unsqueeze(-1), facts, 0)
        links = facts[..., FACT_OBJECT_INDEX]
        kinds = facts[..., FACT_KIND_INDEX]
        if not (
            (links == links.floor()) & (links >= 0) & (links <= objects.shape[1])
        ).all():
            raise ValueError("invalid fact object reference")
        if not ((kinds == kinds.floor()) & (kinds >= 0) & (kinds <= FACT_KINDS)).all():
            raise ValueError("invalid fact kind")
        duration = facts[..., FACT_DURATION_INDEX]
        numeric = torch.stack(
            (
                facts[..., FACT_VALUE_INDEX],
                facts[..., FACT_VALUE_INDEX + 1],
                (duration == 1).float(),
                (duration == 2).float(),
                facts[..., FACT_ORDER_INDEX] / FACT_ROWS,
            ),
            dim=-1,
        )
        hidden = (
            self.kind(kinds.long())
            + self.ident(self._bucket(facts, FACT_ID_INDEX))
            + self.detail(self._bucket(facts, FACT_DETAIL_INDEX))
            + self.source(facts[..., FACT_SOURCE_INDEX].long() % self.buckets)
            + self.numeric(numeric)
        )
        encoded = self.output(torch.tanh(hidden))
        linked = (valid.bool() & (links > 0)).unsqueeze(-1)
        targets = (
            (links.long() - 1)
            .clamp_min(0)
            .unsqueeze(-1)
            .expand(-1, -1, objects.shape[-1])
        )
        delta = torch.zeros_like(objects).scatter_add(
            1, targets, encoded.masked_fill(~linked, 0)
        )
        return objects + delta
