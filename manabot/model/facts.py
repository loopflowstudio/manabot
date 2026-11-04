"""
Policy-input v5–v9 fact rows, joined to the player or permanent each describes.

A fact row states one open-vocabulary fact (a counter kind, an added subtype, a
set name or creature type, a spent ability, a continuous-effect record) about
one object. V8 adds restricted-mana classes and their predicates/output locations;
`manabot.env.observation` documents the columns. Each row is
embedded from its kind, its vocabulary id and effect detail (learning-token
values, rebuilt from their 16-bit halves and hashed into `buckets` rows) and its
amounts, duration, timestamp rank and source, then summed into its object's
pre-attention embedding. The output layer starts at zero, so a fresh model's
objects are unchanged until the facts earn weight; gradients reach every fact
row that names a valid object.

V9 retains v8 wire rows but joins mana amounts, predicates and output locations
within each class before nonlinear pooling. Class identifiers are routing keys,
never embedded features; row order and class renumbering leave the result alone.
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
    FACT_MANA_AMOUNT,
    FACT_MANA_LOCATION,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SOURCE_INDEX,
    FACT_TIMESTAMP_SCALE,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
)


class FactEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        max_facts: int,
        buckets: int = 4096,
        kinds: int = FACT_KINDS,
        *,
        join_mana: bool = False,
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.buckets = buckets
        self.max_facts = max_facts
        self.kind = nn.Embedding(kinds + 1, width)
        self.ident = nn.Embedding(buckets, width)
        self.detail = nn.Embedding(buckets, width)
        self.source = nn.Embedding(buckets, width)
        # Amounts (2), duration one-hot (2) and timestamp rank.
        self.numeric = nn.Linear(5, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.mana_join = nn.Linear(width, width) if join_mana else None

    def _bucket(self, rows: torch.Tensor, index: int) -> torch.Tensor:
        """The vocabulary id at `index` (two 16-bit halves) as a bucket."""
        high = rows[..., index].long()
        low = rows[..., index + 1].long()
        return ((high << 16) | low) % self.buckets

    def forward(self, facts: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        """Add each valid fact's embedding to its object; `objects` is [B,N,D]."""
        if facts.shape[1:] != (self.max_facts, FACT_DIM):
            raise ValueError(f"expected [batch, {self.max_facts}, {FACT_DIM}] facts")
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
        if not ((kinds == kinds.floor()) & (kinds >= 0) & (kinds <= self.kinds)).all():
            raise ValueError("invalid fact kind")
        duration = facts[..., FACT_DURATION_INDEX]
        numeric = torch.stack(
            (
                facts[..., FACT_VALUE_INDEX],
                facts[..., FACT_VALUE_INDEX + 1],
                (duration == 1).float(),
                (duration == 2).float(),
                facts[..., FACT_ORDER_INDEX] / FACT_TIMESTAMP_SCALE,
            ),
            dim=-1,
        )
        mana = None
        identifiers = self._bucket(facts, FACT_ID_INDEX)
        if self.mana_join is not None:
            mana = (
                (kinds >= FACT_MANA_AMOUNT)
                & (kinds <= FACT_MANA_LOCATION)
                & valid.bool()
                & (links > 0)
            )
            identifiers = identifiers.masked_fill(mana, 0)
        hidden = (
            self.kind(kinds.long())
            + self.ident(identifiers)
            + self.detail(self._bucket(facts, FACT_DETAIL_INDEX))
            + self.source(facts[..., FACT_SOURCE_INDEX].long() % self.buckets)
            + self.numeric(numeric)
        )
        encoded = self.output(torch.tanh(hidden))
        linked = (valid.bool() & (links > 0)).unsqueeze(-1)
        if mana is not None:
            linked = linked & ~mana.unsqueeze(-1)
        targets = (
            (links.long() - 1)
            .clamp_min(0)
            .unsqueeze(-1)
            .expand(-1, -1, objects.shape[-1])
        )
        delta = torch.zeros_like(objects).scatter_add(
            1, targets, encoded.masked_fill(~linked, 0)
        )
        if mana is not None and mana.any():
            delta = delta + self._mana_delta(facts, hidden, mana, objects)
        return objects + delta

    def _mana_delta(
        self,
        facts: torch.Tensor,
        hidden: torch.Tensor,
        mana: torch.Tensor,
        objects: torch.Tensor,
    ) -> torch.Tensor:
        """Pool complete classes, then route each to its owning object [B,N,D]."""
        assert self.mana_join is not None
        rows = facts[mana]
        halves = rows[:, FACT_ID_INDEX : FACT_ID_INDEX + 2]
        if not ((halves == halves.floor()) & (halves >= 0) & (halves <= 65535)).all():
            raise ValueError("invalid mana class identifier")
        groups = (halves[:, 0].long() << 16) | halves[:, 1].long()
        if (groups == 0).any():
            raise ValueError("mana class identifier must be positive")
        batch = torch.arange(len(facts), device=facts.device)[:, None].expand_as(mana)[
            mana
        ]
        keys = torch.stack((batch, rows[:, FACT_OBJECT_INDEX].long() - 1, groups), -1)
        classes, inverse = torch.unique(keys, dim=0, return_inverse=True)
        # Nonlinearity after the join is essential: summing each predicate and
        # amount independently would lose which restriction owns which amount.
        pooled = hidden.new_zeros((len(classes), hidden.shape[-1])).index_add(
            0, inverse, torch.tanh(hidden[mana])
        )
        encoded = self.output(torch.tanh(self.mana_join(pooled)))
        targets = classes[:, 0] * objects.shape[1] + classes[:, 1]
        return (
            torch.zeros_like(objects)
            .flatten(0, 1)
            .index_add(0, targets, encoded)
            .reshape_as(objects)
        )
