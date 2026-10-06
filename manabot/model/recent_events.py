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
        obs: dict[str, torch.Tensor],
        objects: torch.Tensor,
        definitions: torch.Tensor,
        definition_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,D], exactly zero for empty history; gradients reach both
        the event projection and shared semantic/object encoders. Padding is
        sanitized before any lookup, including nonfinite padded payloads.
        """
        if "events" not in obs or "events_valid" not in obs:
            raise ValueError("recent-event Agent requires events and events_valid")
        events, valid = obs["events"], obs["events_valid"]
        if (
            events.ndim != 3
            or events.shape[-1] != 12
            or valid.shape != events.shape[:2]
        ):
            raise ValueError("public history v1 requires [batch, events, 12]")
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError("event validity must be binary")
        admitted = valid.bool()
        rows = torch.where(admitted.unsqueeze(-1), events, 0)
        if not torch.isfinite(rows).all():
            raise ValueError("public history must be finite")

        def categorical(values: torch.Tensor, size: int) -> torch.Tensor:
            if not ((values == values.floor()) & (values >= 0) & (values < size)).all():
                raise ValueError("invalid public history categorical field")
            return F.one_hot(values.long(), size).to(events.dtype)

        counts = valid.sum(1, keepdim=True)
        age = (counts - valid.cumsum(1)).clamp_min(0)
        features = [
            categorical(rows[..., 0], 6),
            rows[..., 1:2] / 20,
            age.log1p().unsqueeze(-1),
        ]
        for start in (2, 7):
            ids, owner, zone, focus, status = rows[..., start : start + 5].unbind(-1)
            if not (
                (ids == ids.floor()) & (ids >= 0) & (ids < len(definition_rows))
            ).all():
                raise ValueError("unadmitted historical definition")
            catalog_rows = definition_rows[ids.long()]
            if (catalog_rows < 0).any():
                raise ValueError("unadmitted historical definition")
            if not (
                (focus == focus.floor()) & (focus >= -1) & (focus < objects.shape[1])
            ).all():
                raise ValueError("invalid historical context reference")
            context = objects.gather(
                1,
                focus.clamp_min(0)
                .long()
                .unsqueeze(-1)
                .expand(-1, -1, objects.shape[-1]),
            )
            context = context.masked_fill(((focus < 0) | ~admitted).unsqueeze(-1), 0)
            features.extend(
                (
                    definitions[catalog_rows],
                    context,
                    categorical(owner, 3),
                    categorical(zone, 8),
                    categorical(status, 3),
                )
            )
        encoded = self.projection(torch.cat(features, dim=-1))
        encoded = encoded.masked_fill(~admitted.unsqueeze(-1), 0)
        return encoded.sum(1) / counts.clamp_min(1)
