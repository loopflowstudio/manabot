"""Stateless current-decision context from current stack/continuation relations.

Rows [B,64,32/33] retain frame order, parent links and target clauses. Definition
IDs join the existing catalog; object indexes gather the current object table.
Neither identifier is used as a numeric feature. Gradients reach the shared
object/catalog encoders and this projection; no memory survives a decision.
"""

import torch
from torch import nn

from manabot.env.observation import RELATION_DIM, RELATION_ROWS


class ContinuationEncoder(nn.Module):
    def __init__(self, width: int, relation_dim: int = RELATION_DIM) -> None:
        super().__init__()
        self.relation_dim = relation_dim
        self.projection = nn.Sequential(
            nn.Linear(relation_dim - 5 + 4 * width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )

    def encode_rows(
        self,
        rows: torch.Tensor,
        objects: torch.Tensor,
        definitions: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,R,D], zero for padding; shared by pooling and frame joins."""
        if rows.shape[1:] != (RELATION_ROWS, self.relation_dim):
            raise ValueError(f"expected [batch, 64, {self.relation_dim}] relations")
        valid = rows[..., 31]
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError("relation validity must be binary")
        rows = torch.where(valid.bool().unsqueeze(-1), rows, 0)
        if not torch.isfinite(rows).all():
            raise ValueError("relations must be finite")
        # Waiting occurrence order is presentation, unlike public stack order.
        # Retain its causal/ability facts without treating frame joins as features.
        features_rows = rows.clone()
        pending = (rows[..., 3] == 7) | (rows[..., 3] == 8)
        features_rows[..., 1:3] = torch.where(pending.unsqueeze(-1), 0, rows[..., 1:3])
        features = [
            torch.cat(
                (
                    features_rows[..., :5],
                    features_rows[..., 7:23],
                    features_rows[..., 25:31],
                ),
                -1,
            )
        ]
        if self.relation_dim == RELATION_DIM:
            features.append(rows[..., 32:33])
        for column in (6, 24):
            indexes = rows[..., column]
            if not (
                (indexes == indexes.floor())
                & (indexes >= 0)
                & (indexes <= objects.shape[1])
            ).all():
                raise ValueError("invalid continuation object reference")
            linked = objects.gather(
                1,
                (indexes.long() - 1)
                .clamp_min(0)
                .unsqueeze(-1)
                .expand(-1, -1, objects.shape[-1]),
            )
            features.append(linked.masked_fill((indexes == 0).unsqueeze(-1), 0))
        features.extend((definitions[:, :, 0], definitions[:, :, 1]))
        encoded = self.projection(torch.cat(features, -1))
        encoded = encoded.masked_fill((valid == 0).unsqueeze(-1), 0)
        return encoded

    def forward(
        self, rows: torch.Tensor, objects: torch.Tensor, definitions: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encode_rows(rows, objects, definitions)
        return encoded.sum(1) / rows[..., 31].sum(1, keepdim=True).clamp_min(1)

    @staticmethod
    def frame_context(rows: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Pool every relation of a frame, preserving source, target and causal context.

        Frame indexes are joins only. Gradients flow through all related rows.
        """
        ids = rows[..., 1]
        valid = rows[..., 31]
        if not (
            ((ids == ids.floor()) & (ids >= 1) & (ids <= RELATION_ROWS)) | (valid == 0)
        ).all():
            raise ValueError("invalid continuation frame reference")
        indexes = (torch.where(valid.bool(), ids, 1).long() - 1).clamp_min(0)
        pooled = torch.zeros_like(encoded).scatter_add(
            1, indexes.unsqueeze(-1).expand_as(encoded), encoded
        )
        counts = torch.zeros_like(valid).scatter_add(1, indexes, valid)
        return pooled / counts.clamp_min(1).unsqueeze(-1)
