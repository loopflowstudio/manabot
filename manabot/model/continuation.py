"""Stateless current-decision context from current stack/continuation relations.

Rows [B,64,49] retain frame order, parent links and target clauses. Definition
IDs join the existing catalog; object indexes gather the current object table.
Neither identifier is used as a numeric feature. Gradients reach the shared
object/catalog encoders and this projection; no memory survives a decision.
"""

import torch
from torch import nn

from manabot.backends.torch.arrays import to_numpy
from manabot.env.observation import RELATION_VALID_INDEX
from manabot.policy import admission

# One-based object joins of a relation row (0: none).
_OBJECT_COLUMNS = (6, 24)


def admit_relations(rows: torch.Tensor, relation_dim: int, object_count: int) -> None:
    """Reject malformed relation rows [B,64,relation_dim] before any forward."""
    admission.relations(to_numpy(rows), relation_dim, object_count)


class ContinuationEncoder(nn.Module):
    def __init__(self, width: int, relation_dim: int) -> None:
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
        """Return [B,R,D], zero for padding; shared by pooling and frame joins.

        Rows must be admitted (`admit_relations`); bucketed callers may trim
        them to fewer than the capacity.
        """
        valid = rows[..., RELATION_VALID_INDEX]
        rows = torch.where(valid.bool().unsqueeze(-1), rows, 0)
        # Waiting occurrence order (pending and delayed triggers) is
        # presentation, unlike public stack order. Retain its causal/ability
        # facts without treating frame joins as features.
        features_rows = rows.clone()
        pending = (rows[..., 3] == 7) | (rows[..., 3] == 8) | (rows[..., 3] == 10)
        features_rows[..., 1:3] = torch.where(pending.unsqueeze(-1), 0, rows[..., 1:3])
        features = [
            features_rows[..., :5],
            features_rows[..., 7:23],
            features_rows[..., 25:RELATION_VALID_INDEX],
            # Target eligibility and the spell terms follow the validity column.
            features_rows[..., RELATION_VALID_INDEX + 1 :],
        ]
        for column in _OBJECT_COLUMNS:
            indexes = rows[..., column]
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
        return encoded.sum(1) / rows[..., RELATION_VALID_INDEX].sum(
            1, keepdim=True
        ).clamp_min(1)

    @staticmethod
    def frame_context(rows: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Pool every relation of a frame, preserving source, target and causal context.

        Frame indexes are joins only. Gradients flow through all related rows.
        """
        ids = rows[..., 1]
        valid = rows[..., RELATION_VALID_INDEX]
        indexes = (torch.where(valid.bool(), ids, 1).long() - 1).clamp_min(0)
        pooled = torch.zeros_like(encoded).scatter_add(
            1, indexes.unsqueeze(-1).expand_as(encoded), encoded
        )
        counts = torch.zeros_like(valid).scatter_add(1, indexes, valid)
        return pooled / counts.clamp_min(1).unsqueeze(-1)
