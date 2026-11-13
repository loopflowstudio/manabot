"""Stateless current-decision context from current stack/continuation relations.

Rows [B,64,49] retain frame order, parent links and target clauses. Definition
IDs join the existing catalog; object indexes gather the current object table.
Neither identifier is used as a numeric feature. Gradients reach the shared
object/catalog encoders and this projection; no memory survives a decision.
"""

import torch
from torch import nn

from manabot.backends.torch.arrays import to_numpy
from manabot.policy import admission
from manabot.policy.prepare import Prepared

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
        batch: Prepared[torch.Tensor],
        objects: torch.Tensor,
        definitions: torch.Tensor,
    ) -> torch.Tensor:
        """Project prepared relations; padding contributes neither context nor gradients."""
        features = [batch.relation_features]
        for indexes, mask in zip(
            batch.relation_objects, batch.relation_object_masks, strict=True
        ):
            linked = objects.flatten(0, 1)[indexes].view(
                *batch.relation_valid.shape, objects.shape[-1]
            )
            features.append(linked.masked_fill((mask == 0).unsqueeze(-1), 0))
        features.extend((definitions[:, :, 0], definitions[:, :, 1]))
        encoded = self.projection(torch.cat(features, -1))
        return encoded.masked_fill((batch.relation_valid == 0).unsqueeze(-1), 0)

    @staticmethod
    def frame_context(
        batch: Prepared[torch.Tensor], encoded: torch.Tensor
    ) -> torch.Tensor:
        """Pool each frame's relations in input order, retaining multiplicity."""
        flat = encoded.flatten(0, 1)
        indexes = batch.frame_indexes.unsqueeze(-1).expand_as(flat)
        pooled = torch.zeros_like(flat).scatter_add(0, indexes, flat).view_as(encoded)
        valid = batch.relation_valid
        counts = (
            torch.zeros_like(valid)
            .flatten()
            .scatter_add(0, batch.frame_indexes, valid.flatten())
            .view_as(valid)
        )
        return pooled / counts.clamp_min(1).unsqueeze(-1)
