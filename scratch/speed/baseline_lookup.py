"""Exact kept lookup from 734f2cd5c; compiled alongside the candidate."""

import torch


def embed_definitions(
    self, ids: torch.Tensor, definitions: torch.Tensor
) -> torch.Tensor:
    """Join admitted card or permanent IDs to their card programs."""
    return definitions[self._rows(ids)]
