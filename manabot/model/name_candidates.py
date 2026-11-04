"""Categorical public names, separate from in-game objects and rules programs.

The vocabulary is immutable and bound by the checkpoint naming-vocabulary digest. Each
name has its own trainable vector even when two definitions have identical rules.
Only referenced legal rows receive gradients; padding is zero. Definition IDs route
joins and are never treated as numerical magnitudes.
"""

from collections.abc import Sequence

import torch
from torch import nn

from manabot.model.semantic_cards import admit_definition_ids
import managym


class NameCandidateEncoder(nn.Module):
    def __init__(
        self, candidates: Sequence[managym.NameCandidate], hidden_dim: int
    ) -> None:
        super().__init__()
        if not candidates or len({candidate.name for candidate in candidates}) != len(
            candidates
        ):
            raise ValueError("naming vocabulary must be nonempty and unique")
        # Row identity is the name, not an executable-definition surrogate.
        # Definition-free validated names therefore never enter the card catalog.
        self.register_buffer(
            "definitions",
            torch.tensor(
                [0]
                + [
                    candidate.definition + 1 if candidate.definition is not None else 0
                    for candidate in candidates
                ],
                dtype=torch.long,
            ),
        )
        self.embedding = nn.Embedding(len(candidates) + 1, hidden_dim, padding_idx=0)

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        rows = admit_definition_ids(candidates[..., 0], len(self.definitions))
        if not torch.equal(
            candidates[..., 1], self.definitions[rows].to(candidates.dtype)
        ):
            raise ValueError("name candidate definition disagrees with vocabulary")
        return self.embedding(rows)
