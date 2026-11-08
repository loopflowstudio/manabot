"""Categorical public names, separate from in-game objects and rules programs.

The vocabulary is immutable and bound by the checkpoint naming-vocabulary digest. Each
name has its own trainable vector even when two definitions have identical rules.
Only legal choices and viewer-visible definitions contribute name gradients;
padding is zero. Definition IDs route
joins and are never treated as numerical magnitudes.

Input v9 also offers naming *groups*: ordinary native choices that narrow the
pending prompt to a contiguous span of sorted names. A group's vector is the
mean of its members' name vectors, computed exactly from a prefix sum over the
name table, so every member receives gradient and distinct spans stay distinct
without encoding member programs on each step.
"""

from collections.abc import Sequence

import torch
from torch import nn

from manabot.model.semantic_cards import admit_definition_ids
import managym


class NameCandidateEncoder(nn.Module):
    def __init__(
        self,
        candidates: Sequence[managym.NameCandidate],
        hidden_dim: int,
        definition_rows: torch.Tensor | None = None,
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
        # Optional definition joins are already bound by checkpoint admission.
        # Several validated names may refer to one definition; retain all edges.
        self.register_buffer(
            "program_rows",
            definition_rows[self.definitions] if definition_rows is not None else None,
        )

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        """Embed `[..., 2]` (v8) or `[..., 4]` (v9) one-based candidate rows."""
        rows = admit_definition_ids(candidates[..., 0], len(self.definitions))
        if not torch.equal(
            candidates[..., 1], self.definitions[rows].to(candidates.dtype)
        ):
            raise ValueError("name candidate definition disagrees with vocabulary")
        embedded = self.embedding(rows)
        if candidates.shape[-1] == 4:
            embedded = embedded + self._groups(candidates[..., 2], candidates[..., 3])
        return embedded

    def _groups(self, first: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """Mean member vector of each one-based inclusive span; zero for none."""
        size = len(self.definitions)
        first = admit_definition_ids(first, size)
        last = admit_definition_ids(last, size)
        present = first > 0
        if not torch.equal(present, last > 0) or bool((first > last).any()):
            raise ValueError("invalid naming group span")
        # prefix[k] sums name rows 1..k; padding row 0 never enters the sum,
        # so it receives no gradient.
        names = self.embedding.weight[1:]
        prefix = torch.cat([names.new_zeros((1, names.shape[1])), names.cumsum(0)])
        totals = prefix[last] - prefix[(first - 1).clamp(min=0)]
        counts = (last - first + 1).clamp(min=1).unsqueeze(-1)
        return torch.where(present.unsqueeze(-1), totals / counts, 0.0)

    def public_definitions(self, programs: torch.Tensor) -> torch.Tensor:
        """Join names to public programs before visible-object/knowledge pooling.

        Equal-rule definitions must remain distinguishable when their printed
        names are public. Definition-free names contribute no object identity.
        Hidden membership is never inferred: consumers still index only the
        native viewer-safe card, known-hand, decklist and continuation references.
        The raw program table remains the candidate scorer's semantic input.
        """
        assert self.program_rows is not None
        named = self.definitions != 0
        return programs.index_add(
            0, self.program_rows[named], self.embedding.weight[named].to(programs.dtype)
        )
