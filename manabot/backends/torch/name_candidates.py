"""Categorical public names, separate from in-game objects and rules programs.

The vocabulary is immutable and bound by the checkpoint naming-vocabulary digest. Each
name has its own trainable vector even when two definitions have identical rules.
Only legal choices contribute name gradients (and viewer-visible definitions,
for a checkpoint that adds names onto definitions); padding is zero. Definition IDs route
joins and are never treated as numerical magnitudes.

Input v9 also offers naming *groups*: ordinary native choices that narrow the
pending prompt to a contiguous span of sorted names. A group's vector is the
mean of its members' name vectors, computed exactly from a prefix sum over the
name table, so every member receives gradient and distinct spans stay distinct
without encoding member programs on each step.
"""

import numpy as np
import torch
from torch import nn

from manabot.policy.prepare import Prepared


class NameCandidateEncoder(nn.Module):
    def __init__(
        self,
        definitions: np.ndarray,
        hidden_dim: int,
        program_rows: np.ndarray | None,
        *,
        onto_definitions: bool = False,
    ) -> None:
        super().__init__()
        self.onto_definitions = onto_definitions
        self.register_buffer("definitions", torch.from_numpy(definitions))
        self.embedding = nn.Embedding(len(definitions), hidden_dim, padding_idx=0)
        # Bucketed shapes join names without a data-dependent selection.
        self.bucketed = False
        # Optional definition joins are already bound by checkpoint admission.
        # Several validated names may refer to one definition; retain all edges.
        self.register_buffer(
            "program_rows",
            torch.from_numpy(program_rows) if program_rows is not None else None,
        )

        # Static catalog routes are prepared once on the host; neither the
        # route nor the derived mask is checkpoint state or a learned identity.
        named = self.definitions != 0
        self.register_buffer("_public_mask", named, persistent=False)
        self.register_buffer(
            "_public_names", named.nonzero().flatten(), persistent=False
        )

    def forward(self, batch: Prepared[torch.Tensor]) -> torch.Tensor:
        """Embed legal names and the mean member vector of each naming group."""
        embedded = self.embedding(batch.name_rows)
        names = self.embedding.weight[1:]
        prefix = torch.cat([names.new_zeros((1, names.shape[1])), names.cumsum(0)])
        totals = prefix[batch.group_last] - prefix[batch.group_first]
        groups = torch.where(
            batch.group_mask.bool().unsqueeze(-1),
            totals / batch.group_counts.unsqueeze(-1),
            0.0,
        )
        return embedded + groups

    def public_definitions(self, programs: torch.Tensor) -> torch.Tensor:
        """The definition vectors objects read: their programs, plus each
        public name's vector when `onto_definitions` (`ModelConfig.definition_names`).

        Adding names keeps equal-rule definitions distinguishable, but a
        unit-variance name vector outweighs the rest of a card's token at
        initialization and gives training a second source of card identity, so
        fresh models leave programs alone and use names on NameCard candidates
        only. Definition-free names contribute no object identity. Hidden
        membership is never inferred: consumers still index only the native
        viewer-safe card, known-hand, decklist and continuation references.
        The raw program table remains the candidate scorer's semantic input.
        """
        if not self.onto_definitions:
            return programs
        assert self.program_rows is not None
        named = self._public_mask
        if self.bucketed:
            # Definition-free names add zero to padding row 0.
            return programs.index_add(
                0,
                self.program_rows,
                torch.where(named[:, None], self.embedding.weight, 0).to(
                    programs.dtype
                ),
            )
        return programs.index_add(
            0,
            self.program_rows[self._public_names],
            self.embedding.weight[self._public_names].to(programs.dtype),
        )
