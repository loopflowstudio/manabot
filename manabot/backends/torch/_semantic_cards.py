"""Complete typed card programs joined to viewer-safe card and knowledge rows.

`SemanticCardEncoder` owns the one token vocabulary of card text. With
`records`, it also owns the exact vocabulary of fact and shared-effect ids
(`RecordNamespace` keys): every name native `record_vocabulary` lists reads
the row of the card token for the same concept, or a row appended after the
card tokens when no card text contains it. Nothing is hashed; an id outside
the vocabulary fails loudly.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from manabot.policy.catalog import catalog_tables
from manabot.policy.prepare import CatalogPrograms, Prepared


@dataclass(frozen=True)
class CatalogShape:
    """Separate catalog dimensions; lengths include expanded definition references.

    Counts describe representation cost, not encountered-card diversity or an
    experimentally established difficulty score. Only real tokens are embedded
    and recurrently encoded; padded_tokens describes the stored token table.
    """

    definitions: int
    unique_tokens: int
    expanded_tokens: int
    maximum_program_tokens: int
    padded_tokens: int


class SemanticCardEncoder(nn.Module):
    def __init__(
        self,
        pack_key: str,
        hidden_dim: int,
        decklists: bool = False,
        *,
        records: bool = False,
        decision_memory: bool = False,
        library: bool = False,
        stream_norm: bool = False,
    ) -> None:
        """`decklists` adds a projection of the pooled
        decklist rows into each player's knowledge vector; `records` adds the
        fact and shared-effect id vocabulary (see the module docstring).
        `library` projects positional knowledge separately from hand minima.
        `stream_norm` pools known cards without the `/ 60` attenuation and
        normalizes each player's knowledge vector (`forward`)."""
        super().__init__()
        catalog = catalog_tables(
            pack_key, records=records, decision_memory=decision_memory
        )
        self.token_vocabulary = catalog.vocabulary
        self.token_embedding = nn.Embedding(catalog.embedding_rows, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        for name, array in catalog.buffers.items():
            self.register_buffer(name, torch.from_numpy(array))
        # One count-weighted definition pool per decklist count column (main,
        # unseen, sideboard, remaining sideboard), projected to one vector.
        self.decklist_projection = (
            nn.Linear(4 * hidden_dim, hidden_dim) if decklists else None
        )
        # A library top and a known hand card have different strategic roles.
        # Keep their pools distinct until this learned projection; no bias means
        # an empty positional pool contributes exactly zero.
        self.library_projection = (
            nn.Linear(hidden_dim, hidden_dim, bias=False) if library else None
        )
        self.records = records
        self.knowledge_norm = nn.LayerNorm(hidden_dim) if stream_norm else None

    @property
    def catalog_shape(self) -> CatalogShape:
        """Dimensions of the programs currently bound to this encoder."""
        count, width = self.tokens.shape
        return CatalogShape(
            count,
            len(self.token_vocabulary),
            int(self.lengths.sum()),
            width,
            self.tokens.numel(),
        )

    def definitions(self, programs: CatalogPrograms[torch.Tensor]) -> torch.Tensor:
        """Encode catalog programs into a [definitions + 1, hidden] table; row 0
        is the zero padding row and every consumer indexes by catalog row + 1.

        The host schedule selects the programs that run through the GRU; every
        other definition row stays zero.
        Each program is encoded independently, so selected rows and parameter
        gradients agree with full encoding up to floating-point roundoff;
        unreferenced rows receive none. Training uses this because the catalog
        grows with the world while a batch references a small subset. Inference
        caches the full table once per policy scope instead.
        """
        count = len(self.tokens)
        rows = programs.rows
        table = self.token_embedding.weight.new_zeros(
            (count + 1, self.token_embedding.embedding_dim)
        )
        if len(rows) == 0:
            return table
        # CUDA's packed GRU can return Half even inside BF16 autocast. The
        # definition table follows the embedding weights (also for empty rows),
        # so normalize at this join without detaching the program gradients.
        encoded = self._encode_programs(programs).to(table.dtype)
        return table.index_copy(0, rows, encoded)

    def _encode_programs(self, programs: CatalogPrograms[torch.Tensor]) -> torch.Tensor:
        """Encode nonempty one-based catalog rows, preserving their supplied order.

        Packing sorts programs by length. Within each chunk, completed programs
        occupy the hidden state's suffix; only the active prefix continues.
        """
        # Packed GRU backward creates a packed-input-sized slice gradient at
        # each time step. Bound that copying to 64 steps per call. This is full
        # backpropagation through time: hidden states keep their gradients at
        # every chunk boundary. The same GRU encodes every real token once.
        batches = torch.tensor(programs.schedule.sizes, dtype=torch.long).split(64)
        sizes = [int(batch.sum()) for batch in batches]
        chunks = self.token_embedding(programs.tokens).split(sizes)
        state: torch.Tensor | None = None
        finished: list[torch.Tensor] = []
        for index, (data, batch) in enumerate(zip(chunks, batches, strict=True)):
            active = int(batch[0])
            _, state = self.encoder(
                nn.utils.rnn.PackedSequence(data, batch),
                None if state is None else state[:, :active],
            )
            remaining = int(batches[index + 1][0]) if index + 1 < len(batches) else 0
            finished.append(state[:, remaining:])
        # Packed sequences end shortest first. Reverse their completed suffixes
        # to restore descending-length order, then undo the original sorting.
        encoded = torch.cat(finished[::-1], dim=1)[0]
        return encoded[programs.schedule.restore]

    @staticmethod
    def lookup(definitions: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Read card-program rows; every read of the program table goes here.

        Rows repeat heavily within a batch. Under CUDA autograd, embedding
        backward groups repeated rows before accumulating into the
        differentiable table, where advanced indexing accumulates each one
        separately (`experiments/reports/ppo_cuda.md`). Row 0 still receives gradients;
        no padding suppression or frequency normalization applies. Other
        devices and inference keep advanced indexing, so CPU results are
        unchanged.
        """
        if (
            definitions.is_cuda
            and torch.is_grad_enabled()
            and definitions.requires_grad
        ):
            return F.embedding(rows, definitions)
        return definitions[rows]

    def forward(
        self, batch: Prepared[torch.Tensor], definitions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Card definition rows [B,2,C,D] and each player's knowledge [B,2,D].

        Knowledge pools known hand cards by count, library positions by
        `1 / position` and decklist rows by their share of the deck. Without
        `knowledge_norm` hand and library pools are divided by 60, the
        historical scale. With it the pool is undivided and one LayerNorm sets
        its scale to that of the other object streams; a player with nothing
        known and no disclosed list receives exactly zero.
        """
        cards = self.lookup(definitions, batch.card_rows)
        attenuation = 1.0 if self.knowledge_norm is not None else 60.0
        known = (
            self.lookup(definitions, batch.known_rows)
            * batch.known_counts[..., None]
            / attenuation
        ).sum(dim=2)
        if self.library_projection is not None:
            library = (
                self.lookup(definitions, batch.library_rows)
                * batch.library_weights[..., None]
                / attenuation
            ).sum(dim=2)
            known = known + self.library_projection(library)
        if self.decklist_projection is not None:
            pooled = torch.einsum(
                "bsrh,bsrc->bsch",
                self.lookup(definitions, batch.deck_rows),
                batch.deck_counts,
            ).flatten(start_dim=2)
            known = known + self.decklist_projection(pooled)
        if self.knowledge_norm is not None:
            present = (batch.known_counts > 0).any(-1)
            if self.library_projection is not None:
                present = present | (batch.library_weights > 0).any(-1)
            if self.decklist_projection is not None:
                present = present | (batch.deck_counts > 0).flatten(2).any(-1)
            known = self.knowledge_norm(known) * present.unsqueeze(-1).to(known.dtype)
        return cards, known
