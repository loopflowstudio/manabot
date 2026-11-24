"""
Policy-input fact rows, joined to the player or permanent each describes.

A fact row states one open-vocabulary fact (a counter kind, an added subtype, a
set name or creature type, a spent ability, a continuous-effect record) about
one object; `manabot.env.observation` documents the columns. Each row is
embedded from its kind, its vocabulary id and effect detail (learning-token
values rebuilt from their 16-bit halves), its amounts, duration, timestamp rank
and source, then summed into its object's pre-attention embedding.

Two id vocabularies exist. `hashed` hashes ids and sources into `buckets`
learned rows each. `catalog` reads ids
through the card encoder's token embedding and sources through its row for
that definition; ordinal ids (an ability index, a
continuation frame) are numbers, not vocabulary.

The output layer starts at zero, so a fresh model's objects are unchanged until
the facts earn weight; gradients reach every fact row that names a valid object.

Captured programs are encoded in row order and preserve signed numeric tokens.

Host admission checks rows once. Host preparation
compacts and indexes them; the encoder reads `FactRoutes`, including the
host's packed program schedule in the original equal-length sorting order.
Padded `[B,F]` routes add into each observation's objects with `scatter_add`;
live-only `[1,R]` routes (MPS) `index_add` into the flattened objects.
"""

from typing import Literal

import torch
import torch.nn as nn

from manabot.backends.torch.numerics import embedding_rows
from manabot.backends.torch.packed_gru import packed_gru
from manabot.backends.torch.records import program_number
from manabot.backends.torch.semantic_cards import SemanticCardEncoder
from manabot.env.observation import FACT_KINDS
from manabot.policy.prepare import FactRoutes

RecordVocabulary = Literal["hashed", "catalog"]


class FactEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        numeric: int,
        buckets: int = 4096,
        vocabulary: RecordVocabulary = "hashed",
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        if vocabulary == "hashed":
            self.ident = nn.Embedding(buckets, width)
            self.detail = nn.Embedding(buckets, width)
            self.source = nn.Embedding(buckets, width)
        # Amounts (2), duration flags and timestamp rank; catalog ids add the
        # ordinal id. `numeric` is `Layout.fact_width`.
        self.numeric = nn.Linear(numeric, width)
        self.program_numeric = nn.Linear(1, width)
        self.program_sequence = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rows: FactRoutes[torch.Tensor],
        objects: torch.Tensor,
        token_embedding: nn.Embedding | None = None,
        definitions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed prepared facts and add them to their objects in row order."""
        if self.vocabulary == "hashed":
            ids = (
                self.ident(rows.id_rows)
                + self.detail(rows.detail_rows)
                + self.source(rows.source_rows)
            )
        elif token_embedding is None or definitions is None:
            raise ValueError("catalog fact ids require the card encoder")
        else:

            def masked(index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return torch.where(
                    mask.bool().unsqueeze(-1),
                    embedding_rows(token_embedding, index),
                    0,
                )

            ids = (
                masked(rows.id_rows, rows.id_mask)
                + masked(rows.detail_rows, rows.detail_mask)
                + masked(rows.token_rows, rows.token_mask)
                + SemanticCardEncoder.lookup(definitions, rows.source_rows)
            )
        hidden = self.kind(rows.kinds) + ids + self.numeric(rows.columns)
        program = rows.programs
        if program is not None:
            tokens = hidden.flatten(0, 1)[program.rows] + self.program_numeric(
                program_number(program.numbers).unsqueeze(-1)
            )
            packed, _ = packed_gru(
                self.program_sequence,
                tokens[program.schedule.gather],
                program.schedule.sizes,
            )
            sequence = packed[program.schedule.unpack]
            hidden = hidden.masked_scatter(
                program.mask.view_as(rows.kinds).unsqueeze(-1),
                (tokens + sequence).to(hidden.dtype),
            )
        encoded = self.output(torch.tanh(hidden)).masked_fill(
            (rows.linked == 0).unsqueeze(-1), 0
        )
        if rows.compact:
            # Live rows `[1,R]` add into the flattened `[B*N,W]` objects.
            return objects + torch.zeros_like(objects).flatten(0, 1).index_add(
                0, rows.targets, encoded[0].to(objects.dtype)
            ).view_as(objects)
        targets = (
            rows.targets.view(rows.kinds.shape)
            - torch.arange(objects.shape[0], device=objects.device)[:, None]
            * objects.shape[1]
        )
        delta = torch.zeros_like(objects).scatter_add(
            1,
            targets.unsqueeze(-1).expand(-1, -1, objects.shape[-1]),
            encoded.to(objects.dtype),
        )
        return objects + delta
