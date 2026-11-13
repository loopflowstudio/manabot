"""Encode shared payloads once, then route their gradients into affected objects.

Membership is an explicit join, never learned from object IDs. We retain the
existing unnormalized fact-sum convention: repeated memberships contribute
repeatedly. A mean would erase multiplicity. This learned sum is compression,
not an injective semantic representation; tensor transport retains the records.
Details carry ordered program tokens, reduced by a GRU per record.

`hashed` ids hash modifications, details and sources into 4096 learned rows
each; `catalog` ids read the card encoder's token rows and its row for the
source definition, as fact rows do.

Host admission checks the buffers once. The
encoder reads `EffectRoutes`; host preparation owns all CSR and recurrent routes.
"""

import torch
from torch import nn

from manabot.backends.torch._semantic_cards import SemanticCardEncoder
from manabot.backends.torch._facts import RecordVocabulary
from manabot.backends.torch._numerics import embedding_rows
from manabot.backends.torch._records import RecordReducer, program_number
from manabot.policy.prepare import EffectRoutes


class SharedEffectEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        vocabulary: RecordVocabulary = "hashed",
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        if vocabulary == "hashed":
            self.modification = nn.Embedding(4096, width)
            self.detail = nn.Embedding(4096, width)
        self.detail_kind = nn.Embedding(20, width)
        self.detail_number = nn.Linear(1, width)
        if vocabulary == "hashed":
            self.source = nn.Embedding(4096, width)
        self.header = nn.Linear(8 + 2 * width, width)
        self.reducer = RecordReducer(width)

    def forward(
        self,
        rows: EffectRoutes[torch.Tensor] | None,
        objects: torch.Tensor,
        token_embedding: nn.Embedding | None = None,
        definitions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode each prepared record once; duplicate memberships sum separately."""
        if rows is None:
            return objects
        table = objects.flatten(0, 1)
        controller = table[rows.controller].masked_fill(
            (rows.controller_mask == 0)[:, None], 0
        )
        source_object = table[rows.source_object].masked_fill(
            (rows.source_object_mask == 0)[:, None], 0
        )
        header = self.header(torch.cat((rows.numeric, controller, source_object), -1))
        if self.vocabulary == "hashed":
            mod = self.modification(rows.modification_rows)
            sources = self.source(rows.source_rows)
            detail = embedding_rows(self.detail, rows.detail_rows)
        elif token_embedding is None or definitions is None:
            raise ValueError("catalog effect ids require the card encoder")
        else:
            mod = embedding_rows(token_embedding, rows.modification_rows)
            sources = SemanticCardEncoder.lookup(definitions, rows.source_rows)
            detail = embedding_rows(token_embedding, rows.detail_rows)
        detail_tokens = detail + embedding_rows(self.detail_kind, rows.detail_kinds)
        detail_tokens = detail_tokens + self.detail_number(
            program_number(rows.detail_numbers.to(objects.dtype)).unsqueeze(-1)
        ).masked_fill((rows.detail_is_number == 0).unsqueeze(-1), 0)
        tokens = torch.cat(
            (
                torch.stack((mod, sources, torch.tanh(header)), 1).flatten(0, 1),
                detail_tokens,
            )
        )
        encoded = self.reducer(tokens[rows.token_indexes], rows.schedule)
        repeated = encoded[rows.member_records]
        delta = torch.zeros_like(table).index_add(
            0, rows.routes, repeated.to(table.dtype)
        )
        return objects + delta.reshape_as(objects)
