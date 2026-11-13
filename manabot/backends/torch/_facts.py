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
that definition (`RecordEmbeddings`); ordinal ids (an ability index, a
continuation frame) are numbers, not vocabulary.

The output layer starts at zero, so a fresh model's objects are unchanged until
the facts earn weight; gradients reach every fact row that names a valid object.

Captured programs are encoded in row order and preserve signed numeric tokens.

`admit_facts` checks rows once at observation admission. Host preparation
compacts and indexes them; the encoder reads `FactRoutes` and retains Torch
packing of program sequences to preserve its equal-length sorting order.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from manabot.backends.torch._numerics import embedding_rows
from manabot.backends.torch._records import program_number
from manabot.backends.torch._semantic_cards import (
    RecordEmbeddings,
    SemanticCardEncoder,
    admission_catalog,
)
from manabot.backends.torch.arrays import to_numpy
from manabot.env.observation import (
    FACT_KINDS,
)
from manabot.policy import admission
from manabot.policy.prepare import FactRoutes

RecordVocabulary = Literal["hashed", "catalog"]


class FactEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        max_facts: int,
        buckets: int = 4096,
        *,
        vocabulary: RecordVocabulary = "hashed",
    ) -> None:
        super().__init__()
        self.buckets = buckets
        self.max_facts = max_facts
        self.vocabulary = vocabulary
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        if vocabulary == "hashed":
            self.ident = nn.Embedding(buckets, width)
            self.detail = nn.Embedding(buckets, width)
            self.source = nn.Embedding(buckets, width)
        # Amounts (2), duration one-hot (2) and timestamp rank; catalog ids
        # add the ordinal id.
        self.numeric = nn.Linear(5 if vocabulary == "hashed" else 6, width)
        self.program_numeric = nn.Linear(1, width)
        self.program_sequence = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Bucketed shapes look ids up at the fact axis's shape instead of
        # selecting live rows, and keep every slot on CPU inference.
        self.bucketed = False

    def forward(
        self,
        rows: FactRoutes[torch.Tensor],
        objects: torch.Tensor,
        catalog: RecordEmbeddings | None = None,
    ) -> torch.Tensor:
        """Embed prepared facts and add them to their objects in row order."""
        if self.vocabulary == "hashed":
            ids = (
                self.ident(rows.id_rows)
                + self.detail(rows.detail_rows)
                + self.source(rows.source_rows)
            )
        elif catalog is None:
            raise ValueError("catalog fact ids require the card encoder")
        else:

            def masked(index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return torch.where(
                    mask.bool().unsqueeze(-1),
                    embedding_rows(catalog.cards.token_embedding, index),
                    0,
                )

            ids = (
                masked(rows.id_rows, rows.id_mask)
                + masked(rows.detail_rows, rows.detail_mask)
                + masked(rows.token_rows, rows.token_mask)
                + catalog.definitions[rows.source_rows]
            )
        hidden = self.kind(rows.kinds) + ids + self.numeric(rows.columns)
        program = rows.programs
        if program is not None:
            tokens = hidden.flatten(0, 1)[program.rows] + self.program_numeric(
                program_number(program.numbers).unsqueeze(-1)
            )
            # Keep Torch's historical sort for equal-length sequences; the
            # host owns boundaries, while pack_sequence preserves GRU arithmetic.
            packed, _ = self.program_sequence(
                pack_sequence(
                    list(tokens.split(program.schedule.lengths)), enforce_sorted=False
                )
            )
            sequence = torch.cat(unpack_sequence(packed))
            hidden = hidden.masked_scatter(
                program.mask.view_as(rows.kinds).unsqueeze(-1),
                (tokens + sequence).to(hidden.dtype),
            )
        encoded = self.output(torch.tanh(hidden))
        targets = (
            rows.targets.view(rows.kinds.shape)
            - torch.arange(objects.shape[0], device=objects.device)[:, None]
            * objects.shape[1]
        )
        delta = torch.zeros_like(objects).scatter_add(
            1,
            targets.unsqueeze(-1).expand(-1, -1, objects.shape[-1]),
            encoded.masked_fill((rows.linked == 0).unsqueeze(-1), 0).to(objects.dtype),
        )
        return objects + delta


def admit_facts(
    facts: torch.Tensor,
    max_facts: int,
    object_count: int,
    cards: SemanticCardEncoder | None,
) -> None:
    """Validate host fact rows through the shared policy admission contract."""
    admission.facts(to_numpy(facts), max_facts, object_count, admission_catalog(cards))
