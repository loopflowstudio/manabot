"""Encode shared payloads once, then route their gradients into affected objects.

Membership is an explicit join, never learned from object IDs. We retain the
existing unnormalized fact-sum convention: repeated memberships contribute
repeatedly. A mean would erase multiplicity. This learned sum is compression,
not an injective semantic representation; tensor transport retains the records.
V8 adds ordered program tokens; v7 retains its original reducer and embeddings.

`hashed` ids (every frozen model) hash modifications, details and sources into
4096 learned rows each; `catalog` ids (v8 onward) read the card encoder's
token rows and its row for the source definition, as fact rows do.
"""

from dataclasses import dataclass

import torch
from torch import nn

from manabot.env.shared_effects import (
    DETAIL_DIM,
    EFFECT_DIM,
    EFFECT_ROWS,
    MEMBER_DIM,
    MEMBER_ROWS,
    shared_effect_shapes,
)
from manabot.model.facts import RecordVocabulary
from manabot.model.records import (
    RecordReducer,
    RecordReduction,
    join_rows,
    masked_record_rows,
    program_number,
    record_indexes,
)
from manabot.model.semantic_cards import RecordEmbeddings, RecordNamespace


@dataclass
class _CatalogRecord:
    """One admitted program, before batch-wide vocabulary lookup."""

    header: torch.Tensor
    modification: torch.Tensor
    source: torch.Tensor
    kinds: torch.Tensor
    words: torch.Tensor


class SharedEffectEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        reduction: RecordReduction = "typed_pool",
        *,
        version: int = 7,
        vocabulary: RecordVocabulary = "hashed",
    ) -> None:
        super().__init__()
        if vocabulary == "catalog" and version < 8:
            raise ValueError("catalog effect ids require policy input v8")
        self.vocabulary = vocabulary
        if vocabulary == "hashed":
            self.modification = nn.Embedding(4096, width)
            self.detail = nn.Embedding(4096, width)
        self.version = version
        self.detail_rows = shared_effect_shapes(version)["effect_details"][0]
        self.detail_kind = nn.Embedding(20 if version >= 8 else 4, width)
        if version >= 8:
            self.detail_number = nn.Linear(1, width)
        if vocabulary == "hashed":
            self.source = nn.Embedding(4096, width)
        self.header = nn.Linear(8 + 2 * width, width)
        self.reducer = RecordReducer(width, "typed_gru" if version >= 8 else reduction)

    def _details(
        self,
        kinds: torch.Tensor,
        words: torch.Tensor,
        embedded: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Retain signed integer/boolean values alongside ordered token identities."""
        encoded = embedded + self.detail_kind(kinds)
        if self.version >= 8:
            signed = torch.where(words >= 2**31, words - 2**32, words)
            numeric = ((kinds == 14) | (kinds == 15)).unsqueeze(-1)
            encoded = encoded + self.detail_number(
                program_number(signed.to(dtype)).unsqueeze(-1)
            ).masked_fill(~numeric, 0)
        return encoded

    def _catalog_records(
        self,
        records: list[_CatalogRecord],
        catalog: RecordEmbeddings,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Resolve ids together, retaining each program's order and gradients.

        Vocabulary admission synchronizes CUDA with the host. Batching these
        lookups pays that cost once per family instead of once per effect.
        Invalid ids still fail through the same catalog admission functions.
        """
        modifications = catalog.names(
            RecordNamespace.TOKEN, torch.stack([r.modification for r in records])
        )
        sources = catalog.sources(torch.stack([r.source for r in records]))
        kinds = torch.cat([r.kinds for r in records])
        words = torch.cat([r.words for r in records])
        program = kinds > 10
        details = modifications.new_zeros((len(kinds), modifications.shape[-1]))
        details[~program] = catalog.names(RecordNamespace.TOKEN, words[~program])
        details[program] = catalog.tokens(kinds[program] - 10, words[program])
        encoded = self._details(kinds, words, details, dtype)
        parts = encoded.split([len(r.kinds) for r in records])
        return [
            torch.cat(
                (
                    torch.stack((modifications[i], sources[i], torch.tanh(r.header))),
                    part,
                )
            )
            for i, (r, part) in enumerate(zip(records, parts, strict=True))
        ]

    def _route(
        self,
        records: list[torch.Tensor],
        routes: list[torch.Tensor],
        objects: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce independent programs and sum each membership into [N,D] objects."""
        encoded = self.reducer(records)
        if self.version >= 8:
            repeated = torch.repeat_interleave(
                encoded,
                torch.tensor([len(route) for route in routes], device=objects.device),
                dim=0,
            )
        else:
            repeated = torch.cat(
                [encoded[i].expand(len(route), -1) for i, route in enumerate(routes)]
            )
        # Duplicate memberships accumulate; every target's gradient reaches the
        # same payload. Retained v7 keeps its original expansion and reduction.
        return objects + torch.zeros_like(objects).index_add(
            0, torch.cat(routes), repeated.to(objects.dtype)
        )

    def forward(
        self,
        headers: torch.Tensor,
        details: torch.Tensor,
        members: torch.Tensor,
        objects: torch.Tensor,
        catalog: RecordEmbeddings | None = None,
    ) -> torch.Tensor:
        """Catalog ids need `catalog`; hashed ids ignore it."""
        if self.vocabulary == "hashed":
            catalog = None
        elif catalog is None:
            raise ValueError("catalog effect ids require the card encoder")
        headers = masked_record_rows(headers, EFFECT_ROWS, EFFECT_DIM, 17)
        batch = len(objects)
        if (
            headers.shape[0] != batch
            or details.shape != (batch, self.detail_rows, DETAIL_DIM)
            or members.shape != (batch, MEMBER_ROWS, MEMBER_DIM)
        ):
            raise ValueError("invalid shared effect buffer shape")
        results = list(objects.unbind(0))
        all_records: list[torch.Tensor] = []
        all_routes: list[torch.Tensor] = []
        catalog_records: list[_CatalogRecord] = []
        for b in headers[..., 17].bool().any(-1).nonzero().flatten().tolist():
            live = headers[b, headers[b, :, 17].bool()]
            halves = record_indexes(live[:, :2], 65535, "effect vocabulary half")
            source = record_indexes(live[:, 11], 2**24, "effect source definition")
            links = record_indexes(
                live[:, [10, 12]], objects.shape[1], "effect source/controller"
            )
            if (links[:, 0] == 0).any():
                raise ValueError("effect controller is absent")
            numeric = live[:, 2:10].clone()
            numeric[:, -1] /= EFFECT_ROWS
            header = self.header(
                torch.cat(
                    (
                        numeric,
                        join_rows(objects[b], links[:, 0]),
                        join_rows(objects[b], links[:, 1]),
                    ),
                    -1,
                )
            )
            modification = (halves[:, 0] << 16) | halves[:, 1]
            if catalog is None:
                mod = self.modification(modification % 4096)
                sources = self.source(source % 4096)
            offsets = record_indexes(
                live[:, 13:17],
                max(MEMBER_ROWS, self.detail_rows),
                "effect CSR offset/count",
            ).tolist()
            records: list[torch.Tensor] = []
            routes: list[torch.Tensor] = []
            d_end = m_end = 0
            member_orders: list[torch.Tensor] = []
            for i, (d, count, m, targets) in enumerate(offsets):
                if (
                    d != d_end
                    or m != m_end
                    or d + count > self.detail_rows
                    or targets == 0
                    or m + targets > MEMBER_ROWS
                ):
                    raise ValueError("invalid effect CSR partition")
                d_end, m_end = d + count, m + targets
                tokens = details[b, d:d_end]
                kinds = record_indexes(
                    tokens[:, 0], 19 if self.version >= 8 else 3, "effect detail kind"
                )
                ids = record_indexes(tokens[:, 1:], 65535, "effect detail half")
                if ((kinds == 0) | ((kinds > 3) & (kinds < 11))).any():
                    raise ValueError("live effect detail has no kind")
                unsigned = (ids[:, 0] << 16) | ids[:, 1]
                if catalog is None:
                    detail = self.detail(unsigned % 4096)
                    records.append(
                        torch.cat(
                            (
                                torch.stack(
                                    (mod[i], sources[i], torch.tanh(header[i]))
                                ),
                                self._details(kinds, unsigned, detail, objects.dtype),
                            )
                        )
                    )
                else:
                    catalog_records.append(
                        _CatalogRecord(
                            header[i], modification[i], source[i], kinds, unsigned
                        )
                    )
                memberships = members[b, m:m_end]
                targets_index = record_indexes(
                    memberships[:, 0], objects.shape[1], "effect member"
                )
                if (targets_index == 0).any():
                    raise ValueError("effect membership is absent")
                orders = record_indexes(
                    memberships[:, 1], MEMBER_ROWS, "effect membership order"
                )
                if (orders == 0).any():
                    raise ValueError("effect membership order is absent")
                member_orders.append(orders)
                routes.append(targets_index - 1)
            orders = torch.cat(member_orders)
            if len(orders.unique()) != len(orders):
                raise ValueError("duplicate effect membership order")
            if self.version >= 8:
                # Programs remain independent packed sequences. One reduction
                # across the batch avoids a separate GRU dispatch per sample.
                all_records.extend(records)
                all_routes.extend(route + b * objects.shape[1] for route in routes)
                continue
            results[b] = self._route(records, routes, objects[b])
        if catalog_records:
            assert catalog is not None
            all_records = self._catalog_records(catalog_records, catalog, objects.dtype)
        if self.version >= 8 and all_records:
            return self._route(
                all_records, all_routes, objects.flatten(0, 1)
            ).reshape_as(objects)
        return torch.stack(results)
