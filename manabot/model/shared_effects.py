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
            else:
                mod = catalog.names(RecordNamespace.TOKEN, modification)
                sources = catalog.sources(source)
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
                else:
                    # Kinds 1-3 name a keyword, subtype or color; 11-19 carry
                    # a learning token of kind 1-9.
                    program = kinds > 10
                    detail = mod.new_zeros((len(kinds), mod.shape[-1]))
                    detail[~program] = catalog.names(
                        RecordNamespace.TOKEN, unsigned[~program]
                    )
                    detail[program] = catalog.tokens(
                        kinds[program] - 10, unsigned[program]
                    )
                encoded_details = detail + self.detail_kind(kinds)
                if self.version >= 8:
                    # Integer/boolean tokens retain signed numeric meaning;
                    # the GRU retains the ordered shared program structure.
                    signed = torch.where(unsigned >= 2**31, unsigned - 2**32, unsigned)
                    numeric = ((kinds == 14) | (kinds == 15)).unsqueeze(-1)
                    encoded_details = encoded_details + self.detail_number(
                        program_number(signed.to(objects.dtype)).unsqueeze(-1)
                    ).masked_fill(~numeric, 0)
                records.append(
                    torch.cat(
                        (
                            torch.stack((mod[i], sources[i], torch.tanh(header[i]))),
                            encoded_details,
                        )
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
        if self.version >= 8 and all_records:
            return self._route(
                all_records, all_routes, objects.flatten(0, 1)
            ).reshape_as(objects)
        return torch.stack(results)
