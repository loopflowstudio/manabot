"""Encode shared payloads once, then route their gradients into affected objects.

Membership is an explicit join, never learned from object IDs. We retain the
existing unnormalized fact-sum convention: repeated memberships contribute
repeatedly. A mean would erase multiplicity. This learned sum is compression,
not an injective semantic representation; tensor transport retains the records.
Details carry ordered program tokens, reduced by a GRU per record.

`hashed` ids hash modifications, details and sources into 4096 learned rows
each; `catalog` ids read the card encoder's token rows and its row for the
source definition, as fact rows do.

`admit_shared_effects` checks the buffers once at observation admission; the
encoder's host reads only build CSR routes.
"""

import torch
from torch import nn

from manabot.env.shared_effects import (
    DETAIL_DIM,
    DETAIL_ROWS,
    EFFECT_DIM,
    EFFECT_ROWS,
    MEMBER_DIM,
    MEMBER_ROWS,
)
from manabot.model.facts import RecordVocabulary
from manabot.model.numerics import buffer_on, embedding_rows
from manabot.model.records import (
    RecordReducer,
    admit_record_rows,
    join_record_rows,
    masked_record_rows,
    program_number,
    record_indexes,
)
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    RecordNamespace,
    SemanticCardEncoder,
    admit_definition_ids,
)


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
        self.detail_rows = DETAIL_ROWS
        self.detail_kind = nn.Embedding(20, width)
        self.detail_number = nn.Linear(1, width)
        if vocabulary == "hashed":
            self.source = nn.Embedding(4096, width)
        self.header = nn.Linear(8 + 2 * width, width)
        self.reducer = RecordReducer(width)

    def forward(
        self,
        headers: torch.Tensor,
        details: torch.Tensor,
        members: torch.Tensor,
        objects: torch.Tensor,
        catalog: RecordEmbeddings | None = None,
    ) -> torch.Tensor:
        """Route admitted records (`admit_shared_effects`) into `objects` [B,N,D].

        Catalog ids need `catalog`; hashed ids ignore it. The host reads below
        build CSR routes; admission has already checked every partition.
        Bucketed callers may trim headers, details and members to fewer rows
        than their capacities; row strides follow the tensors.
        """
        if self.vocabulary == "hashed":
            catalog = None
        elif catalog is None:
            raise ValueError("catalog effect ids require the card encoder")
        # One metadata transfer routes the entire batch. Numerical features and
        # object joins below still read the original tensors, retaining gradients.
        host = masked_record_rows(headers.detach().cpu(), 17)
        batch, object_count, width = objects.shape
        header_rows, detail_rows, member_rows = (
            headers.shape[1],
            details.shape[1],
            members.shape[1],
        )
        live_positions = host[..., 17].nonzero()
        if len(live_positions) == 0:
            return objects
        live = host[live_positions[:, 0], live_positions[:, 1]].long()
        halves, source, links = live[:, :2], live[:, 11], live[:, [10, 12]]
        offsets = live[:, 13:17].tolist()
        detail_indexes: list[int] = []
        member_indexes: list[int] = []
        member_records: list[int] = []
        token_indexes: list[int] = []
        lengths: list[int] = []
        # Projections produce all three-token headers, then all details. Gather
        # them into header/detail sequences without per-record device operations.
        header_token_count = 3 * len(live)
        for i, ((b, _), (d, count, m, targets)) in enumerate(
            zip(live_positions.tolist(), offsets, strict=True)
        ):
            detail_start = header_token_count + len(detail_indexes)
            token_indexes.extend(range(3 * i, 3 * i + 3))
            token_indexes.extend(range(detail_start, detail_start + count))
            lengths.append(3 + count)
            detail_indexes.extend(
                range(b * detail_rows + d, b * detail_rows + d + count)
            )
            member_indexes.extend(
                range(b * member_rows + m, b * member_rows + m + targets)
            )
            member_records.extend([i] * targets)
        detail_host = (
            details.detach().cpu().reshape(-1, DETAIL_DIM)[detail_indexes].long()
        )
        kinds, ids = detail_host[:, 0], detail_host[:, 1:]
        member_host = members.detach().cpu().reshape(-1, MEMBER_DIM)[member_indexes]
        targets = member_host[:, 0].long()
        batches = live_positions[:, 0]
        member_batches = batches[member_records]
        device = objects.device
        header_indexes = (batches * header_rows + live_positions[:, 1]).to(device)
        numeric = headers.reshape(-1, EFFECT_DIM)[header_indexes, 2:10].clone()
        numeric[:, -1] /= EFFECT_ROWS
        controller, source_object = join_record_rows(
            objects, batches[:, None], links
        ).unbind(1)
        header = self.header(torch.cat((numeric, controller, source_object), -1))
        modification = ((halves[:, 0] << 16) | halves[:, 1]).to(device)
        source = source.to(device)
        if catalog is None:
            mod = self.modification(modification % 4096)
            sources = self.source(source % 4096)
        else:
            mod = catalog.names(RecordNamespace.TOKEN, modification)
            sources = catalog.sources(source)
        unsigned = ((ids[:, 0] << 16) | ids[:, 1]).to(device)
        kinds = kinds.to(device)
        if catalog is None:
            detail = embedding_rows(self.detail, unsigned % 4096)
        else:
            # Kinds 1-3 name a keyword, subtype or color; 11-19 carry
            # a learning token of kind 1-9.
            program = kinds > 10
            detail = mod.new_zeros((len(kinds), mod.shape[-1]))
            detail[~program] = catalog.names(RecordNamespace.TOKEN, unsigned[~program])
            detail[program] = catalog.tokens(kinds[program] - 10, unsigned[program])
        detail_tokens = detail + embedding_rows(self.detail_kind, kinds)
        # Integer/boolean tokens retain signed numeric meaning;
        # the GRU retains the ordered shared program structure.
        signed = torch.where(unsigned >= 2**31, unsigned - 2**32, unsigned)
        is_number = ((kinds == 14) | (kinds == 15)).unsqueeze(-1).to(device)
        detail_tokens = detail_tokens + self.detail_number(
            program_number(signed.to(device=device, dtype=objects.dtype)).unsqueeze(-1)
        ).masked_fill(~is_number, 0)
        tokens = torch.cat(
            (
                torch.stack((mod, sources, torch.tanh(header)), 1).flatten(0, 1),
                detail_tokens,
            )
        )
        encoded = self.reducer(
            tokens[torch.tensor(token_indexes, device=device)], lengths
        )
        # Duplicate memberships remain separate scatter additions and gradients.
        routes = (member_batches * object_count + targets - 1).to(device)
        repeated = encoded[torch.tensor(member_records, device=device)]
        table = objects.reshape(-1, width)
        delta = torch.zeros_like(table).index_add(0, routes, repeated.to(table.dtype))
        return objects + delta.reshape_as(objects)


def admit_shared_effects(
    headers: torch.Tensor,
    details: torch.Tensor,
    members: torch.Tensor,
    object_count: int,
    cards: SemanticCardEncoder | None,
) -> None:
    """Reject malformed shared-effect buffers before any forward.

    Checks shapes, binary validity, one-based joins into `object_count`
    objects, contiguous CSR partitions, live detail kinds and unique membership
    order; padding rows are ignored, even when poisoned. With `cards` (catalog
    ids), modification, detail and source ids must be in its vocabulary.
    """
    host = admit_record_rows(headers.detach().cpu(), EFFECT_ROWS, EFFECT_DIM, 17)
    batch = host.shape[0]
    if details.shape != (batch, DETAIL_ROWS, DETAIL_DIM) or members.shape != (
        batch,
        MEMBER_ROWS,
        MEMBER_DIM,
    ):
        raise ValueError("invalid shared effect buffer shape")
    live_positions = host[..., 17].nonzero()
    if len(live_positions) == 0:
        return
    live = host[live_positions[:, 0], live_positions[:, 1]]
    halves = record_indexes(live[:, :2], 65535, "effect vocabulary half")
    source = record_indexes(live[:, 11], 2**24, "effect source definition")
    links = record_indexes(live[:, [10, 12]], object_count, "effect source/controller")
    if (links[:, 0] == 0).any():
        raise ValueError("effect controller is absent")
    offsets = record_indexes(
        live[:, 13:17], max(MEMBER_ROWS, DETAIL_ROWS), "effect CSR offset/count"
    ).tolist()
    detail_indexes: list[int] = []
    member_indexes: list[int] = []
    member_records: list[int] = []
    previous_batch = -1
    d_end = m_end = 0
    for i, ((b, _), (d, count, m, targets)) in enumerate(
        zip(live_positions.tolist(), offsets, strict=True)
    ):
        if b != previous_batch:
            d_end = m_end = 0
            previous_batch = b
        if (
            d != d_end
            or m != m_end
            or d + count > DETAIL_ROWS
            or targets == 0
            or m + targets > MEMBER_ROWS
        ):
            raise ValueError("invalid effect CSR partition")
        d_end, m_end = d + count, m + targets
        detail_indexes.extend(range(b * DETAIL_ROWS + d, b * DETAIL_ROWS + d_end))
        member_indexes.extend(range(b * MEMBER_ROWS + m, b * MEMBER_ROWS + m_end))
        member_records.extend([i] * targets)
    # Validate all live details/members together, ignoring poisoned padding.
    detail_host = details.detach().cpu().reshape(-1, DETAIL_DIM)[detail_indexes]
    kinds = record_indexes(detail_host[:, 0], 19, "effect detail kind")
    ids = record_indexes(detail_host[:, 1:], 65535, "effect detail half")
    if ((kinds == 0) | ((kinds > 3) & (kinds < 11))).any():
        raise ValueError("live effect detail has no kind")
    member_host = members.detach().cpu().reshape(-1, MEMBER_DIM)[member_indexes]
    record_indexes(member_host[:, 0], object_count, "effect member")
    orders = record_indexes(member_host[:, 1], MEMBER_ROWS, "effect membership order")
    if (member_host[:, 0] == 0).any():
        raise ValueError("effect membership is absent")
    if (orders == 0).any():
        raise ValueError("effect membership order is absent")
    global_orders = live_positions[:, 0][member_records] * (MEMBER_ROWS + 1) + orders
    if len(global_orders.unique()) != len(global_orders):
        raise ValueError("duplicate effect membership order")
    if cards is None:
        return
    cards.require_record_names(
        RecordNamespace.TOKEN, (halves[:, 0] << 16) | halves[:, 1]
    )
    lookup = buffer_on(cards, "definition_rows", source.device)
    admit_definition_ids(source, len(lookup), lookup)
    unsigned = (ids[:, 0] << 16) | ids[:, 1]
    program = kinds > 10
    cards.require_record_names(RecordNamespace.TOKEN, unsigned[~program])
    cards.require_record_tokens(kinds[program] - 10, unsigned[program])
