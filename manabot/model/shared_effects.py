"""Encode v7 shared payloads once, then route their gradients into affected objects.

Membership is an explicit join, never learned from object IDs. We retain the
existing unnormalized fact-sum convention: repeated memberships contribute
repeatedly. A mean would erase multiplicity. This learned sum is compression,
not an injective semantic representation; tensor transport retains the records.
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
from manabot.model.records import (
    RecordReducer,
    RecordReduction,
    join_record_rows,
    masked_record_rows,
    record_indexes,
)


class SharedEffectEncoder(nn.Module):
    def __init__(self, width: int, reduction: RecordReduction = "typed_pool") -> None:
        super().__init__()
        self.modification = nn.Embedding(4096, width)
        self.detail = nn.Embedding(4096, width)
        self.detail_kind = nn.Embedding(4, width)
        self.source = nn.Embedding(4096, width)
        self.header = nn.Linear(8 + 2 * width, width)
        self.reducer = RecordReducer(width, reduction)

    def forward(
        self,
        headers: torch.Tensor,
        details: torch.Tensor,
        members: torch.Tensor,
        objects: torch.Tensor,
    ) -> torch.Tensor:
        # One metadata transfer admits the entire batch. Numerical features and
        # object joins below still read the original tensors, retaining gradients.
        host = masked_record_rows(headers.detach().cpu(), EFFECT_ROWS, EFFECT_DIM, 17)
        batch, object_count, width = objects.shape
        if (
            host.shape[0] != batch
            or details.shape != (batch, DETAIL_ROWS, DETAIL_DIM)
            or members.shape != (batch, MEMBER_ROWS, MEMBER_DIM)
        ):
            raise ValueError("invalid shared effect buffer shape")
        live_positions = host[..., 17].nonzero()
        if len(live_positions) == 0:
            return objects
        live = host[live_positions[:, 0], live_positions[:, 1]]
        halves = record_indexes(live[:, :2], 65535, "effect vocabulary half")
        source = record_indexes(live[:, 11], 2**24, "effect source definition")
        links = record_indexes(
            live[:, [10, 12]], object_count, "effect source/controller"
        )
        if (links[:, 0] == 0).any():
            raise ValueError("effect controller is absent")
        offsets = record_indexes(
            live[:, 13:17], MEMBER_ROWS, "effect CSR offset/count"
        ).tolist()
        detail_indexes: list[int] = []
        member_indexes: list[int] = []
        member_records: list[int] = []
        token_indexes: list[int] = []
        lengths: list[int] = []
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
            token_indexes.extend((3 * i, 3 * i + 1, 3 * i + 2))
            token_indexes.extend(
                range(
                    3 * len(live) + len(detail_indexes),
                    3 * len(live) + len(detail_indexes) + count,
                )
            )
            lengths.append(3 + count)
            detail_indexes.extend(range(b * DETAIL_ROWS + d, b * DETAIL_ROWS + d_end))
            member_indexes.extend(range(b * MEMBER_ROWS + m, b * MEMBER_ROWS + m_end))
            member_records.extend([i] * targets)
        # Validate all live details/members together, ignoring poisoned padding.
        detail_host = details.detach().cpu().reshape(-1, DETAIL_DIM)[detail_indexes]
        kinds = record_indexes(detail_host[:, 0], 3, "effect detail kind")
        ids = record_indexes(detail_host[:, 1:], 65535, "effect detail half")
        if (kinds == 0).any():
            raise ValueError("live effect detail has no kind")
        member_host = members.detach().cpu().reshape(-1, MEMBER_DIM)[member_indexes]
        targets = record_indexes(member_host[:, 0], object_count, "effect member")
        orders = record_indexes(
            member_host[:, 1], MEMBER_ROWS, "effect membership order"
        )
        if (targets == 0).any():
            raise ValueError("effect membership is absent")
        if (orders == 0).any():
            raise ValueError("effect membership order is absent")
        batches = live_positions[:, 0]
        member_batches = batches[member_records]
        global_orders = member_batches * (MEMBER_ROWS + 1) + orders
        if len(global_orders.unique()) != len(global_orders):
            raise ValueError("duplicate effect membership order")
        device = objects.device
        header_indexes = (batches * EFFECT_ROWS + live_positions[:, 1]).to(device)
        numeric = headers.reshape(-1, EFFECT_DIM)[header_indexes, 2:10].clone()
        numeric[:, -1] /= EFFECT_ROWS
        controller, source_object = join_record_rows(
            objects, batches[:, None], links
        ).unbind(1)
        header = self.header(torch.cat((numeric, controller, source_object), -1))
        mod = self.modification(
            (((halves[:, 0] << 16) | halves[:, 1]) % 4096).to(device)
        )
        sources = self.source((source % 4096).to(device))
        detail_tokens = self.detail(
            (((ids[:, 0] << 16) | ids[:, 1]) % 4096).to(device)
        ) + self.detail_kind(kinds.to(device))
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
        delta = torch.zeros_like(table).index_add(0, routes, repeated)
        return objects + delta.reshape_as(objects)
