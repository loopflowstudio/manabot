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
from manabot.model.state_records import (
    RecordReducer,
    RecordReduction,
    _indexes,
    _join,
    _rows,
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
        headers = _rows(headers, EFFECT_ROWS, EFFECT_DIM, 17)
        batch = len(objects)
        if (
            headers.shape[0] != batch
            or details.shape != (batch, DETAIL_ROWS, DETAIL_DIM)
            or members.shape != (batch, MEMBER_ROWS, MEMBER_DIM)
        ):
            raise ValueError("invalid shared effect buffer shape")
        results = list(objects.unbind(0))
        for b in headers[..., 17].bool().any(-1).nonzero().flatten().tolist():
            live = headers[b, headers[b, :, 17].bool()]
            halves = _indexes(live[:, :2], 65535, "effect vocabulary half")
            source = _indexes(live[:, 11], 2**24, "effect source definition")
            links = _indexes(
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
                        _join(objects[b], links[:, 0]),
                        _join(objects[b], links[:, 1]),
                    ),
                    -1,
                )
            )
            mod = self.modification(((halves[:, 0] << 16) | halves[:, 1]) % 4096)
            sources = self.source(source % 4096)
            offsets = _indexes(
                live[:, 13:17], MEMBER_ROWS, "effect CSR offset/count"
            ).tolist()
            records: list[torch.Tensor] = []
            routes: list[torch.Tensor] = []
            d_end = m_end = 0
            member_orders: list[torch.Tensor] = []
            for i, (d, count, m, targets) in enumerate(offsets):
                if (
                    d != d_end
                    or m != m_end
                    or d + count > DETAIL_ROWS
                    or targets == 0
                    or m + targets > MEMBER_ROWS
                ):
                    raise ValueError("invalid effect CSR partition")
                d_end, m_end = d + count, m + targets
                tokens = details[b, d:d_end]
                kinds = _indexes(tokens[:, 0], 3, "effect detail kind")
                ids = _indexes(tokens[:, 1:], 65535, "effect detail half")
                if (kinds == 0).any():
                    raise ValueError("live effect detail has no kind")
                encoded_details = self.detail(
                    ((ids[:, 0] << 16) | ids[:, 1]) % 4096
                ) + self.detail_kind(kinds)
                records.append(
                    torch.cat(
                        (
                            torch.stack((mod[i], sources[i], torch.tanh(header[i]))),
                            encoded_details,
                        )
                    )
                )
                memberships = members[b, m:m_end]
                targets_index = _indexes(
                    memberships[:, 0], objects.shape[1], "effect member"
                )
                if (targets_index == 0).any():
                    raise ValueError("effect membership is absent")
                orders = _indexes(
                    memberships[:, 1], MEMBER_ROWS, "effect membership order"
                )
                if (orders == 0).any():
                    raise ValueError("effect membership order is absent")
                member_orders.append(orders)
                routes.append(targets_index - 1)
            orders = torch.cat(member_orders)
            if len(orders.unique()) != len(orders):
                raise ValueError("duplicate effect membership order")
            encoded = self.reducer(records)
            # index_add accumulates duplicate memberships and all target losses
            # flow through the same encoded payload and shared encoder parameters.
            repeated = torch.cat(
                [encoded[i].expand(len(route), -1) for i, route in enumerate(routes)]
            )
            delta = torch.zeros_like(objects[b]).index_add(
                0, torch.cat(routes), repeated
            )
            results[b] = objects[b] + delta
        return torch.stack(results)
