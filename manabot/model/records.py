"""Shared primitives for typed observation records, independent of their schema.

Validate live rows and one-based joins before indexing; RecordReducer combines
within-observation tokens by sum or GRU without retaining temporal state.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

RecordReduction = Literal["typed_pool", "typed_gru"]


def masked_record_rows(
    value: torch.Tensor, count: int, width: int, valid_column: int
) -> torch.Tensor:
    """Mask padding before any arithmetic or indexing; reject malformed live rows."""
    if value.ndim != 3 or value.shape[1:] != (count, width):
        raise ValueError(f"expected [batch, {count}, {width}] state records")
    valid = value[..., valid_column]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("state record validity must be binary")
    clean = torch.where(valid.bool().unsqueeze(-1), value, 0)
    if not torch.isfinite(clean).all():
        raise ValueError("state records must be finite")
    return clean


def record_indexes(values: torch.Tensor, maximum: int, name: str) -> torch.Tensor:
    if not ((values == values.floor()) & (values >= 0) & (values <= maximum)).all():
        raise ValueError(f"invalid {name} reference")
    return values.long()


def join_rows(values: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """One-based local references; zero is explicit absence, never object zero."""
    return values[(indexes - 1).clamp_min(0)].masked_fill(
        (indexes == 0).unsqueeze(-1), 0
    )


class RecordReducer(nn.Module):
    """Reduce nonempty ordered token lists; no hidden state crosses calls."""

    def __init__(self, width: int, reduction: RecordReduction) -> None:
        super().__init__()
        self.gru = (
            nn.GRU(width, width, batch_first=True) if reduction == "typed_gru" else None
        )
        self.output = nn.Linear(width, width)

    def forward(self, records: list[torch.Tensor]) -> torch.Tensor:
        return self.reduce_tokens(
            torch.cat(records), [len(record) for record in records]
        )

    def reduce_tokens(self, tokens: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Reduce a flat sequence table with host-owned, positive record lengths.

        Packing indexes are built on the host once. No per-record device copy,
        scalar read or padded CopySlices autograd chain is needed. GRU order and
        parameters are identical to packing each original sequence separately.
        """
        if not lengths or min(lengths) < 1 or sum(lengths) != len(tokens):
            raise ValueError("record lengths must partition nonempty token sequences")
        if self.gru is None:
            pooled = torch.segment_reduce(
                tokens, "sum", lengths=torch.tensor(lengths, device=tokens.device)
            )
        else:
            order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
            offsets: list[int] = []
            offset = 0
            for length in lengths:
                offsets.append(offset)
                offset += length
            indexes: list[int] = []
            sizes: list[int] = []
            for step in range(max(lengths)):
                live = [i for i in order if lengths[i] > step]
                sizes.append(len(live))
                indexes.extend(offsets[i] + step for i in live)
            data = tokens[torch.tensor(indexes, device=tokens.device)]
            _, hidden = self.gru(PackedSequence(data, torch.tensor(sizes)))
            inverse = sorted(range(len(order)), key=order.__getitem__)
            pooled = hidden[-1][torch.tensor(inverse, device=tokens.device)]
        return self.output(torch.tanh(pooled))
