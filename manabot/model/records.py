"""Shared primitives for typed observation records, independent of their schema.

Validate live rows and one-based joins before indexing; RecordReducer combines
within-observation tokens by sum or GRU without retaining temporal state.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

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
        if self.gru is None:
            pooled = torch.stack([record.sum(0) for record in records])
        else:
            # Embedding happens before packing only for actual tokens, not a
            # batch-wide padded record array. Gradients reach every input token.
            _, hidden = self.gru(pack_sequence(records, enforce_sorted=False))
            pooled = hidden[-1]
        return self.output(torch.tanh(pooled))
