"""Shared primitives for typed observation records, independent of their schema.

Admission validates live rows and one-based joins (`admit_record_rows`,
`record_indexes`); the forward only masks padding before indexing.
RecordReducer combines within-observation tokens with a GRU without retaining
temporal state.
"""

from itertools import accumulate

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence


def program_number(value: torch.Tensor) -> torch.Tensor:
    """Signed log magnitude for integer program tokens (the transport stays exact).

    Raw i32 values saturate tanh/GRU gates and erase differences between hash
    collisions. This monotone scale maps the signed-i32 range into about ±2.15;
    it retains zero and sign without letting large values dominate embeddings.
    """
    return value.sign() * value.abs().log1p() / 10


def masked_record_rows(value: torch.Tensor, valid_column: int) -> torch.Tensor:
    """Zero padding rows before any arithmetic or indexing."""
    return torch.where(value[..., valid_column].bool().unsqueeze(-1), value, 0)


def admit_record_rows(
    value: torch.Tensor, count: int, width: int, valid_column: int
) -> torch.Tensor:
    """Reject malformed live rows; return them with padding zeroed."""
    if value.ndim != 3 or value.shape[1:] != (count, width):
        raise ValueError(f"expected [batch, {count}, {width}] state records")
    valid = value[..., valid_column]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("state record validity must be binary")
    clean = masked_record_rows(value, valid_column)
    if not torch.isfinite(clean).all():
        raise ValueError("state records must be finite")
    return clean


def record_indexes(values: torch.Tensor, maximum: int, name: str) -> torch.Tensor:
    """Admit integral references in [0, maximum]; return them as int64."""
    if not ((values == values.floor()) & (values >= 0) & (values <= maximum)).all():
        raise ValueError(f"invalid {name} reference")
    return values.long()


def join_record_rows(
    values: torch.Tensor, batches: torch.Tensor, references: torch.Tensor
) -> torch.Tensor:
    """Join [batch, row, width] values through validated host-owned references.

    Batches broadcast over one-based references. Zero stays absent
    even in later batches; only routing moves devices, never learned values.
    """
    indexes = torch.where(references > 0, batches * values.shape[1] + references, 0).to(
        values.device
    )
    return values.flatten(0, 1)[(indexes - 1).clamp_min(0)].masked_fill(
        (indexes == 0).unsqueeze(-1), 0
    )


class RecordReducer(nn.Module):
    """Reduce nonempty ordered token lists; no hidden state crosses calls."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width)

    def forward(self, tokens: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Reduce a flat sequence table with host-owned, positive record lengths.

        Packing indexes are built on the host once. No per-record device copy,
        scalar read or padded CopySlices autograd chain is needed. GRU order and
        parameters are identical to packing each original sequence separately.
        """
        if not lengths or min(lengths) < 1 or sum(lengths) != len(tokens):
            raise ValueError("record lengths must partition nonempty token sequences")
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        offsets = list(accumulate(lengths, initial=0))
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
