"""Shared primitives for typed observation records, independent of their schema.

Host preparation owns the sequence schedule. RecordReducer combines each
record independently with a GRU; no temporal state survives a forward.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from manabot.policy.prepare import Packed


def program_number(value: torch.Tensor) -> torch.Tensor:
    """Signed log magnitude for integer program tokens (the transport stays exact).

    Raw i32 values saturate tanh/GRU gates and erase differences between hash
    collisions. This monotone scale maps the signed-i32 range into about ±2.15;
    it retains zero and sign without letting large values dominate embeddings.
    """
    return value.sign() * value.abs().log1p() / 10


class RecordReducer(nn.Module):
    """Reduce nonempty ordered token lists; no hidden state crosses calls."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width)

    def forward(
        self, tokens: torch.Tensor, schedule: Packed[torch.Tensor]
    ) -> torch.Tensor:
        """Reduce prepared sequences without rediscovering their routes on device."""
        data = tokens[schedule.gather]
        _, hidden = self.gru(PackedSequence(data, torch.tensor(schedule.sizes)))
        pooled = hidden[-1][schedule.restore]
        return self.output(torch.tanh(pooled))
