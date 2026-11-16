"""Stateless full-prefix memory encoders and their shared record featurizer.

Policy/value gradients flow through the J1 board summaries into both the record
featurizer and temporal encoder. Every call starts from the supplied prefix;
neither encoder retains a learned state across decisions or weight updates.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from manabot.policy.opponent_memory import MemoryRows


class OpponentMemory(nn.Module):
    def __init__(
        self, encoder: Literal["transformer", "gru"], summaries: int, board_width: int
    ) -> None:
        super().__init__()
        self.summaries = summaries
        self.encoder = encoder
        self.record_numbers = nn.Linear(8, 64)
        self.detail_numbers = nn.Linear(4, 64)
        self.tokens = nn.Linear(board_width, 64, bias=False)
        self.record_output = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0, batch_first=True)
            self.transformer = nn.TransformerEncoder(
                layer, 2, enable_nested_tensor=False
            )
            self.queries = nn.Parameter(torch.randn(1, summaries, 64) / 8)
            self.gru = None
            self.output = nn.Linear(64, board_width)
        else:
            self.transformer = None
            self.queries = None
            self.gru = nn.GRU(64, 64, batch_first=True)
            self.output = nn.Linear(64, summaries * board_width)
        self.memory_type = nn.Parameter(
            torch.randn(1, summaries, board_width) / board_width**0.5
        )

    def forward(
        self,
        rows: MemoryRows[torch.Tensor],
        embedding: nn.Embedding,
        definitions: torch.Tensor,
        names: nn.Embedding,
    ) -> torch.Tensor:
        batch = len(rows.lengths)
        board_width = self.memory_type.shape[-1]
        # Empty histories bypass recurrence/attention and produce exactly zero,
        # including the type embedding, in both encoders.
        if not rows.schedule.sizes:
            return self.memory_type.new_zeros((batch, self.summaries, board_width))
        details = self.tokens(
            embedding(rows.detail_names) + embedding(rows.detail_values)
        )
        details = (
            details
            + self.detail_numbers(rows.detail_numbers)
            + self.tokens(
                definitions[rows.detail_definitions] + names(rows.detail_name_rows)
            )
        )
        records = self.record_numbers(rows.numbers) + self.tokens(embedding(rows.kinds))
        counts = records.new_zeros((len(records), 1)).index_add(
            0, rows.detail_records, details.new_ones((len(details), 1))
        )
        pooled = records.new_zeros(records.shape).index_add(
            0, rows.detail_records, details
        )
        records = self.record_output(records + pooled / counts.clamp_min(1).sqrt())
        if self.gru is not None:
            packed = PackedSequence(
                records[rows.schedule.gather], torch.tensor(rows.schedule.sizes)
            )
            _, hidden = self.gru(packed)
            summaries = self.output(hidden[-1][rows.schedule.restore]).reshape(
                -1, self.summaries, board_width
            )
            result = records.new_zeros((batch, self.summaries, board_width))
            return result.index_copy(
                0, rows.nonempty_batches, summaries + self.memory_type
            )
        assert self.transformer is not None and self.queries is not None
        sequence = records.new_zeros((batch, rows.width, 64))
        sequence = sequence.index_put((rows.record_batches, rows.record_slots), records)
        present = torch.zeros(
            (batch, rows.width), dtype=torch.bool, device=records.device
        )
        present = present.index_put(
            (rows.record_batches, rows.record_slots),
            torch.ones_like(rows.record_batches, dtype=torch.bool),
        )
        sequence = torch.cat((sequence, self.queries.expand(batch, -1, -1)), 1)
        padding = torch.cat(
            (
                ~present,
                torch.zeros(
                    (batch, self.summaries), dtype=torch.bool, device=records.device
                ),
            ),
            1,
        )
        summaries = (
            self.output(
                self.transformer(sequence, src_key_padding_mask=padding)[
                    :, -self.summaries :
                ]
            )
            + self.memory_type
        )
        nonempty = present.any(1).to(summaries.dtype)
        return summaries * nonempty[:, None, None]
