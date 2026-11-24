"""Encode prepared decision records and their joined public references into board summaries.

GRU folds share differentiable ancestor states in learning. Serving folds can
start from frozen roots and retain intermediate states for subsequent decisions.
"""

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from manabot.backends.mlx.gru import PackedGRU
from manabot.policy.decision_memory import MemoryRows


class MemoryLayer(nn.Module):
    """Torch's pre-norm TransformerEncoderLayer with ReLU and zero dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(64, 4, bias=True)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)

    def __call__(self, sequence: mx.array, mask: mx.array) -> mx.array:
        x = self.norm1(sequence)
        sequence = sequence + self.self_attn(x, x, x, mask)
        return sequence + self.linear2(nn.relu(self.linear1(self.norm2(sequence))))


def pool(values: mx.array, groups: np.ndarray, count: int) -> tuple[mx.array, mx.array]:
    """Pool joined rows with square-root count normalization."""
    pooled = mx.zeros((count, 64), values.dtype)
    counts = mx.zeros((count, 1), values.dtype)
    if len(groups):
        routes = mx.array(groups)
        pooled = pooled.at[routes].add(values)
        counts = counts.at[routes].add(mx.ones((len(groups), 1), values.dtype))
    return pooled * mx.rsqrt(mx.maximum(counts, 1)), counts > 0


class DecisionMemory(nn.Module):
    def __init__(
        self, encoder: Literal["transformer", "gru"], summaries: int, width: int
    ) -> None:
        super().__init__()
        self.summaries = summaries

        def feature_map(inputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
            )

        self.detail_map = feature_map(2 * width + 5)
        self.reference_map = feature_map(width + 64 + 8)
        self.member_map = feature_map(65)
        self.record_output = nn.Sequential(
            nn.Linear(38 + 3 * width + 3 * 65, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.join_norm = nn.LayerNorm(width)
        self.memory_type = mx.random.normal((1, summaries, width)) * width**-0.5
        if encoder == "gru":
            self.gru = PackedGRU(64, 64)
            self.output = nn.Linear(64, summaries * width)
        else:
            self.transformer = {"layers": [MemoryLayer(), MemoryLayer()]}
            self.queries = mx.random.normal((1, summaries, 64)) / 8
            self.recency = nn.Linear(1, 64, bias=False)
            self.output = nn.Linear(64, width)

    def __call__(
        self,
        rows: MemoryRows[np.ndarray],
        tokens: mx.array,
        definitions: mx.array,
        names: mx.array,
    ) -> mx.array:
        batch = len(rows.lengths)
        width = self.memory_type.shape[-1]
        dtype = tokens.dtype
        result = mx.zeros((batch, self.summaries, width), dtype)
        if not rows.schedule.sizes and (rows.fold is None or not rows.fold.node_count):
            return result

        # Empty joins also occur when all referenced snapshots belong to cached roots.
        details = mx.zeros((0, 64), dtype)
        if len(rows.detail_fields):
            details = self.detail_map(
                mx.concatenate(
                    (
                        tokens[mx.array(rows.detail_fields)],
                        tokens[mx.array(rows.detail_values)],
                        mx.array(rows.detail_numbers).astype(dtype),
                    ),
                    -1,
                )
            )
        pooled, _ = pool(details, rows.detail_references, len(rows.reference_numbers))
        refs = mx.zeros((0, 64), dtype)
        if len(rows.reference_numbers):
            refs = self.reference_map(
                mx.concatenate(
                    (
                        definitions[mx.array(rows.reference_definitions)],
                        mx.array(rows.reference_numbers).astype(dtype),
                        pooled,
                    ),
                    -1,
                )
            )
        fold = rows.fold
        if fold is not None and fold.root_references is not None:
            refs = mx.concatenate((fold.root_references, refs))
        members = mx.zeros((0, 64), dtype)
        if len(rows.member_references):
            members = self.member_map(
                mx.concatenate(
                    (
                        refs[mx.array(rows.member_references)],
                        mx.array(rows.member_positions).astype(dtype),
                    ),
                    -1,
                )
            )
        count = len(rows.numbers)
        grouped, present = pool(members, rows.member_groups, count * 3)
        names = mx.concatenate((mx.stop_gradient(names[:1]), names[1:]))
        records = self.record_output(
            mx.concatenate(
                (
                    mx.array(rows.numbers).astype(dtype),
                    tokens[mx.array(rows.kinds)],
                    tokens[mx.array(rows.choices)],
                    names[mx.array(rows.choice_names)],
                    mx.concatenate((grouped, present.astype(dtype)), -1).reshape(
                        count, 3 * 65
                    ),
                ),
                -1,
            )
        )

        if "gru" in self:
            if fold is None:
                _, hidden = self.gru.run(
                    records, mx.array(rows.schedule.gather), rows.schedule.sizes
                )
                hidden = hidden[mx.array(rows.schedule.restore)]
                result[mx.array(rows.nonempty_batches)] = self.join_norm(
                    self.output(hidden).reshape(-1, self.summaries, width)
                    + self.memory_type
                )
                return result
            states = mx.zeros((fold.node_count + 1, 64), dtype)
            if fold.root_states is not None:
                states[mx.array(fold.root_nodes)] = fold.root_states
            for level in fold.levels:
                every, final = self.gru.run(
                    records,
                    mx.array(level.records),
                    level.schedule.sizes,
                    states[mx.array(level.parents)],
                )
                if level.record_nodes is None:
                    states[mx.array(level.nodes)] = final[
                        mx.array(level.schedule.restore)
                    ]
                else:
                    states[mx.array(level.record_nodes)] = every
            if fold.retain is not None:
                fold.retain(mx.stop_gradient(states), mx.stop_gradient(refs))
            tails = mx.array(fold.tails)
            hidden = self.output(states[tails]).reshape(batch, self.summaries, width)
            return (
                self.join_norm(hidden + self.memory_type) * (tails != 0)[:, None, None]
            )

        lengths = np.asarray(rows.lengths)
        ages = lengths[rows.record_batches] - 1 - rows.record_slots
        records = records + self.recency(mx.array(ages[:, None]).astype(dtype) / 128)
        sequence = mx.zeros((batch, rows.width, 64), dtype)
        sequence[mx.array(rows.record_batches), mx.array(rows.record_slots)] = records
        sequence = mx.concatenate(
            (sequence, mx.broadcast_to(self.queries, (batch, self.summaries, 64))),
            axis=1,
        )
        present = np.arange(rows.width) < lengths[:, None]
        mask = mx.array(
            np.concatenate((present, np.ones((batch, self.summaries), bool)), 1)
        )[:, None, None, :]
        for layer in self.transformer["layers"]:
            sequence = layer(sequence, mask)
        summaries = self.output(sequence[:, -self.summaries :]) + self.memory_type
        return self.join_norm(summaries) * mx.array(lengths > 0)[:, None, None]
