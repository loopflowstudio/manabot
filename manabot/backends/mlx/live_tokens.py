"""Compact live object coordinates for MLX's batched scaled dot-product attention."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass(frozen=True)
class LiveTokens:
    index: mx.array  # packed -> original flat object coordinate
    batches: mx.array
    slots: mx.array  # packed -> compact within-sequence coordinate
    mask: mx.array  # [B,1,1,max_live] additive mask, zero at valid keys

    @classmethod
    def from_offsets(cls, index: np.ndarray, offsets: np.ndarray) -> "LiveTokens":
        lengths = np.diff(offsets)
        batches = np.repeat(np.arange(len(lengths)), lengths)
        slots = np.arange(len(index)) - offsets[batches]
        keys = np.arange(lengths.max())[None, :] < lengths[:, None]
        mask = np.where(keys, np.float32(0), np.float32(-np.inf))
        return cls(
            mx.array(index),
            mx.array(batches),
            mx.array(slots),
            mx.array(mask[:, None, None, :]),
        )

    def attend(self, mha: nn.MultiHeadAttention, packed: mx.array) -> mx.array:
        """Only Q/K/V and SDPA use [B,H,max_live,d]; project real tokens once."""
        batch, _, _, longest = self.mask.shape
        width = packed.shape[-1]

        def compact(projected: mx.array) -> mx.array:
            padded = mx.zeros((batch, longest, width), projected.dtype)
            padded[self.batches, self.slots] = projected
            return padded.reshape(batch, longest, mha.num_heads, -1).transpose(
                0, 2, 1, 3
            )

        queries = compact(mha.query_proj(packed))
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            compact(mha.key_proj(packed)),
            compact(mha.value_proj(packed)),
            scale=queries.shape[-1] ** -0.5,
            mask=self.mask.astype(packed.dtype),
        )
        rows = attended.transpose(0, 2, 1, 3).reshape(batch, longest, width)
        return mha.out_proj(rows[self.batches, self.slots])
