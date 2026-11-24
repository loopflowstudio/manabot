"""MLX attention on packed objects using the prepared compact coordinates."""

import mlx.core as mx
import mlx.nn as nn

from manabot.policy.prepare import CompactAttention


def attend_live(
    mha: nn.MultiHeadAttention, packed: mx.array, live: CompactAttention[mx.array]
) -> mx.array:
    """Project real tokens once and compact Q/K/V to [B,H,max_live,d] for SDPA."""
    batch, _, _, longest = live.key_mask.shape
    width = packed.shape[-1]

    def compact(projected: mx.array) -> mx.array:
        padded = mx.zeros((batch, longest, width), projected.dtype)
        padded[live.batches, live.slots] = projected
        return padded.reshape(batch, longest, mha.num_heads, -1).transpose(0, 2, 1, 3)

    queries = compact(mha.query_proj(packed))
    attended = mx.fast.scaled_dot_product_attention(
        queries,
        compact(mha.key_proj(packed)),
        compact(mha.value_proj(packed)),
        scale=queries.shape[-1] ** -0.5,
        mask=live.key_mask,
    )
    rows = attended.transpose(0, 2, 1, 3).reshape(batch, longest, width)
    return mha.out_proj(rows[live.batches, live.slots])
