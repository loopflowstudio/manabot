"""
One typed token sequence per observation, padded to a per-batch bucket.

The fixed object tables give every observation the same slot count per entity
type, so attention pays for each type's capacity whether or not it is filled.
A pooled stream instead concatenates every embedded entity row (players,
cards, permanents, continuation relations, legal actions), keeps the valid
ones in their original order, and pads the batch only to the smallest
`TOKEN_BUCKET` multiple at or above its longest observation. Slot variance is
then shared across types: a board-heavy observation and a hand-heavy one cost
the same total length.

`pack` and `unpack` are exact inverses on valid tokens: unpack scatters each
packed position back to its original coordinate and leaves every invalid
coordinate zero, which is what masked attention layers already write there.
Packing reads the batch's longest valid count on the host once per forward; a
batch therefore costs one synchronization and takes one of at most
`ceil(T / TOKEN_BUCKET)` sequence shapes.
"""

import torch

# Sequence lengths are multiples of this; it bounds how many shapes a run sees.
TOKEN_BUCKET = 32

# Token types, embedded once per type in `ActorCritic.token_types`.
PLAYER, CARD, PERMANENT, RELATION, ACTION = range(5)
TOKEN_TYPES = 5


def bucket_length(longest: int, slots: int, multiple: int = TOKEN_BUCKET) -> int:
    """The padded length for a batch whose longest observation has `longest` tokens."""
    return min(slots, max(multiple, -(-longest // multiple) * multiple))


def pack(
    tokens: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move each row's valid tokens [B,T,D] to its front, in order, and trim.

    Returns packed tokens [B,L,D], their validity [B,L] and `order` [B,L], the
    original coordinate of each packed position (positions past a row's count
    hold its invalid coordinates, so `order` stays a partial permutation).
    """
    if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
        raise ValueError("token validity must be boolean [batch, tokens]")
    length = bucket_length(int(valid.sum(1).max()), tokens.shape[1])
    # Stable sort of the invalid flag puts valid tokens first in table order.
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)[:, :length]
    packed = tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return packed, valid.gather(1, order), order


def unpack(packed: torch.Tensor, order: torch.Tensor, slots: int) -> torch.Tensor:
    """Scatter packed tokens [B,L,D] back to [B,slots,D]; unlisted slots are zero."""
    return packed.new_zeros((packed.shape[0], slots, packed.shape[-1])).scatter(
        1, order.unsqueeze(-1).expand_as(packed), packed
    )
